//! Schema-driven mechanical conformance — fixes the error classes the model
//! repeats every run before the repair loop spends tokens on them:
//!
//! * bare scalars where a WithSource object is required → wrapped
//! * missing required keys → type-appropriate defaults (null when nullable)
//! * string→number coercion on money/amount fields
//! * enum casing / separator normalisation ("Senior Debt" → senior_debt)
//!
//! Runs `conform` after assemble and after every repair round; only the
//! residual issues reach the LLM.

use serde_json::{Map, Value};

use crate::contract::OUTPUT_SCHEMA;

/// Apply mechanical fixes in place. Returns the number of fixes applied.
pub fn conform_document(doc: &mut Value) -> usize {
    let schema: Value = match serde_json::from_str(OUTPUT_SCHEMA) {
        Ok(s) => s,
        Err(_) => return 0,
    };
    let mut fixes = 0usize;
    conform_node(doc, &schema, &mut fixes);
    fixes
}

/// The non-null member of a `"type": ["object","null"]` union, or the single
/// type string. Returns None for non-object-typed schemas.
fn primary_type<'a>(schema: &'a Value) -> Option<&'a str> {
    match schema.get("type") {
        Some(Value::String(t)) => Some(t.as_str()),
        Some(Value::Array(types)) => types
            .iter()
            .filter_map(|t| t.as_str())
            .find(|t| *t != "null"),
        _ => None,
    }
}

fn nullable(schema: &Value) -> bool {
    matches!(schema.get("type"), Some(Value::Array(t)) if t.iter().any(|v| v.as_str() == Some("null")))
}

/// WithSource shape: object schema whose properties include value+source.
fn is_with_source(schema: &Value) -> bool {
    schema
        .get("properties")
        .and_then(Value::as_object)
        .map(|p| p.contains_key("value") && p.contains_key("source"))
        .unwrap_or(false)
}

/// Type-appropriate default for a required key that the model omitted.
fn default_for(schema: &Value) -> Value {
    if nullable(schema) {
        return Value::Null;
    }
    match primary_type(schema) {
        Some("string") => Value::String(String::new()),
        Some("number") | Some("integer") => Value::from(0),
        Some("boolean") => Value::Bool(false),
        Some("array") => Value::Array(Vec::new()),
        Some("object") => Value::Object(Map::new()),
        _ => Value::Null,
    }
}

const WITH_SOURCE_DEFAULTS: &[(&str, Value)] = &[];

fn with_source_wrapper(value: Value) -> Value {
    let _ = WITH_SOURCE_DEFAULTS;
    let mut o = Map::new();
    o.insert("source".to_string(), Value::String(String::new()));
    o.insert(
        "source_type".to_string(),
        Value::String("extracted".to_string()),
    );
    o.insert("notes".to_string(), Value::Null);
    o.insert("scope".to_string(), Value::Null);
    o.insert("evidence_text".to_string(), Value::Null);
    o.insert("value".to_string(), value);
    Value::Object(o)
}

/// Strip currency symbols / thousands separators / whitespace and parse.
fn parse_amount(s: &str) -> Option<f64> {
    let cleaned: String = s
        .chars()
        .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
        .collect();
    cleaned.parse::<f64>().ok().filter(|v| v.is_finite())
}

/// Case/separator-insensitive enum match ("Senior Debt" → "senior_debt").
fn enum_fix(schema: &Value, v: &str) -> Option<String> {
    let members = schema.get("enum")?.as_array()?;
    let norm = |x: &str| x.to_lowercase().replace([' ', '-'], "_");
    let nv = norm(v);
    members
        .iter()
        .filter_map(|m| m.as_str())
        .find(|m| norm(m) == nv)
        .map(str::to_string)
}

fn conform_node(inst: &mut Value, schema: &Value, fixes: &mut usize) {
    let ty = primary_type(schema);
    match ty {
        Some("object") => conform_object(inst, schema, fixes),
        Some("array") => {
            if let Some(items) = schema.get("items") {
                if let Value::Array(arr) = inst {
                    for item in arr.iter_mut() {
                        conform_node(item, items, fixes);
                    }
                }
            }
        }
        Some("string") => {
            match inst.clone() {
                Value::Number(n) => {
                    *inst = Value::String(n.to_string());
                    *fixes += 1;
                }
                Value::Bool(b) => {
                    *inst = Value::String(b.to_string());
                    *fixes += 1;
                }
                Value::String(s) => {
                    if let Some(fixed) = enum_fix(schema, &s) {
                        if fixed != s {
                            *inst = Value::String(fixed);
                            *fixes += 1;
                            return;
                        }
                    }
                    // Decimal-string fields: strip £/commas/whitespace so
                    // "£25,750,000" satisfies the exact-decimal pattern.
                    if schema.get("pattern").is_some() {
                        let cleaned: String = s
                            .chars()
                            .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
                            .collect();
                        if !cleaned.is_empty() && cleaned != s && cleaned.parse::<f64>().is_ok() {
                            *inst = Value::String(cleaned);
                            *fixes += 1;
                        }
                    }
                }
                _ => {}
            }
        }
        Some("number") | Some("integer") => {
            if let Value::String(s) = inst.clone() {
                if let Some(n) = parse_amount(&s) {
                    *inst = Value::from(n);
                    *fixes += 1;
                }
            }
        }
        _ => {}
    }
}

fn conform_object(inst: &mut Value, schema: &Value, fixes: &mut usize) {
    // Scalar/array where an object is required → wrap if it's a WithSource.
    if !inst.is_object() {
        if is_with_source(schema) && !inst.is_null() {
            let inner = std::mem::replace(inst, Value::Null);
            *inst = with_source_wrapper(inner);
            *fixes += 1;
        } else {
            return;
        }
    }
    let Some(props) = schema.get("properties").and_then(Value::as_object) else {
        return;
    };
    let required: Vec<String> = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|r| {
            r.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();

    let Some(obj) = inst.as_object_mut() else {
        return;
    };

    // Fill missing required keys.
    for key in &required {
        if !obj.contains_key(key) {
            let ps = props.get(key).cloned().unwrap_or(Value::Null);
            obj.insert(key.clone(), default_for(&ps));
            *fixes += 1;
        }
    }
    // Recurse into present properties.
    for (key, prop_schema) in props {
        if let Some(v) = obj.get_mut(key) {
            conform_node(v, prop_schema, fixes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn wraps_bare_scalar_in_with_source() {
        let mut doc = json!({
            "canonical_deal": {}, "details": { "exit_plan": "sell units on completion" }, "kpis": {}, "classification": {}
        });
        let n = conform_document(&mut doc);
        assert!(n > 0);
        let t = &doc["details"]["exit_plan"];
        assert_eq!(t["value"], "sell units on completion");
        assert_eq!(t["source_type"], "extracted");
    }

    #[test]
    fn fills_missing_required_keys() {
        let mut doc = json!({
            "canonical_deal": {}, "details": { "exit_plan": {"value": "sell units on completion"} }, "kpis": {}, "classification": {}
        });
        conform_document(&mut doc);
        let t = &doc["details"]["exit_plan"];
        for k in [
            "source",
            "source_type",
            "notes",
            "scope",
            "evidence_text",
            "value",
        ] {
            assert!(t.get(k).is_some(), "missing {k}");
        }
    }

    #[test]
    fn normalises_enum_casing() {
        let mut doc = json!({
            "canonical_deal": {"facilities": [{"id": "f1", "name": "x", "instrument_type": "Debt Facility", "role": "Senior Debt"}]},
            "details": {}, "kpis": {}, "classification": {}
        });
        conform_document(&mut doc);
        let f = &doc["canonical_deal"]["facilities"][0];
        assert_eq!(f["instrument_type"], "debt_facility");
        assert_eq!(f["role"], "senior_debt");
    }

    #[test]
    fn cleans_decimal_strings() {
        let mut doc = json!({
            "canonical_deal": {"facilities": [{"id": "f1", "name": "x", "commitment": {"amount": "£25,750,000"}}]},
            "details": {}, "kpis": {}, "classification": {}
        });
        conform_document(&mut doc);
        assert_eq!(
            doc["canonical_deal"]["facilities"][0]["commitment"]["amount"],
            "25750000"
        );
    }
}
