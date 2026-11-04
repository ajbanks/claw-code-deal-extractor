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
    if is_with_source(schema) {
        return with_source_empty();
    }
    if nullable(schema) {
        return Value::Null;
    }
    if schema.get("enum").is_some() {
        return default_enum(schema);
    }
    match primary_type(schema) {
        Some("string") => {
            if schema
                .get("minLength")
                .and_then(Value::as_u64)
                .is_some_and(|m| m >= 1)
            {
                Value::String("unknown".to_string())
            } else {
                Value::String(String::new())
            }
        }
        Some("number") | Some("integer") => Value::from(0),
        Some("boolean") => Value::Bool(false),
        Some("array") => Value::Array(Vec::new()),
        Some("object") => Value::Object(Map::new()),
        _ => Value::Null,
    }
}

fn default_enum(schema: &Value) -> Value {
    for candidate in [
        "unknown",
        "other",
        "parsed",
        "extracted",
        "deal",
        "committed",
    ] {
        if let Some(m) = enum_fix(schema, candidate) {
            return Value::String(m);
        }
    }
    if let Some(first) = schema
        .get("enum")
        .and_then(Value::as_array)
        .and_then(|arr| arr.iter().find_map(Value::as_str))
    {
        return Value::String(first.to_string());
    }
    Value::String(String::new())
}

const WITH_SOURCE_DEFAULTS: &[(&str, Value)] = &[];

fn with_source_empty() -> Value {
    let mut o = Map::new();
    o.insert("source".to_string(), Value::String(String::new()));
    o.insert(
        "source_type".to_string(),
        Value::String("unknown".to_string()),
    );
    o.insert("notes".to_string(), Value::Null);
    o.insert("scope".to_string(), Value::Null);
    o.insert("evidence_text".to_string(), Value::Null);
    o.insert("value".to_string(), Value::Null);
    Value::Object(o)
}

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
            if let Value::Object(o) = inst.clone() {
                if o.values().all(Value::is_null) && nullable(schema) {
                    *inst = Value::Null;
                } else {
                    *inst = Value::Array(vec![Value::Object(o)]);
                }
                *fixes += 1;
            }
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
                Value::Null if !nullable(schema) => {
                    *inst = default_for(schema);
                    *fixes += 1;
                }
                Value::Object(o) if !nullable(schema) => {
                    let s = o
                        .get("document")
                        .or_else(|| o.get("name"))
                        .or_else(|| o.get("file_id"))
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    *inst = Value::String(s);
                    *fixes += 1;
                }
                Value::Number(n) => {
                    *inst = Value::String(n.to_string());
                    *fixes += 1;
                }
                Value::Bool(b) => {
                    *inst = Value::String(b.to_string());
                    *fixes += 1;
                }
                Value::String(s) => {
                    if schema.get("enum").is_some() {
                        if !s.trim().is_empty() {
                            if let Some(fixed) = enum_fix(schema, &s) {
                                if fixed != s {
                                    *inst = Value::String(fixed);
                                    *fixes += 1;
                                }
                                return;
                            }
                        }
                        // Unrecognised or empty enum value: null it out when
                        // nullable, else fall back to a catch-all enum member.
                        if nullable(schema) {
                            *inst = Value::Null;
                            *fixes += 1;
                            return;
                        }
                        let fallback = default_enum(schema);
                        if fallback != Value::String(s) {
                            *inst = fallback;
                            *fixes += 1;
                        }
                        return;
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
                            return;
                        }
                    }
                    if s.trim().is_empty()
                        && !nullable(schema)
                        && schema
                            .get("minLength")
                            .and_then(Value::as_u64)
                            .is_some_and(|m| m >= 1)
                    {
                        *inst = Value::String("unknown".to_string());
                        *fixes += 1;
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
    // Scalar/array/null where an object is required → wrap if it's a WithSource.
    if !inst.is_object() {
        if is_with_source(schema) {
            if inst.is_null() {
                *inst = with_source_empty();
            } else {
                let inner = std::mem::replace(inst, Value::Null);
                *inst = with_source_wrapper(inner);
            }
            *fixes += 1;
        } else if let Value::Number(n) = inst.clone() {
            let props = schema.get("properties").and_then(Value::as_object);
            let mut o = Map::new();
            if props.is_some_and(|p| p.contains_key("amount")) {
                o.insert("amount".to_string(), Value::from(n));
                if props.is_some_and(|p| p.contains_key("currency")) {
                    o.insert("currency".to_string(), Value::String("GBP".to_string()));
                }
            } else if props.is_some_and(|p| p.contains_key("value")) {
                o.insert("value".to_string(), Value::from(n.clone()));
                if props.is_some_and(|p| p.contains_key("display")) {
                    o.insert("display".to_string(), Value::String(format!("{n}%")));
                }
            } else if props.is_some_and(|p| p.contains_key("months")) {
                o.insert("months".to_string(), Value::from(n));
            } else {
                return;
            }
            *inst = Value::Object(o);
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

    // Map prompt-style aliases onto the canonical JSON Schema field names
    // before filling required defaults:
    // - evidence entries: `{ document, page, quote }` → `{ file_id, pdf_page, document_type, quote }`
    // - canonical documents: `{ name, role }` → `{ id, file_id, file_name, document_type, status }`
    if props.contains_key("file_id")
        && props.contains_key("pdf_page")
        && props.contains_key("quote")
    {
        if !obj.contains_key("file_id")
            || obj
                .get("file_id")
                .and_then(Value::as_str)
                .is_some_and(|s| s.trim().is_empty())
        {
            if let Some(doc_name) = obj
                .get("document")
                .and_then(Value::as_str)
                .filter(|s| !s.trim().is_empty())
                .map(str::to_string)
            {
                obj.insert("file_id".to_string(), Value::String(doc_name));
                *fixes += 1;
            }
        }
        if !obj.contains_key("pdf_page") || obj.get("pdf_page").is_some_and(Value::is_null) {
            if let Some(p) = obj.get("page").and_then(|v| {
                v.as_u64().or_else(|| {
                    v.as_str().and_then(|s| {
                        s.chars()
                            .filter(char::is_ascii_digit)
                            .collect::<String>()
                            .parse::<u64>()
                            .ok()
                    })
                })
            }) {
                if p >= 1 {
                    obj.insert("pdf_page".to_string(), Value::from(p));
                    *fixes += 1;
                }
            }
        }
    }
    if props.contains_key("file_id")
        && props.contains_key("file_name")
        && props.contains_key("document_type")
        && props.contains_key("status")
    {
        let name = obj
            .get("name")
            .or_else(|| obj.get("file_name"))
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .map(str::to_string);
        if let Some(n) = name {
            for k in ["id", "file_id", "file_name"] {
                if !obj.contains_key(k)
                    || obj
                        .get(k)
                        .and_then(Value::as_str)
                        .is_some_and(|s| s.trim().is_empty())
                {
                    obj.insert(k.to_string(), Value::String(n.clone()));
                    *fixes += 1;
                }
            }
        }
        if !obj.contains_key("document_type") {
            if let Some(role) = obj.get("role").and_then(Value::as_str).map(str::to_string) {
                obj.insert("document_type".to_string(), Value::String(role));
                *fixes += 1;
            }
        }
    }

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
    fn empty_enum_string_becomes_other_or_null() {
        let mut doc = json!({
            "canonical_deal": {
                "facilities": [{"id": "f1", "name": "x"}],
                "facility_parties": [
                    {"facility_id": "f1", "name": "a", "role": ""},
                    {"facility_id": "f1", "name": "b", "role": ""}
                ]
            },
            "details": {}, "kpis": {}, "classification": {}
        });
        conform_document(&mut doc);
        let p = &doc["canonical_deal"]["facility_parties"];
        // facility_parties[].role is non-nullable with an "other" member —
        // the empty string the model emitted becomes "other" instead of a
        // schema_violation rejection.
        assert_eq!(p[0]["role"], "other");
        assert_eq!(p[1]["role"], "other");
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

    #[test]
    fn maps_evidence_aliases_and_validates_missing_details() {
        let mut doc = json!({
            "canonical_deal": {
                "schema_version": "1",
                "facilities": [{
                    "id": "fac_1",
                    "name": "Senior Loan",
                    "instrument_type": "debt_facility",
                    "role": "senior_debt",
                    "status": "candidate",
                    "disposition": "requested",
                    "commitment": {"amount": "10000000", "currency": "gbp"},
                    "evidence": [{"document": "im.pdf", "page": "5", "quote": "£10m senior"}]
                }],
                "documents": [{"name": "im.pdf", "role": "information_memorandum"}],
                "focus_facility_id": "fac_1",
                "focus_resolution_status": "resolved",
                "candidate_facility_ids": ["fac_1"]
            },
            "details": {
                "address": {
                    "source": "im.pdf",
                    "source_type": "extracted",
                    "scope": null,
                    "value": {
                        "full_address": "Frankfurt am Main",
                        "region": "Hesse"
                    }
                }
            },
            "kpis": {},
            "classification": {
                "capital_position": "debt",
                "loan_type": {"category": "bridge"},
                "loan_rank": {"category": "senior"}
            }
        });
        conform_document(&mut doc);
        assert_eq!(
            doc["canonical_deal"]["facilities"][0]["evidence"][0]["file_id"],
            "im.pdf"
        );
        assert_eq!(
            doc["canonical_deal"]["facilities"][0]["evidence"][0]["pdf_page"],
            5
        );
        assert_eq!(doc["canonical_deal"]["documents"][0]["file_id"], "im.pdf");
        assert_eq!(doc["details"]["address"]["value"]["region"], Value::Null);
        let issues = crate::validate::validate(&doc);
        let scope_errs: Vec<_> = issues
            .iter()
            .filter(|i| i.path.ends_with("/scope") || i.path.ends_with("/source_type"))
            .collect();
        assert!(
            scope_errs.is_empty(),
            "unexpected scope/source_type errors: {scope_errs:?}"
        );
    }
}
