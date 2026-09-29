//! The extraction contract: JSON Schemas dumped from the shared TS TypeBox
//! definitions (see `copilot/scripts/dump-schemas.ts`). The schemas are the
//! single source of truth — the prompt contract is *rendered from them*,
//! not hand-written, so it cannot drift from the TS pipeline.

use serde_json::Value;

pub const CANONICAL_DEAL_SCHEMA: &str = include_str!("../schemas/canonical-deal.schema.json");
pub const DEAL_DETAILS_SCHEMA: &str = include_str!("../schemas/deal-details.schema.json");
pub const OUTPUT_SCHEMA: &str = include_str!("../schemas/extraction-output.schema.json");

/// Canonical-deal top-level fields the agent is asked to fill. The rest are
/// TS-pipeline internals (assertions, conflicts, review workflow, phase-6
/// legal entities) that the agent emits as empty arrays.
const CD_ACTIVE_FIELDS: &[&str] = &[
    "schema_version",
    "facilities",
    "tranches",
    "facility_parties",
    "capital_events",
    "capital_snapshots",
    "focus_facility_id",
    "focus_resolution_status",
    "candidate_facility_ids",
    "deal_scope",
    "documents",
    "assets",
    "phases",
    "planning_applications",
    "valuations",
    "construction_budgets",
];

/// Details fields that must carry a real value when evidence exists. The
/// full 111-field list is still validated; this subset is what the prompt
/// calls out by name so the model does not starve them.
const DETAILS_KEY_FIELDS: &[&str] = &[
    "address",
    "loan_amount",
    "senior_debt",
    "mezzanine_debt",
    "gross_debt",
    "developer_equity",
    "market_value",
    "gross_development_value",
    "gross_internal_area",
    "loan_to_value",
    "loan_to_gdv",
    "loan_type",
    "loan_rank",
    "classification",
    "loan_term",
    "interest_rate",
    "deal_type",
    "purchase_price",
    "asking_price",
    "day_1_value",
    "residential_units",
    "use_of_funds",
    "source_of_funds",
    "exit_plan",
];

fn canonical_schema() -> Value {
    serde_json::from_str(CANONICAL_DEAL_SCHEMA).expect("embedded schema must parse")
}

/// The compact JSON-Schema "extraction profile" sent to the model as part of
/// the prompt. Derived from the real schemas by pruning TS-internal fields —
/// so enums, required lists, and the DecimalString money shape are always
/// the true contract, never a paraphrase.
#[must_use]
pub fn profile_schema() -> Value {
    let cd = canonical_schema();
    let mut props = serde_json::Map::new();
    if let Some(p) = cd.get("properties").and_then(|v| v.as_object()) {
        for key in CD_ACTIVE_FIELDS {
            if let Some(v) = p.get(*key) {
                props.insert((*key).to_string(), v.clone());
            }
        }
        // The agent must still emit the required arrays it does not fill.
        for key in ["assertions", "conflicts", "validation_issues"] {
            if let Some(v) = p.get(key) {
                props.insert(key.to_string(), v.clone());
            }
        }
    }
    let details =
        serde_json::from_str::<Value>(DEAL_DETAILS_SCHEMA).expect("embedded schema must parse");

    Value::Object(serde_json::Map::from_iter([
        ("type".into(), Value::String("object".into())),
        (
            "required".into(),
            serde_json::json!(["canonical_deal", "details"]),
        ),
        (
            "properties".into(),
            serde_json::json!({
                "canonical_deal": {
                    "type": "object",
                    "required": cd.get("required").cloned().unwrap_or(serde_json::json!([])),
                    "properties": props,
                },
                "details": details,
                "kpis": {"type": ["object", "null"]},
                "classification": {"type": ["object", "null"]},
            }),
        ),
    ]))
}

/// Render one JSON-Schema property into a compact `name: type` line.
fn render_field(name: &str, schema: &Value, depth: usize, out: &mut String) {
    let indent = "  ".repeat(depth);
    let ty = schema
        .get("type")
        .map(|t| match t {
            Value::String(s) => s.clone(),
            Value::Array(a) => a
                .iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join("|"),
            _ => "any".into(),
        })
        .unwrap_or_else(|| {
            if schema.get("properties").is_some() {
                "object".into()
            } else {
                "any".into()
            }
        });

    let mut desc = String::new();
    if let Some(e) = schema.get("enum").and_then(|v| v.as_array()) {
        let vals: Vec<String> = e
            .iter()
            .map(|v| v.as_str().unwrap_or("?").to_string())
            .collect();
        desc.push_str(&format!(" — one of: {}", vals.join(", ")));
    }
    if let Some(p) = schema.get("pattern").and_then(|v| v.as_str()) {
        desc.push_str(&format!(" — pattern {p}"));
    }

    out.push_str(&format!("{indent}{name}: {ty}{desc}\n"));

    // Recurse one level into object properties and array items so the model
    // sees entity shape without a full schema dump.
    if depth < 3 {
        if let Some(p) = schema.get("properties").and_then(|v| v.as_object()) {
            for (k, v) in p {
                render_field(k, v, depth + 1, out);
            }
        }
        if let Some(items) = schema.get("items") {
            if let Some(p) = items.get("properties").and_then(|v| v.as_object()) {
                out.push_str(&format!("{indent}  [item]\n"));
                for (k, v) in p {
                    render_field(k, v, depth + 2, out);
                }
            }
        }
    }
}

/// The compact text contract embedded in the extraction prompts. Rendered
/// from the dumped JSON Schema so enums / required fields / DecimalString
/// amounts are always the real TS contract.
#[must_use]
pub fn render_contract() -> String {
    let cd = canonical_schema();
    let mut out = String::new();

    out.push_str("## canonical_deal — required top-level fields\n");
    if let Some(req) = cd.get("required").and_then(|v| v.as_array()) {
        let names: Vec<&str> = req.iter().filter_map(|v| v.as_str()).collect();
        out.push_str(&format!("required: {}\n\n", names.join(", ")));
    }

    out.push_str("canonical_deal fields (fill these; emit `assertions`, `conflicts`, `validation_issues` as []):\n");
    if let Some(p) = cd.get("properties").and_then(|v| v.as_object()) {
        for key in CD_ACTIVE_FIELDS {
            if let Some(v) = p.get(*key) {
                render_field(key, v, 1, &mut out);
            }
        }
    }

    out.push_str(
        "\n## details — every field is a sourced value object:\n\
         { \"source\": <document file_name>, \"source_type\": \
         \"extracted|calculated|assumed|web_search|unknown\", \
         \"notes\": <string|null>, \"scope\": \"deal|project|null\", \
         \"evidence_text\": <string|null>, \"value\": <field-specific object|null> }\n\n\
         Emit every field; use \"value\": null where the documents give no evidence.\n\
         Critical fields: ",
    );
    out.push_str(&DETAILS_KEY_FIELDS.join(", "));
    out.push('\n');

    // Render the full details field list so the model knows every name.
    let details =
        serde_json::from_str::<Value>(DEAL_DETAILS_SCHEMA).expect("embedded schema must parse");
    if let Some(p) = details.get("properties").and_then(|v| v.as_object()) {
        out.push_str("\nAll details fields:\n");
        let names: Vec<&String> = p.keys().collect();
        for chunk in names.chunks(8) {
            out.push_str(
                &chunk
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
            );
            out.push('\n');
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schemas_parse() {
        assert!(serde_json::from_str::<Value>(CANONICAL_DEAL_SCHEMA).is_ok());
        assert!(serde_json::from_str::<Value>(DEAL_DETAILS_SCHEMA).is_ok());
        assert!(serde_json::from_str::<Value>(OUTPUT_SCHEMA).is_ok());
    }

    #[test]
    fn profile_contains_active_fields() {
        let p = profile_schema();
        let cd = p["properties"]["canonical_deal"]["properties"]
            .as_object()
            .unwrap();
        assert!(cd.contains_key("facilities"));
        assert!(cd.contains_key("focus_facility_id"));
        assert!(cd.contains_key("phases"));
        assert!(cd.contains_key("valuations"));
        // TS-internal fields the agent emits empty.
        assert!(cd.contains_key("assertions"));
        // Truly internal fields stay out of the profile.
        assert!(!cd.contains_key("conflict_groups"));
        assert!(!cd.contains_key("review_status"));
    }

    #[test]
    fn contract_mentions_decimal_string_money() {
        let c = render_contract();
        assert!(c.contains("facilities"));
        assert!(c.contains("focus_facility_id"));
        assert!(c.contains("loan_amount"));
    }
}
