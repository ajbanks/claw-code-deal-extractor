//! Scorer-view structs — the shape the gold-standard scorer reads.
//!
//! IMPORTANT: these are NOT the extraction contract. The contract is the
//! dumped JSON Schema in `schemas/*.json`, enforced by `validate.rs` against
//! `serde_json::Value`. These structs exist only so the offline scorer can
//! grade whatever the agent emitted — including non-conformant output —
//! which is why the deserializers are lenient and `SourcedValue.value` is
//! an untyped `Value` (the real TS shape nests `{total:{amount}}` etc.).

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

/// Lenient f64 deserializer: accepts integers, floats, or numeric strings.
fn lenient_f64<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
    use serde::de::Error;
    let v = serde_json::Value::deserialize(d)?;
    match v {
        serde_json::Value::Number(n) => n.as_f64().ok_or_else(|| D::Error::custom("not a number")),
        serde_json::Value::String(s) => s.parse::<f64>().map_err(D::Error::custom),
        serde_json::Value::Null => Ok(0.0),
        other => Err(D::Error::custom(format!("expected number, got {other}"))),
    }
}

/// Lenient optional u32 deserializer: accepts integers or numeric strings.
fn lenient_opt_u32<'de, D: Deserializer<'de>>(d: D) -> Result<Option<u32>, D::Error> {
    let v = serde_json::Value::deserialize(d)?;
    Ok(match v {
        serde_json::Value::Null => None,
        serde_json::Value::Number(n) => n.as_u64().map(|n| n as u32),
        serde_json::Value::String(s) => s.trim().parse::<f64>().ok().map(|n| n as u32),
        _ => None,
    })
}

/// Lenient String deserializer: accepts strings or integers (converts to string).
fn lenient_string<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    let v = serde_json::Value::deserialize(d)?;
    Ok(match v {
        serde_json::Value::String(s) => s,
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Null => String::new(),
        serde_json::Value::Bool(b) => b.to_string(),
        other => other.to_string(),
    })
}

/// Lenient optional String deserializer.
fn lenient_opt_string<'de, D: Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
    let v = serde_json::Value::deserialize(d)?;
    Ok(match v {
        serde_json::Value::Null => None,
        serde_json::Value::String(s) => Some(s),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        other => Some(other.to_string()),
    })
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Money {#[serde(default, deserialize_with = "lenient_f64")]
    pub amount: f64,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "lenient_opt_string"
    )]
    pub currency: Option<String>,
    /// Lossless catch-all for fields not yet declared on the struct.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}

/// A sourced-value wrapper. `value` is untyped — the real contract nests
/// field-specific objects (`{total:{amount,currency}}`, `{value:<n>}`,
/// `{postcode:...}`), so the scorer walks it with `extract_amount`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourcedValue {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<Value>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "lenient_opt_string"
    )]
    pub source: Option<String>,
    /// Lossless catch-all for fields not yet declared on the struct.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Address {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line1: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line2: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub postcode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Lossless catch-all for fields not yet declared on the struct.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Facility {
    #[serde(default, deserialize_with = "lenient_string")]
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instrument_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope_level: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase_ids: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary_phase_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loan_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commitment: Option<Money>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "lenient_opt_u32"
    )]
    pub term_months: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maturity_date: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interest_terms: Option<Vec<serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stated_interest: Option<Money>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stated_redemption: Option<Money>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub facility_party_ids: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tranche_ids: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disposition: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lender_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub facility_group_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_reference: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_entity_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<Vec<Evidence>>,
    /// Lossless catch-all for fields not yet declared on the struct.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Evidence {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quote: Option<String>,
    /// Lossless catch-all for fields not yet declared on the struct.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Valuation {
    #[serde(default, deserialize_with = "lenient_string")]
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value_type: Option<String>,
    #[serde(default)]
    pub amount: Money,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valuation_basis: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "lenient_opt_u32"
    )]
    pub sale_period_days: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scenario: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<Vec<Evidence>>,
    /// Lossless catch-all for fields not yet declared on the struct.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Phase {
    #[serde(default, deserialize_with = "lenient_string")]
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "lenient_opt_u32"
    )]
    pub sequence: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "lenient_opt_u32"
    )]
    pub unit_count: Option<u32>,
    /// Lossless catch-all for fields not yet declared on the struct.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CapitalEvent {
    #[serde(default, deserialize_with = "lenient_string")]
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub facility_id: Option<String>,
    /// Lossless catch-all for fields not yet declared on the struct.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CapitalSnapshotPosition {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub facility_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Lossless catch-all for fields not yet declared on the struct.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CapitalSnapshot {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope_level: Option<String>,
    #[serde(default)]
    pub positions: Vec<CapitalSnapshotPosition>,
    /// Lossless catch-all for fields not yet declared on the struct.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CanonicalDeal {
    #[serde(default)]
    pub facilities: Vec<Facility>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub focus_facility_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub focus_resolution_status: Option<String>,
    #[serde(default)]
    pub valuations: Vec<Valuation>,
    #[serde(default)]
    pub phases: Vec<Phase>,
    #[serde(default)]
    pub capital_events: Vec<CapitalEvent>,
    #[serde(default)]
    pub capital_snapshots: Vec<CapitalSnapshot>,

    // ── Passthrough fields (declared in canonical-deal.schema.json; kept
    // opaque so the struct stays lossless when the schema evolves) ──
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub assertions: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub assets: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_facility_ids: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conditions_precedent: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conflict_groups: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conflicts: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub construction_budgets: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub covenants: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deal_scope: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub documents: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub facility_parties: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub financial_models: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guarantees: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub leases: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub obligations: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parties: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planning_applications: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prepayment_terms: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relations: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_focus_facility_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_focus_resolution_status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub review_status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sections: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub security: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tranches: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub units: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validation_issues: Option<Vec<Value>>,
    /// Lossless catch-all for fields not yet declared on the struct.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Classification {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capital_position: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset_class_primary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_type: Option<CategoryValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loan_type: Option<CategoryValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loan_rank: Option<CategoryValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operating_model: Option<CategoryValue>,
    // Plural classification dimensions (Oct 2026 schema expansion) — optional,
    // multi-valued supersets of the singular axes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset_classes: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transaction_types: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub investment_strategies: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub development_status: Option<String>,
    /// Lossless catch-all for fields not yet declared on the struct.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CategoryValue {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    /// Lossless catch-all for fields not yet declared on the struct.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DealDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loan_amount: Option<SourcedValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub senior_debt: Option<SourcedValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mezzanine_debt: Option<SourcedValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gross_debt: Option<SourcedValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub developer_equity: Option<SourcedValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub market_value: Option<SourcedValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gross_development_value: Option<SourcedValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub address: Option<SourcedValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gross_internal_area: Option<SourcedValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residential_units: Option<SourcedValue>,
    /// Lossless catch-all for fields not yet declared on the struct.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Kpis {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loan_to_value: Option<SourcedValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loan_to_gdv: Option<SourcedValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build_cost_per_area_unit: Option<SourcedValue>,
    /// Lossless catch-all for fields not yet declared on the struct.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}

/// The full deal document the gold-standard tests assert against.
/// `canonical_deal` is the structured entity model; `details`/`kpis`/
/// `classification` are the legacy flat projections.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DealDocument {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canonical_deal: Option<CanonicalDeal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub classification: Option<Classification>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<DealDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kpis: Option<Kpis>,
    /// Lossless catch-all for fields not yet declared on the struct.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deal_document_round_trips() {
        let doc = DealDocument {
            canonical_deal: Some(CanonicalDeal {
                facilities: vec![Facility {
                    id: "f1".into(),
                    name: Some("Bridge".into()),
                    loan_type: Some("bridge".into()),
                    commitment: Some(Money {amount: 25_750_000.0,
                        currency: Some("GBP".into()), ..Default::default() }),
                    ..Default::default()
                }],
                focus_facility_id: Some("f1".into()),
                focus_resolution_status: Some("resolved".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let json = serde_json::to_string(&doc).unwrap();
        let back: DealDocument = serde_json::from_str(&json).unwrap();
        assert_eq!(
            back.canonical_deal
                .as_ref()
                .unwrap()
                .focus_facility_id
                .as_deref(),
            Some("f1")
        );
    }

    #[test]
    fn sourced_value_accepts_nested_object() {
        // The real TS contract: details.loan_amount.value = {total:{amount,currency}}
        let v: SourcedValue = serde_json::from_str(
            r#"{"value":{"total":{"amount":"25750000","currency":"gbp"}},"source":"f.pdf"}"#,
        )
        .unwrap();
        assert!(v.value.is_some());
        assert_eq!(v.source.as_deref(), Some("f.pdf"));
    }
}
