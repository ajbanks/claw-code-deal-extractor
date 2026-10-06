//! Semantic validation of an extracted deal document.
//!
//! Two layers:
//! 1. JSON Schema conformance against the dumped TS contract
//!    (`schemas/extraction-output.schema.json`).
//! 2. Domain invariants — deal-agnostic rules with no gold-standard values,
//!    so they generalise to unseen deals. `Error`-severity issues drive the
//!    repair loop; `Warn` issues are reported but never drive repair.

use serde_json::Value;

use crate::contract;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warn,
}

#[derive(Debug, Clone)]
pub struct Issue {
    pub code: &'static str,
    pub path: String,
    pub message: String,
    pub severity: Severity,
}

impl Issue {
    fn error(code: &'static str, path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code,
            path: path.into(),
            message: message.into(),
            severity: Severity::Error,
        }
    }
    fn warn(code: &'static str, path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code,
            path: path.into(),
            message: message.into(),
            severity: Severity::Warn,
        }
    }
}

/// Full validation: JSON Schema errors + domain invariants.
#[must_use]
pub fn validate(doc: &Value) -> Vec<Issue> {
    let mut issues = schema_issues(doc);
    issues.extend(domain_issues(doc));
    issues
}

/// Only `Error`-severity issues drive the repair loop.
#[must_use]
pub fn error_count(issues: &[Issue]) -> usize {
    issues
        .iter()
        .filter(|i| i.severity == Severity::Error)
        .count()
}

// ─── Layer 1: JSON Schema ────────────────────────────────────────────────────

fn schema_issues(doc: &Value) -> Vec<Issue> {
    let schema: Value = match serde_json::from_str(contract::OUTPUT_SCHEMA) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let validator = match jsonschema::validator_for(&schema) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for err in validator.iter_errors(doc) {
        out.push(Issue::error(
            "schema_violation",
            err.instance_path().to_string(),
            err.to_string(),
        ));
    }
    out
}

// ─── Layer 2: domain invariants ─────────────────────────────────────────────

fn domain_issues(doc: &Value) -> Vec<Issue> {
    let mut out = Vec::new();
    let Some(cd) = doc.get("canonical_deal") else {
        return out;
    };

    referential_integrity(cd, &mut out);
    focus_semantics(cd, &mut out);
    lifecycle(cd, &mut out);
    enums_and_formats(cd, &mut out);
    projection(doc, cd, &mut out);
    completeness(doc, cd, &mut out);
    out
}

fn facilities(cd: &Value) -> &[Value] {
    cd.get("facilities")
        .and_then(|v| v.as_array())
        .map_or(&[], |a| a.as_slice())
}

fn facility_ids(cd: &Value) -> Vec<&str> {
    facilities(cd)
        .iter()
        .filter_map(|f| f.get("id").and_then(|v| v.as_str()))
        .collect()
}

fn amount_of(money: &Value) -> Option<f64> {
    // canonical_money.amount is a DecimalString — exact decimal, never a
    // number and never millions shorthand.
    money
        .get("amount")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .or_else(|| money.get("amount").and_then(|v| v.as_f64()))
}

fn details_amount(doc: &Value, key: &str) -> Option<f64> {
    let field = doc.get("details")?.get(key)?;
    let v = field.get("value")?;
    // details fields are sourced-value objects: { value: { ... } } or
    // { value: <number> } depending on the field. Try both.
    if let Some(n) = v.as_f64() {
        return Some(n);
    }
    v.get("amount")
        .and_then(|a| a.as_str())
        .and_then(|s| s.parse().ok())
        .or_else(|| v.get("amount").and_then(|a| a.as_f64()))
        .or_else(|| v.get("value").and_then(|n| n.as_f64()))
}

// 1–3. Referential integrity
fn referential_integrity(cd: &Value, out: &mut Vec<Issue>) {
    let ids = facility_ids(cd);
    let idset: std::collections::BTreeSet<&str> = ids.iter().copied().collect();

    if let Some(focus) = cd.get("focus_facility_id").and_then(|v| v.as_str()) {
        if !focus.is_empty() && !idset.contains(focus) {
            out.push(Issue::error(
                "focus_not_in_facilities",
                "canonical_deal.focus_facility_id",
                format!("focus_facility_id \"{focus}\" does not match any facility id. Point it at the requested/new-money facility."),
            ));
        }
    }

    if let Some(cands) = cd.get("candidate_facility_ids").and_then(|v| v.as_array()) {
        for c in cands {
            if let Some(c) = c.as_str() {
                if !idset.contains(c) {
                    out.push(Issue::error(
                        "candidate_not_in_facilities",
                        "canonical_deal.candidate_facility_ids",
                        format!("candidate_facility_id \"{c}\" does not match any facility id."),
                    ));
                }
            }
        }
    }

    if let Some(events) = cd.get("capital_events").and_then(|v| v.as_array()) {
        for (i, e) in events.iter().enumerate() {
            if let Some(fid) = e.get("facility_id").and_then(|v| v.as_str()) {
                if !idset.contains(fid) {
                    out.push(Issue::error(
                        "capital_event_unknown_facility",
                        format!("canonical_deal.capital_events[{i}].facility_id"),
                        format!("capital_event references facility \"{fid}\" which is not in facilities[]."),
                    ));
                }
            }
        }
    }

    if let Some(snaps) = cd.get("capital_snapshots").and_then(|v| v.as_array()) {
        for (i, s) in snaps.iter().enumerate() {
            if let Some(pos) = s.get("positions").and_then(|v| v.as_array()) {
                for (j, p) in pos.iter().enumerate() {
                    if let Some(fid) = p.get("facility_id").and_then(|v| v.as_str()) {
                        if !idset.contains(fid) {
                            out.push(Issue::error(
                                "snapshot_unknown_facility",
                                format!("canonical_deal.capital_snapshots[{i}].positions[{j}]"),
                                format!("capital_snapshot position references facility \"{fid}\" which is not in facilities[]."),
                            ));
                        }
                    }
                }
            }
        }
    }

    // phase_id references
    if let Some(vals) = cd.get("valuations").and_then(|v| v.as_array()) {
        let phase_ids: Vec<&str> = cd
            .get("phases")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|p| p.get("id").and_then(|v| v.as_str()))
                    .collect()
            })
            .unwrap_or_default();
        let phase_set: std::collections::BTreeSet<&str> = phase_ids.iter().copied().collect();
        if !phase_set.is_empty() {
            for (i, v) in vals.iter().enumerate() {
                if let Some(pid) = v.get("phase_id").and_then(|v| v.as_str()) {
                    if !pid.is_empty() && !phase_set.contains(pid) {
                        out.push(Issue::error(
                            "valuation_unknown_phase",
                            format!("canonical_deal.valuations[{i}].phase_id"),
                            format!(
                                "valuation references phase \"{pid}\" which is not in phases[]."
                            ),
                        ));
                    }
                }
            }
        }
    }
}

// 4–5. Focus-facility semantics
fn focus_semantics(cd: &Value, out: &mut Vec<Issue>) {
    let focus_id = cd
        .get("focus_facility_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if focus_id.is_empty() {
        // Whether an empty focus is an error depends on the deal; the
        // completeness rule flags it for debt deals. Don't double-report.
        return;
    }
    let Some(focus) = facilities(cd)
        .iter()
        .find(|f| f.get("id").and_then(|v| v.as_str()) == Some(focus_id))
    else {
        return; // covered by referential_integrity
    };

    // The focus facility is the new money being requested — its disposition
    // must be "requested". existing/repaid/proposed are wrong for the ask.
    if let Some(disp) = focus.get("disposition").and_then(|v| v.as_str()) {
        if disp != "requested" {
            out.push(Issue::error(
                "focus_not_requested",
                "canonical_deal.focus_facility_id",
                format!("focus facility \"{}\" has disposition \"{disp}\" — the focus must be the facility being requested (disposition \"requested\").",
                    focus.get("name").and_then(|v| v.as_str()).unwrap_or(focus_id)),
            ));
        }
    }

    // An undrawn ask cannot already be drawn.
    if let Some(events) = cd.get("capital_events").and_then(|v| v.as_array()) {
        let drawn = events.iter().any(|e| {
            e.get("facility_id").and_then(|v| v.as_str()) == Some(focus_id)
                && e.get("event_type").and_then(|v| v.as_str()) == Some("drawn")
        });
        if drawn {
            out.push(Issue::error(
                "focus_has_drawn_event",
                "canonical_deal.capital_events",
                "the focus facility (the new money being requested) has a `drawn` capital event — it is not yet drawn.",
            ));
        }
    }
}

// 6–7. Lifecycle coherence
fn lifecycle(cd: &Value, out: &mut Vec<Issue>) {
    let events = cd
        .get("capital_events")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let repaid_ids: std::collections::BTreeSet<&str> = events
        .iter()
        .filter(|e| {
            matches!(
                e.get("event_type").and_then(|v| v.as_str()),
                Some("fully_repaid") | Some("refinanced")
            )
        })
        .filter_map(|e| e.get("facility_id").and_then(|v| v.as_str()))
        .collect();

    for (i, f) in facilities(cd).iter().enumerate() {
        let fid = f.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let status = f.get("status").and_then(|v| v.as_str()).unwrap_or("");
        if status == "repaid" && !repaid_ids.contains(fid) {
            out.push(Issue::error(
                "repaid_without_event",
                format!("canonical_deal.facilities[{i}]"),
                format!("facility \"{}\" is status \"repaid\" but has no fully_repaid/refinanced capital event.",
                    f.get("name").and_then(|v| v.as_str()).unwrap_or(fid)),
            ));
        }
        if repaid_ids.contains(fid) && status != "repaid" && status != "superseded" {
            out.push(Issue::error(
                "event_without_repaid_status",
                format!("canonical_deal.facilities[{i}]"),
                format!("facility \"{}\" has a fully_repaid/refinanced event but status is \"{status}\" — it should be \"repaid\" or \"superseded\".",
                    f.get("name").and_then(|v| v.as_str()).unwrap_or(fid)),
            ));
        }
    }

    // Duplicate facilities: identical name + commitment appearing twice.
    let mut seen: std::collections::BTreeMap<(String, String), usize> =
        std::collections::BTreeMap::new();
    for (i, f) in facilities(cd).iter().enumerate() {
        let name = f
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_lowercase();
        let amt = f
            .get("commitment")
            .map(|m| m.get("amount").cloned().unwrap_or(Value::Null))
            .unwrap_or(Value::Null);
        let key = (name.clone(), amt.to_string());
        if let Some(first) = seen.insert(key, i) {
            if !name.is_empty() {
                out.push(Issue::warn(
                    "duplicate_facility",
                    format!("canonical_deal.facilities[{i}]"),
                    format!("facility \"{}\" duplicates facilities[{first}] (same name and commitment).",
                        f.get("name").and_then(|v| v.as_str()).unwrap_or("")),
                ));
            }
        }
    }
}

// 8–9. Enums and amount formats
fn enums_and_formats(cd: &Value, out: &mut Vec<Issue>) {
    const LOAN_TYPES: &[&str] = &["bridge", "development", "investment_term"];
    const ROLES: &[&str] = &[
        "senior_debt",
        "mezzanine_debt",
        "junior_debt",
        "whole_loan",
        "developer_equity",
        "preferred_equity",
        "other",
    ];
    const INSTRUMENTS: &[&str] = &[
        "debt_facility",
        "ordinary_equity",
        "preferred_equity",
        "shareholder_loan",
        "vendor_finance",
    ];
    const STATUSES: &[&str] = &["candidate", "active", "repaid", "cancelled", "superseded"];
    const DISPOSITIONS: &[&str] = &["requested", "existing", "repaid", "proposed"];

    for (i, f) in facilities(cd).iter().enumerate() {
        let path = format!("canonical_deal.facilities[{i}]");
        if let Some(lt) = f.get("loan_type").and_then(|v| v.as_str()) {
            if !lt.is_empty() && !LOAN_TYPES.contains(&lt) {
                out.push(Issue::error(
                    "bad_loan_type",
                    format!("{path}.loan_type"),
                    format!(
                        "loan_type \"{lt}\" must be one of: {}",
                        LOAN_TYPES.join(", ")
                    ),
                ));
            }
        }
        if let Some(r) = f.get("role").and_then(|v| v.as_str()) {
            if !ROLES.contains(&r) {
                out.push(Issue::error(
                    "bad_role",
                    format!("{path}.role"),
                    format!("role \"{r}\" must be one of: {}", ROLES.join(", ")),
                ));
            }
        }
        if let Some(it) = f.get("instrument_type").and_then(|v| v.as_str()) {
            if !INSTRUMENTS.contains(&it) {
                out.push(Issue::error(
                    "bad_instrument_type",
                    format!("{path}.instrument_type"),
                    format!(
                        "instrument_type \"{it}\" must be one of: {}",
                        INSTRUMENTS.join(", ")
                    ),
                ));
            }
        }
        if let Some(s) = f.get("status").and_then(|v| v.as_str()) {
            if !STATUSES.contains(&s) {
                out.push(Issue::error(
                    "bad_status",
                    format!("{path}.status"),
                    format!("status \"{s}\" must be one of: {}", STATUSES.join(", ")),
                ));
            }
        }
        if let Some(d) = f.get("disposition").and_then(|v| v.as_str()) {
            if !DISPOSITIONS.contains(&d) {
                out.push(Issue::error(
                    "bad_disposition",
                    format!("{path}.disposition"),
                    format!(
                        "disposition \"{d}\" must be one of: {}",
                        DISPOSITIONS.join(", ")
                    ),
                ));
            }
        }

        // Magnitude sanity: a commitment below £100k is almost certainly
        // millions shorthand. Flag it for restatement in absolute units.
        if let Some(c) = f.get("commitment") {
            if let Some(amt) = amount_of(c) {
                if amt > 0.0 && amt < 100_000.0 {
                    out.push(Issue::error("amount_millions_shorthand", format!("{path}.commitment.amount"),
                        format!("commitment amount {amt} is implausibly small for a facility — it looks like millions shorthand. Restate it as an exact decimal string in absolute units (e.g. \"25750000\", not 25.75).")));
                } else if amt > 2_000_000_000.0 {
                    out.push(Issue::error("amount_pence_scaled", format!("{path}.commitment.amount"),
                        format!("commitment amount {amt} is implausibly large for a facility — it looks like a pence-scaled figure (the decimal point was dropped). Restate it in absolute currency units (e.g. \"73393202.13\", not \"7339320213\").")));
                }
            }
        }
    }

    // Same magnitude check on valuations.
    if let Some(vals) = cd.get("valuations").and_then(|v| v.as_array()) {
        for (i, v) in vals.iter().enumerate() {
            if let Some(amt) = v.get("amount").and_then(amount_of) {
                if amt > 0.0 && amt < 100_000.0 {
                    out.push(Issue::error("amount_millions_shorthand", format!("canonical_deal.valuations[{i}].amount.amount"),
                        format!("valuation amount {amt} looks like millions shorthand — restate in absolute units.")));
                }
            }
        }
    }
}

// 10–12. Projection consistency
fn projection(doc: &Value, cd: &Value, out: &mut Vec<Issue>) {
    let focus_id = cd
        .get("focus_facility_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let focus = facilities(cd)
        .iter()
        .find(|f| f.get("id").and_then(|v| v.as_str()) == Some(focus_id));

    // loan_amount equals the focus facility's commitment — never a sum
    // across facilities.
    if let (Some(f), Some(la)) = (focus, details_amount(doc, "loan_amount")) {
        if let Some(commit) = f.get("commitment").and_then(amount_of) {
            if commit > 0.0 && (la - commit).abs() / commit > 0.005 {
                out.push(Issue::error(
                    "loan_amount_ne_focus_commitment",
                    "details.loan_amount",
                    format!("loan_amount {la:.0} does not equal the focus facility commitment {commit:.0}. loan_amount is the requested facility's commitment — never a sum across facilities."),
                ));
            }
        }
    }

    // gross_debt sums only in-scope debt facilities.
    if let Some(gd) = details_amount(doc, "gross_debt") {
        let in_scope: f64 = facilities(cd)
            .iter()
            .filter(|f| f.get("instrument_type").and_then(|v| v.as_str()) == Some("debt_facility"))
            .filter(|f| {
                let disp = f.get("disposition").and_then(|v| v.as_str());
                let scope = f.get("scope_level").and_then(|v| v.as_str());
                // Include requested + repaid-in-deal facilities; exclude
                // proposed/existing project-scope facilities.
                disp != Some("proposed") && scope != Some("project")
            })
            .filter_map(|f| f.get("commitment").and_then(amount_of))
            .sum();
        if in_scope > 0.0 && gd > 0.0 && (gd - in_scope).abs() / in_scope > 0.05 {
            out.push(Issue::warn(
                "gross_debt_scope_mismatch",
                "details.gross_debt",
                format!("gross_debt {gd:.0} differs from the sum of in-scope debt facilities {in_scope:.0}. gross_debt should cover deal-scope debt only — exclude proposed/existing project-scope facilities."),
            ));
        }
    }

    // KPI consistency: LTV = loan / market_value; LTGDV = loan / GDV.
    if let (Some(la), Some(mv)) = (
        details_amount(doc, "loan_amount"),
        details_amount(doc, "market_value"),
    ) {
        if la > 0.0 && mv > 0.0 {
            let ltv = doc
                .get("kpis")
                .and_then(|k| k.get("loan_to_value"))
                .and_then(|v| v.get("value"))
                .and_then(|v| v.as_f64());
            if let Some(ltv) = ltv {
                let expected = la / mv * 100.0;
                if (ltv - expected).abs() > 2.0 {
                    out.push(Issue::warn(
                        "ltv_inconsistent",
                        "kpis.loan_to_value",
                        format!("loan_to_value {ltv:.1}% but loan_amount/market_value = {expected:.1}%. LTV is loan / market value, not loan / GDV."),
                    ));
                }
            }
        }
    }
}

// 13–15. Completeness
fn completeness(doc: &Value, cd: &Value, out: &mut Vec<Issue>) {
    // Debt deals must have loan_type + loan_rank categories.
    let capital_position = doc
        .get("classification")
        .and_then(|c| c.get("capital_position"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let looks_debt = capital_position == "debt"
        || facilities(cd)
            .iter()
            .any(|f| f.get("instrument_type").and_then(|v| v.as_str()) == Some("debt_facility"));
    if looks_debt {
        let lt = doc
            .get("classification")
            .and_then(|c| c.get("loan_type"))
            .and_then(|v| v.get("category"))
            .and_then(|v| v.as_str());
        if lt.is_none() || lt == Some("") {
            out.push(Issue::error(
                "debt_deal_missing_loan_type",
                "classification.loan_type",
                "this is a debt deal — classification.loan_type.category must be one of: bridge, development, investment_term.",
            ));
        }
        let lr = doc
            .get("classification")
            .and_then(|c| c.get("loan_rank"))
            .and_then(|v| v.get("category"))
            .and_then(|v| v.as_str());
        if lr.is_none() || lr == Some("") {
            out.push(Issue::error(
                "debt_deal_missing_loan_rank",
                "classification.loan_rank",
                "this is a debt deal — classification.loan_rank.category must be one of: senior, mezzanine, junior, whole_loan.",
            ));
        }
    }

    // Phase arithmetic: stated totals must equal the sum.
    if let Some(phases) = cd.get("phases").and_then(|v| v.as_array()) {
        let total: u64 = phases
            .iter()
            .filter_map(|p| p.get("unit_count").and_then(|v| v.as_u64()))
            .sum();
        let stated = doc
            .get("details")
            .and_then(|d| d.get("residential_units"))
            .and_then(|f| f.get("value"))
            .and_then(|v| {
                v.get("value")
                    .and_then(|n| n.as_u64())
                    .or_else(|| v.as_u64())
            });
        if let Some(stated) = stated {
            if total > 0 && stated != total {
                out.push(Issue::error(
                    "unit_count_mismatch",
                    "details.residential_units",
                    format!("residential_units {stated} does not equal the sum of phase unit_counts {total} — phases are being double-counted or a phase is missing."),
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn valid_doc() -> Value {
        json!({
            "canonical_deal": {
                "schema_version": "1",
                "facilities": [{
                    "id": "fac-1", "external_reference": null, "name": "Bridge Loan",
                    "instrument_type": "debt_facility", "role": "senior_debt",
                    "scope_level": "deal", "loan_type": "bridge",
                    "commitment": {"amount": "25750000", "currency": "gbp"},
                    "term_months": 6, "maturity_date": null, "interest_terms": null,
                    "facility_party_ids": [], "tranche_ids": [],
                    "status": "candidate", "disposition": "requested", "evidence": [],
                }],
                "tranches": [], "facility_parties": [],
                "capital_events": [], "capital_snapshots": [],
                "focus_facility_id": "fac-1",
                "focus_resolution_status": "resolved",
                "candidate_facility_ids": ["fac-1"],
                "assertions": [], "conflicts": [], "validation_issues": [],
            },
            "classification": {
                "capital_position": "debt",
                "loan_type": {"category": "bridge"},
                "loan_rank": {"category": "whole_loan"},
            },
            "details": {
                "loan_amount": {"value": {"amount": "25750000"}},
                "market_value": {"value": {"amount": "32500000"}},
            },
            "kpis": {"loan_to_value": {"value": 79.2}},
        })
    }

    #[test]
    fn valid_doc_has_no_errors() {
        let issues = validate(&valid_doc());
        let errs: Vec<_> = issues
            .iter()
            .filter(|i| i.severity == Severity::Error)
            .collect();
        // Schema errors are acceptable here (the test doc is a subset); only
        // domain-invariant errors should be zero.
        let domain: Vec<_> = errs
            .iter()
            .filter(|i| i.code != "schema_violation")
            .collect();
        assert!(domain.is_empty(), "domain errors: {domain:?}");
    }

    #[test]
    fn focus_not_in_facilities_is_error() {
        let mut d = valid_doc();
        d["canonical_deal"]["focus_facility_id"] = json!("nope");
        let issues = validate(&d);
        assert!(issues.iter().any(|i| i.code == "focus_not_in_facilities"));
    }

    #[test]
    fn focus_with_drawn_event_is_error() {
        let mut d = valid_doc();
        d["canonical_deal"]["capital_events"] = json!([{
            "id": "e1", "facility_id": "fac-1", "event_type": "drawn",
            "effective_date": null, "trigger": null, "amount": null,
            "resulting_balance": null, "evidence": [],
        }]);
        let issues = validate(&d);
        assert!(issues.iter().any(|i| i.code == "focus_has_drawn_event"));
    }

    #[test]
    fn repaid_without_event_is_error() {
        let mut d = valid_doc();
        d["canonical_deal"]["facilities"][0]["status"] = json!("repaid");
        d["canonical_deal"]["facilities"][0]["disposition"] = json!("repaid");
        let issues = validate(&d);
        assert!(issues.iter().any(|i| i.code == "repaid_without_event"));
    }

    #[test]
    fn millions_shorthand_is_flagged() {
        let mut d = valid_doc();
        d["canonical_deal"]["facilities"][0]["commitment"]["amount"] = json!("25.75");
        let issues = validate(&d);
        assert!(issues.iter().any(|i| i.code == "amount_millions_shorthand"));
    }

    #[test]
    fn loan_amount_mismatch_is_error() {
        let mut d = valid_doc();
        d["details"]["loan_amount"] = json!({"value": {"amount": "44000000"}});
        let issues = validate(&d);
        assert!(issues
            .iter()
            .any(|i| i.code == "loan_amount_ne_focus_commitment"));
    }

    #[test]
    fn bad_loan_type_is_error() {
        let mut d = valid_doc();
        d["canonical_deal"]["facilities"][0]["loan_type"] = json!("finance requirement");
        let issues = validate(&d);
        assert!(issues.iter().any(|i| i.code == "bad_loan_type"));
    }

    #[test]
    fn debt_deal_missing_rank_is_error() {
        let mut d = valid_doc();
        d["classification"]["loan_rank"] = json!(null);
        let issues = validate(&d);
        assert!(issues
            .iter()
            .any(|i| i.code == "debt_deal_missing_loan_rank"));
    }

    #[test]
    fn unit_count_mismatch_is_error() {
        let mut d = valid_doc();
        d["canonical_deal"]["phases"] = json!([
            {"id":"p1","unit_count":148},{"id":"p2","unit_count":41}
        ]);
        d["details"]["residential_units"] = json!({"value": {"value": 337}});
        let issues = validate(&d);
        assert!(issues.iter().any(|i| i.code == "unit_count_mismatch"));
    }
}
