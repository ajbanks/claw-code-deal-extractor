//! Domain-invariant tests: a violating and a clean fixture per rule.
//! These are the semantic critic — no gold-standard values, so they
//! generalise to unseen deals.

use deal_extractor::validate::{self, Severity};
use serde_json::json;

fn has_error(doc: &serde_json::Value, code: &str) -> bool {
    validate::validate(doc)
        .iter()
        .any(|i| i.code == code && i.severity == Severity::Error)
}

/// Minimal conformant skeleton — one requested debt facility, resolved focus.
fn base() -> serde_json::Value {
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
fn focus_must_reference_existing_facility() {
    let mut d = base();
    d["canonical_deal"]["focus_facility_id"] = json!("nope");
    assert!(has_error(&d, "focus_not_in_facilities"));
}

#[test]
fn focus_must_be_requested_disposition() {
    let mut d = base();
    d["canonical_deal"]["facilities"][0]["disposition"] = json!("existing");
    assert!(has_error(&d, "focus_not_requested"));
}

#[test]
fn focus_cannot_have_drawn_event() {
    let mut d = base();
    d["canonical_deal"]["capital_events"] = json!([{
        "id": "e1", "facility_id": "fac-1", "event_type": "drawn",
        "effective_date": null, "trigger": null, "amount": null,
        "resulting_balance": null, "evidence": [],
    }]);
    assert!(has_error(&d, "focus_has_drawn_event"));
}

#[test]
fn repaid_status_requires_repaid_event() {
    let mut d = base();
    d["canonical_deal"]["facilities"][0]["status"] = json!("repaid");
    assert!(has_error(&d, "repaid_without_event"));

    // And the reverse: a repaid event on a non-repaid facility.
    let mut d2 = base();
    d2["canonical_deal"]["facilities"]
        .as_array_mut()
        .unwrap()
        .push(json!({
            "id": "fac-2", "external_reference": null, "name": "Old Debt",
            "instrument_type": "debt_facility", "role": "senior_debt",
            "scope_level": "deal", "loan_type": null,
            "commitment": {"amount": "1000000", "currency": "gbp"},
            "term_months": null, "maturity_date": null, "interest_terms": null,
            "facility_party_ids": [], "tranche_ids": [],
            "status": "active", "disposition": "existing", "evidence": [],
        }));
    d2["canonical_deal"]["capital_events"] = json!([{
        "id": "e2", "facility_id": "fac-2", "event_type": "fully_repaid",
        "effective_date": null, "trigger": null, "amount": null,
        "resulting_balance": null, "evidence": [],
    }]);
    assert!(has_error(&d2, "event_without_repaid_status"));
}

#[test]
fn enums_are_enforced() {
    let mut d = base();
    d["canonical_deal"]["facilities"][0]["loan_type"] = json!("finance requirement");
    assert!(has_error(&d, "bad_loan_type"));

    let mut d2 = base();
    d2["canonical_deal"]["facilities"][0]["role"] = json!("nonsense");
    assert!(has_error(&d2, "bad_role"));

    let mut d3 = base();
    d3["canonical_deal"]["facilities"][0]["status"] = json!("bogus");
    assert!(has_error(&d3, "bad_status"));
}

#[test]
fn millions_shorthand_is_flagged() {
    let mut d = base();
    d["canonical_deal"]["facilities"][0]["commitment"]["amount"] = json!("25.75");
    assert!(has_error(&d, "amount_millions_shorthand"));
}

#[test]
fn loan_amount_must_equal_focus_commitment() {
    let mut d = base();
    d["details"]["loan_amount"] = json!({"value": {"amount": "44000000"}});
    assert!(has_error(&d, "loan_amount_ne_focus_commitment"));

    // Matching is fine.
    assert!(!has_error(&base(), "loan_amount_ne_focus_commitment"));
}

#[test]
fn debt_deal_requires_loan_type_and_rank() {
    let mut d = base();
    d["classification"]["loan_type"] = json!(null);
    assert!(has_error(&d, "debt_deal_missing_loan_type"));

    let mut d2 = base();
    d2["classification"]["loan_rank"] = json!(null);
    assert!(has_error(&d2, "debt_deal_missing_loan_rank"));
}

#[test]
fn phase_units_must_not_double_count() {
    let mut d = base();
    d["canonical_deal"]["phases"] = json!([
        {"id":"p1","unit_count":148},{"id":"p2","unit_count":41}
    ]);
    d["details"]["residential_units"] = json!({"value": {"value": 337}});
    assert!(has_error(&d, "unit_count_mismatch"));
}

#[test]
fn warn_issues_do_not_count_as_errors() {
    // gross_debt scope mismatch is a warn — never drives repair.
    let mut d = base();
    d["details"]["gross_debt"] = json!({"value": {"amount": "999000000"}});
    let issues = validate::validate(&d);
    let warn = issues
        .iter()
        .find(|i| i.code == "gross_debt_scope_mismatch");
    assert!(
        warn.is_some(),
        "expected gross_debt_scope_mismatch, got {issues:?}"
    );
    assert_eq!(warn.unwrap().severity, Severity::Warn);
    // The warn must not appear in the error count — count only errors with
    // this code, not schema completeness errors on the minimal fixture.
    assert!(!issues
        .iter()
        .any(|i| i.code == "gross_debt_scope_mismatch" && i.severity == Severity::Error));
}
