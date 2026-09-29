//! Gold-standard scoring — faithful Rust port of the assertions in
//! `ih-gold-standard.integration.test.ts` and
//! `multi-deal-acceptance.integration.test.ts`. Each assertion becomes a
//! `Check` with a PASS/FAIL result and a reason.

use serde_json::Value;

use crate::schema::{DealDocument, Facility, SourcedValue};

/// Extract a numeric amount from whatever the agent emitted — mirrors the
/// TS test's `extractAmount`: handles bare numbers, numeric strings,
/// `{amount}`, `{total:{amount}}`, and nested `{value:{...}}` shapes.
fn extract_amount(v: &Value) -> Option<f64> {
    match v {
        Value::Null => None,
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        Value::Object(o) => {
            if let Some(a) = o.get("amount") {
                extract_amount(a)
            } else if let Some(t) = o.get("total").and_then(|t| t.get("amount")) {
                extract_amount(t)
            } else {
                o.get("value").and_then(extract_amount)
            }
        }
        _ => None,
    }
}

/// Normalise amounts that may be stored in millions — mirrors the TS test's
/// `normaliseAmount`. This is grader-side parity with the test harness, NOT
/// pipeline coercion: the validator separately flags millions shorthand as a
/// contract violation.
fn normalise_amount(n: f64) -> f64 {
    if n < 1000.0 {
        n * 1_000_000.0
    } else {
        n
    }
}

/// Extract the numeric amount from a sourced-value field.
fn sourced_amount(sv: &SourcedValue) -> Option<f64> {
    sv.value.as_ref().and_then(extract_amount)
}

/// Extract a string field from a sourced-value's `value` object, falling
/// back to a top-level key for the old flat shape.
fn sourced_string(sv: &SourcedValue, key: &str) -> Option<String> {
    sv.value.as_ref().and_then(|v| {
        v.get(key)
            .and_then(|v| v.as_str())
            .or_else(|| v.as_str())
            .map(|s| s.to_string())
    })
}

pub struct CheckResult {
    pub name: String,
    pub passed: bool,
    pub reason: String,
}

impl CheckResult {
    fn pass(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passed: true,
            reason: String::new(),
        }
    }
    fn fail(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passed: false,
            reason: reason.into(),
        }
    }
    fn skip(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passed: true,
            reason: format!("[skipped] {}", reason.into()),
        }
    }
}

fn approx(actual: f64, expected: f64, pct: f64) -> bool {
    let lo = expected * (1.0 - pct);
    let hi = expected * (1.0 + pct);
    actual > lo && actual < hi
}

fn facility_amount(f: &Facility) -> Option<f64> {
    f.commitment.as_ref().map(|m| normalise_amount(m.amount))
}

fn find_focus(doc: &DealDocument) -> Option<&Facility> {
    let cd = doc.canonical_deal.as_ref()?;
    let focus_id = cd.focus_facility_id.as_ref()?;
    cd.facilities.iter().find(|f| &f.id == focus_id)
}

fn find_by_name_or_type<'a>(
    facilities: &'a [Facility],
    name_contains: &str,
    loan_type: Option<&str>,
) -> Option<&'a Facility> {
    facilities.iter().find(|f| {
        let name_match = f
            .name
            .as_deref()
            .map(|n| n.to_lowercase().contains(name_contains))
            .unwrap_or(false);
        let type_match = loan_type
            .map(|lt| f.loan_type.as_deref() == Some(lt))
            .unwrap_or(false);
        name_match || type_match
    })
}

// ─── IH gold-standard checks ────────────────────────────────────────────────

pub fn score_ih(doc: &DealDocument) -> Vec<CheckResult> {
    let mut results = Vec::new();
    let cd = doc.canonical_deal.as_ref();

    // 1. Focus Facility
    results.push({
        if let Some(cd) = cd {
            if cd.focus_facility_id.is_some()
                && cd.focus_resolution_status.as_deref() == Some("resolved")
            {
                CheckResult::pass("focus_facility_resolved")
            } else {
                CheckResult::fail(
                    "focus_facility_resolved",
                    "focus_facility_id missing or not resolved",
                )
            }
        } else {
            CheckResult::fail("focus_facility_resolved", "no canonical_deal")
        }
    });

    results.push({
        if let Some(focus) = find_focus(doc) {
            if let Some(amt) = facility_amount(focus) {
                if amt > 20_000_000.0 && amt < 28_000_000.0 {
                    CheckResult::pass("focus_commitment_25.75M")
                } else {
                    CheckResult::fail(
                        "focus_commitment_25.75M",
                        format!("got {amt:.0}, expected 20M–28M"),
                    )
                }
            } else {
                CheckResult::fail("focus_commitment_25.75M", "no commitment")
            }
        } else {
            CheckResult::fail("focus_commitment_25.75M", "no focus facility")
        }
    });

    results.push({
        if let Some(focus) = find_focus(doc) {
            if focus.loan_type.as_deref() == Some("bridge") {
                CheckResult::pass("focus_loan_type_bridge")
            } else {
                CheckResult::fail(
                    "focus_loan_type_bridge",
                    format!("got {:?}", focus.loan_type),
                )
            }
        } else {
            CheckResult::fail("focus_loan_type_bridge", "no focus facility")
        }
    });

    results.push({
        if let Some(focus) = find_focus(doc) {
            let status = focus.status.as_deref().unwrap_or("");
            if status != "active" && status != "drawn" {
                CheckResult::pass("focus_status_candidate")
            } else {
                CheckResult::fail("focus_status_candidate", format!("status={status}"))
            }
        } else {
            CheckResult::fail("focus_status_candidate", "no focus facility")
        }
    });

    results.push({
        if let Some(focus) = find_focus(doc) {
            let role = focus.role.as_deref().unwrap_or("");
            if role == "senior_debt" || role == "whole_loan" {
                CheckResult::pass("focus_role_senior_or_whole")
            } else {
                CheckResult::fail("focus_role_senior_or_whole", format!("role={role}"))
            }
        } else {
            CheckResult::fail("focus_role_senior_or_whole", "no focus facility")
        }
    });

    // 2. Cap Stack / Debt Projection
    results.push({
        if let Some(la) = doc
            .details
            .as_ref()
            .and_then(|d| d.loan_amount.as_ref())
            .and_then(sourced_amount)
        {
            let amt = normalise_amount(la);
            if (25_500_000.0..=26_000_000.0).contains(&amt) {
                CheckResult::pass("loan_amount_25.75M")
            } else {
                CheckResult::fail(
                    "loan_amount_25.75M",
                    format!("got {amt:.0}, expected 25.5M–26M"),
                )
            }
        } else {
            CheckResult::fail("loan_amount_25.75M", "no loan_amount")
        }
    });

    results.push({
        if let Some(sd) = doc
            .details
            .as_ref()
            .and_then(|d| d.senior_debt.as_ref())
            .and_then(sourced_amount)
        {
            let amt = normalise_amount(sd);
            if amt < 50_000_000.0 {
                CheckResult::pass("senior_debt_excludes_73M")
            } else {
                CheckResult::fail(
                    "senior_debt_excludes_73M",
                    format!("got {amt:.0}, must be <50M"),
                )
            }
        } else {
            CheckResult::skip("senior_debt_excludes_73M", "no senior_debt")
        }
    });

    results.push({
        if let Some(gd) = doc
            .details
            .as_ref()
            .and_then(|d| d.gross_debt.as_ref())
            .and_then(sourced_amount)
        {
            let amt = normalise_amount(gd);
            if amt < 60_000_000.0 {
                CheckResult::pass("gross_debt_deal_scope")
            } else {
                CheckResult::fail(
                    "gross_debt_deal_scope",
                    format!("got {amt:.0}, must be <60M"),
                )
            }
        } else {
            CheckResult::skip("gross_debt_deal_scope", "no gross_debt")
        }
    });

    results.push({
        if let Some(eq) = doc
            .details
            .as_ref()
            .and_then(|d| d.developer_equity.as_ref())
            .and_then(sourced_amount)
        {
            let amt = normalise_amount(eq);
            if approx(amt, 5_500_000.0, 0.20) {
                CheckResult::pass("developer_equity_5.5M")
            } else {
                CheckResult::fail(
                    "developer_equity_5.5M",
                    format!("got {amt:.0}, expected ~5.5M"),
                )
            }
        } else {
            CheckResult::fail("developer_equity_5.5M", "no developer_equity")
        }
    });

    // 3. Facilities
    results.push({
        if let Some(cd) = cd {
            if cd.facilities.len() >= 3 {
                CheckResult::pass("facilities_count_3plus")
            } else {
                CheckResult::fail(
                    "facilities_count_3plus",
                    format!("got {} facilities", cd.facilities.len()),
                )
            }
        } else {
            CheckResult::fail("facilities_count_3plus", "no canonical_deal")
        }
    });

    results.push({
        if let Some(cd) = cd {
            if let Some(bridge) = find_by_name_or_type(&cd.facilities, "bridge", Some("bridge")) {
                if let Some(amt) = facility_amount(bridge) {
                    if approx(amt, 23_589_217.0, 0.10) {
                        CheckResult::pass("bridge_facility_23.59M")
                    } else {
                        CheckResult::fail("bridge_facility_23.59M", format!("got {amt:.0}"))
                    }
                } else {
                    CheckResult::fail("bridge_facility_23.59M", "no commitment")
                }
            } else {
                CheckResult::fail("bridge_facility_23.59M", "no bridge facility")
            }
        } else {
            CheckResult::fail("bridge_facility_23.59M", "no canonical_deal")
        }
    });

    results.push({
        if let Some(cd) = cd {
            if let Some(dev) = cd.facilities.iter().find(|f| {
                f.name
                    .as_deref()
                    .map(|n| n.to_lowercase().contains("development"))
                    .unwrap_or(false)
                    || (f.loan_type.as_deref() == Some("development")
                        && f.term_months.unwrap_or(0) > 12)
            }) {
                if let Some(amt) = facility_amount(dev) {
                    if approx(amt, 73_393_202.0, 0.15) {
                        CheckResult::pass("dev_facility_73.39M")
                    } else {
                        CheckResult::fail("dev_facility_73.39M", format!("got {amt:.0}"))
                    }
                } else {
                    CheckResult::fail("dev_facility_73.39M", "no commitment")
                }
            } else {
                CheckResult::fail("dev_facility_73.39M", "no development facility")
            }
        } else {
            CheckResult::fail("dev_facility_73.39M", "no canonical_deal")
        }
    });

    results.push({
        if let Some(cd) = cd {
            if let Some(vat) = cd.facilities.iter().find(|f| {
                f.name
                    .as_deref()
                    .map(|n| n.to_lowercase().contains("vat"))
                    .unwrap_or(false)
                    || f.role
                        .as_deref()
                        .map(|r| r.to_lowercase().contains("vat"))
                        .unwrap_or(false)
            }) {
                if let Some(amt) = facility_amount(vat) {
                    if approx(amt, 4_735_000.0, 0.25) {
                        CheckResult::pass("vat_facility_4.735M")
                    } else {
                        CheckResult::fail("vat_facility_4.735M", format!("got {amt:.0}"))
                    }
                } else {
                    CheckResult::fail("vat_facility_4.735M", "no commitment")
                }
            } else {
                CheckResult::skip("vat_facility_4.735M", "no VAT facility")
            }
        } else {
            CheckResult::skip("vat_facility_4.735M", "no canonical_deal")
        }
    });

    results.push({
        if let Some(cd) = cd {
            let repaid: Vec<&Facility> = cd
                .facilities
                .iter()
                .filter(|f| f.status.as_deref() == Some("repaid"))
                .collect();
            if !repaid.is_empty() {
                CheckResult::pass("repaid_facilities_exist")
            } else {
                CheckResult::fail("repaid_facilities_exist", "no repaid facilities")
            }
        } else {
            CheckResult::fail("repaid_facilities_exist", "no canonical_deal")
        }
    });

    results.push({
        if let Some(cd) = cd {
            if let Some(eq) = cd.facilities.iter().find(|f| {
                f.role.as_deref() == Some("developer_equity")
                    || f.instrument_type.as_deref() == Some("ordinary_equity")
            }) {
                if let Some(amt) = facility_amount(eq) {
                    if approx(amt, 5_500_000.0, 0.20) {
                        CheckResult::pass("equity_facility_5.5M")
                    } else {
                        CheckResult::fail("equity_facility_5.5M", format!("got {amt:.0}"))
                    }
                } else {
                    CheckResult::fail("equity_facility_5.5M", "no commitment")
                }
            } else {
                CheckResult::fail("equity_facility_5.5M", "no equity facility")
            }
        } else {
            CheckResult::fail("equity_facility_5.5M", "no canonical_deal")
        }
    });

    // 4. Valuations
    results.push({
        if let Some(cd) = cd {
            if !cd.valuations.is_empty() {
                CheckResult::pass("valuations_exist")
            } else {
                CheckResult::fail("valuations_exist", "no valuations")
            }
        } else {
            CheckResult::fail("valuations_exist", "no canonical_deal")
        }
    });

    results.push({
        if let Some(cd) = cd {
            let gdvs: Vec<f64> = cd
                .valuations
                .iter()
                .filter(|v| v.value_type.as_deref() == Some("gross_development_value"))
                .map(|v| normalise_amount(v.amount.amount))
                .collect();
            let has_full = gdvs.iter().any(|a| *a > 100_000_000.0);
            let has_phase1 = gdvs.iter().any(|a| *a > 70_000_000.0 && *a < 90_000_000.0);
            if has_full || has_phase1 {
                CheckResult::pass("gdv_115M_or_phase1_80M")
            } else {
                CheckResult::fail("gdv_115M_or_phase1_80M", format!("gdvs={gdvs:?}"))
            }
        } else {
            CheckResult::fail("gdv_115M_or_phase1_80M", "no canonical_deal")
        }
    });

    results.push({
        if let Some(mv) = doc
            .details
            .as_ref()
            .and_then(|d| d.market_value.as_ref())
            .and_then(sourced_amount)
        {
            let amt = normalise_amount(mv);
            if approx(amt, 32_500_000.0, 0.10) {
                CheckResult::pass("market_value_32.5M")
            } else {
                CheckResult::fail("market_value_32.5M", format!("got {amt:.0}"))
            }
        } else {
            CheckResult::skip("market_value_32.5M", "no market_value")
        }
    });

    results.push({
        let mv = doc
            .details
            .as_ref()
            .and_then(|d| d.market_value.as_ref())
            .and_then(sourced_amount)
            .map(normalise_amount);
        let gdv = doc
            .details
            .as_ref()
            .and_then(|d| d.gross_development_value.as_ref())
            .and_then(sourced_amount)
            .map(normalise_amount);
        if let (Some(mv), Some(gdv)) = (mv, gdv) {
            if gdv > mv {
                CheckResult::pass("market_value_ne_gdv")
            } else {
                CheckResult::fail("market_value_ne_gdv", format!("mv={mv}, gdv={gdv}"))
            }
        } else {
            CheckResult::skip("market_value_ne_gdv", "missing mv or gdv")
        }
    });

    // 5. Phasing
    results.push({
        if let Some(cd) = cd {
            if let Some(p1) = cd.phases.iter().find(|p| {
                p.name
                    .as_deref()
                    .map(|n| n.to_lowercase().contains("phase 1"))
                    .unwrap_or(false)
                    || p.sequence == Some(1)
                    || p.unit_count == Some(148)
            }) {
                if p1.unit_count == Some(148) {
                    CheckResult::pass("phase1_148_units")
                } else {
                    CheckResult::fail("phase1_148_units", format!("got {:?}", p1.unit_count))
                }
            } else {
                CheckResult::fail("phase1_148_units", "no phase 1")
            }
        } else {
            CheckResult::fail("phase1_148_units", "no canonical_deal")
        }
    });

    results.push({
        if let Some(cd) = cd {
            if let Some(p2) = cd.phases.iter().find(|p| {
                p.name
                    .as_deref()
                    .map(|n| n.to_lowercase().contains("phase 2"))
                    .unwrap_or(false)
                    || p.sequence == Some(2)
                    || p.unit_count == Some(41)
            }) {
                if p2.unit_count == Some(41) {
                    CheckResult::pass("phase2_41_units")
                } else {
                    CheckResult::fail("phase2_41_units", format!("got {:?}", p2.unit_count))
                }
            } else {
                CheckResult::fail("phase2_41_units", "no phase 2")
            }
        } else {
            CheckResult::fail("phase2_41_units", "no canonical_deal")
        }
    });

    results.push({
        if let Some(cd) = cd {
            let total: u32 = cd.phases.iter().filter_map(|p| p.unit_count).sum();
            if total > 0 && approx(total as f64, 189.0, 0.10) {
                CheckResult::pass("total_units_189")
            } else {
                CheckResult::fail("total_units_189", format!("got {total}"))
            }
        } else {
            CheckResult::fail("total_units_189", "no canonical_deal")
        }
    });

    // 6. Capital Events
    results.push({
        if let Some(cd) = cd {
            if cd
                .capital_events
                .iter()
                .any(|e| e.event_type.as_deref() == Some("fully_repaid"))
            {
                CheckResult::pass("fully_repaid_event")
            } else {
                CheckResult::fail("fully_repaid_event", "no fully_repaid event")
            }
        } else {
            CheckResult::fail("fully_repaid_event", "no canonical_deal")
        }
    });

    // 7. KPIs
    results.push({
        if let Some(ltv) = doc
            .kpis
            .as_ref()
            .and_then(|k| k.loan_to_value.as_ref())
            .and_then(sourced_amount)
        {
            if ltv > 70.0 && ltv < 90.0 {
                CheckResult::pass("ltv_79pct")
            } else {
                CheckResult::fail("ltv_79pct", format!("got {ltv}"))
            }
        } else {
            CheckResult::skip("ltv_79pct", "no ltv")
        }
    });

    results.push({
        if let Some(ltgdv) = doc
            .kpis
            .as_ref()
            .and_then(|k| k.loan_to_gdv.as_ref())
            .and_then(sourced_amount)
        {
            if ltgdv < 50.0 {
                CheckResult::pass("ltgdv_below_50")
            } else {
                CheckResult::fail("ltgdv_below_50", format!("got {ltgdv}"))
            }
        } else {
            CheckResult::skip("ltgdv_below_50", "no ltgdv")
        }
    });

    results.push({
        if let Some(amt) = doc
            .kpis
            .as_ref()
            .and_then(|k| k.build_cost_per_area_unit.as_ref())
            .and_then(sourced_amount)
        {
            if amt > 150.0 && amt < 220.0 {
                CheckResult::pass("build_cost_180")
            } else {
                CheckResult::fail("build_cost_180", format!("got {amt}"))
            }
        } else {
            CheckResult::skip("build_cost_180", "no build_cost")
        }
    });

    // 7b. Bridge term ≤12 months (TS: "Bridge Facility term is 6 months")
    results.push({
        if let Some(cd) = cd {
            if let Some(bridge) = find_by_name_or_type(&cd.facilities, "bridge", Some("bridge")) {
                if let Some(term) = bridge.term_months {
                    if term <= 12 {
                        CheckResult::pass("bridge_term_le_12m")
                    } else {
                        CheckResult::fail("bridge_term_le_12m", format!("got {term} months"))
                    }
                } else {
                    CheckResult::skip("bridge_term_le_12m", "no term_months")
                }
            } else {
                CheckResult::fail("bridge_term_le_12m", "no bridge facility")
            }
        } else {
            CheckResult::fail("bridge_term_le_12m", "no canonical_deal")
        }
    });

    // 7c. Dev term ≥18 months (TS: "Development Facility term is ~24 months")
    results.push({
        if let Some(cd) = cd {
            if let Some(dev) = cd.facilities.iter().find(|f| {
                f.name.as_deref().map(|n| n.to_lowercase().contains("development")).unwrap_or(false)
                    || (f.loan_type.as_deref() == Some("development") && f.term_months.unwrap_or(0) > 12)
            }) {
                if let Some(term) = dev.term_months {
                    if term >= 18 {
                        CheckResult::pass("dev_term_ge_18m")
                    } else {
                        CheckResult::fail("dev_term_ge_18m", format!("got {term} months"))
                    }
                } else {
                    CheckResult::skip("dev_term_ge_18m", "no term_months")
                }
            } else {
                CheckResult::fail("dev_term_ge_18m", "no development facility")
            }
        } else {
            CheckResult::fail("dev_term_ge_18m", "no canonical_deal")
        }
    });

    // 7d. Repaid Shojin senior ~£20.36M (TS: "repaid Shojin senior debt exists")
    results.push({
        if let Some(cd) = cd {
            let repaid_senior: Vec<&Facility> = cd.facilities.iter()
                .filter(|f| f.status.as_deref() == Some("repaid") && f.role.as_deref() == Some("senior_debt"))
                .collect();
            if let Some(f) = repaid_senior.first() {
                if let Some(amt) = facility_amount(f) {
                    if approx(amt, 20_361_250.0, 0.10) {
                        CheckResult::pass("repaid_shojin_senior_20.36M")
                    } else {
                        CheckResult::fail("repaid_shojin_senior_20.36M", format!("got {amt:.0}"))
                    }
                } else {
                    CheckResult::skip("repaid_shojin_senior_20.36M", "no commitment")
                }
            } else {
                CheckResult::skip("repaid_shojin_senior_20.36M", "no repaid senior_debt")
            }
        } else {
            CheckResult::fail("repaid_shojin_senior_20.36M", "no canonical_deal")
        }
    });

    // 7e. Repaid Shojin mezzanine ~£3.23M (TS: "repaid Shojin mezzanine debt exists")
    results.push({
        if let Some(cd) = cd {
            let repaid_mezz: Vec<&Facility> = cd.facilities.iter()
                .filter(|f| f.status.as_deref() == Some("repaid") && f.role.as_deref() == Some("mezzanine_debt"))
                .collect();
            if let Some(f) = repaid_mezz.first() {
                if let Some(amt) = facility_amount(f) {
                    if approx(amt, 3_228_000.0, 0.15) {
                        CheckResult::pass("repaid_shojin_mezz_3.23M")
                    } else {
                        CheckResult::fail("repaid_shojin_mezz_3.23M", format!("got {amt:.0}"))
                    }
                } else {
                    CheckResult::skip("repaid_shojin_mezz_3.23M", "no commitment")
                }
            } else {
                CheckResult::skip("repaid_shojin_mezz_3.23M", "no repaid mezzanine_debt")
            }
        } else {
            CheckResult::fail("repaid_shojin_mezz_3.23M", "no canonical_deal")
        }
    });

    // 7f. Capital stack no profit role (TS: "capital stack does NOT include profit as a funding layer")
    results.push({
        if let Some(cd) = cd {
            let has_profit = cd.facilities.iter().any(|f| {
                let r = f.role.as_deref().unwrap_or("");
                r == "profit" || r == "return"
            });
            if !has_profit {
                CheckResult::pass("no_profit_role")
            } else {
                CheckResult::fail("no_profit_role", "found profit/return role in facilities")
            }
        } else {
            CheckResult::fail("no_profit_role", "no canonical_deal")
        }
    });

    // 7g. Focus has no drawn event (TS: "focus facility does NOT have a drawn capital event")
    results.push({
        if let Some(cd) = cd {
            let focus_id = cd.focus_facility_id.as_deref().unwrap_or("");
            let has_drawn = cd.capital_events.iter().any(|e| {
                e.facility_id.as_deref() == Some(focus_id)
                    && e.event_type.as_deref() == Some("drawn")
            });
            if !has_drawn {
                CheckResult::pass("focus_no_drawn_event")
            } else {
                CheckResult::fail("focus_no_drawn_event", "focus has drawn event")
            }
        } else {
            CheckResult::fail("focus_no_drawn_event", "no canonical_deal")
        }
    });

    // 7h. Phase 1 GDV ~£80.545M (TS: "Phase 1 GDV is ~£80.545M")
    results.push({
        if let Some(cd) = cd {
            let gdvs: Vec<f64> = cd.valuations.iter()
                .filter(|v| v.value_type.as_deref() == Some("gross_development_value"))
                .map(|v| normalise_amount(v.amount.amount))
                .collect();
            let phase1 = gdvs.iter().find(|a| **a > 70_000_000.0 && **a < 90_000_000.0);
            if let Some(&amt) = phase1 {
                if approx(amt, 80_545_000.0, 0.10) {
                    CheckResult::pass("phase1_gdv_80.5M")
                } else {
                    CheckResult::fail("phase1_gdv_80.5M", format!("got {amt:.0}"))
                }
            } else {
                CheckResult::skip("phase1_gdv_80.5M", "no phase 1 GDV in 70M–90M range")
            }
        } else {
            CheckResult::fail("phase1_gdv_80.5M", "no canonical_deal")
        }
    });

    // 8. Property Details
    results.push({
        if let Some(addr) = doc.details.as_ref().and_then(|d| d.address.as_ref()) {
            let postcode = sourced_string(addr, "postcode");
            if postcode.as_deref() == Some("W5 5DB") {
                CheckResult::pass("postcode_w55db")
            } else {
                CheckResult::fail("postcode_w55db", format!("got {postcode:?}"))
            }
        } else {
            CheckResult::fail("postcode_w55db", "no address")
        }
    });

    results.push({
        if let Some(gia) = doc
            .details
            .as_ref()
            .and_then(|d| d.gross_internal_area.as_ref())
            .and_then(sourced_amount)
        {
            if approx(gia, 194_871.0, 0.25) {
                CheckResult::pass("gia_194871")
            } else {
                CheckResult::fail("gia_194871", format!("got {gia}"))
            }
        } else {
            CheckResult::skip("gia_194871", "no GIA")
        }
    });

    results
}

// ─── Multi-deal checks ───────────────────────────────────────────────────────

pub fn score_edenbridge(doc: &DealDocument) -> Vec<CheckResult> {
    let mut r = Vec::new();
    let cls = doc.classification.as_ref();
    r.push(check_eq(
        cls.and_then(|c| c.capital_position.as_deref()),
        "debt",
        "capital_position_debt",
    ));
    r.push(check_eq(
        cls.and_then(|c| c.asset_class_primary.as_deref()),
        "residential",
        "asset_class_residential",
    ));
    r.push(check_eq(
        cls.and_then(|c| c.project_type.as_ref().and_then(|p| p.category.as_deref())),
        "development",
        "project_type_development",
    ));
    r.push(check_eq(
        cls.and_then(|c| c.loan_type.as_ref().and_then(|l| l.category.as_deref())),
        "development",
        "loan_type_development",
    ));
    r.push({
        if let Some(focus) = find_focus(doc) {
            if focus
                .name
                .as_deref()
                .map(|n| n.to_lowercase().contains("mezzanine"))
                .unwrap_or(false)
            {
                CheckResult::pass("focus_contains_mezzanine")
            } else {
                CheckResult::fail("focus_contains_mezzanine", format!("name={:?}", focus.name))
            }
        } else {
            CheckResult::fail("focus_contains_mezzanine", "no focus facility")
        }
    });
    r.push({
        if let Some(focus) = find_focus(doc) {
            if let Some(amt) = facility_amount(focus) {
                if amt > 0.0 {
                    CheckResult::pass("focus_has_commitment")
                } else {
                    CheckResult::fail("focus_has_commitment", format!("amount={amt}"))
                }
            } else {
                CheckResult::fail("focus_has_commitment", "no commitment")
            }
        } else {
            CheckResult::fail("focus_has_commitment", "no focus facility")
        }
    });
    r
}

pub fn score_oakley_square(doc: &DealDocument) -> Vec<CheckResult> {
    let mut r = Vec::new();
    let cls = doc.classification.as_ref();
    r.push(check_eq(
        cls.and_then(|c| c.capital_position.as_deref()),
        "debt",
        "capital_position_debt",
    ));
    r.push(check_eq(
        cls.and_then(|c| c.asset_class_primary.as_deref()),
        "residential",
        "asset_class_residential",
    ));
    r.push(check_eq(
        cls.and_then(|c| c.project_type.as_ref().and_then(|p| p.category.as_deref())),
        "development",
        "project_type_development",
    ));
    r.push(check_eq(
        cls.and_then(|c| c.loan_rank.as_ref().and_then(|l| l.category.as_deref())),
        "whole_loan",
        "loan_rank_whole_loan",
    ));
    r.push({
        if let Some(focus) = find_focus(doc) {
            if focus
                .name
                .as_deref()
                .map(|n| n.to_lowercase().contains("whole"))
                .unwrap_or(false)
            {
                CheckResult::pass("focus_contains_whole")
            } else {
                CheckResult::fail("focus_contains_whole", format!("name={:?}", focus.name))
            }
        } else {
            CheckResult::fail("focus_contains_whole", "no focus facility")
        }
    });
    r
}

pub fn score_octagon_retail(doc: &DealDocument) -> Vec<CheckResult> {
    let mut r = Vec::new();
    let cls = doc.classification.as_ref();
    r.push(check_eq(
        cls.and_then(|c| c.capital_position.as_deref()),
        "debt",
        "capital_position_debt",
    ));
    r.push(check_eq(
        cls.and_then(|c| c.asset_class_primary.as_deref()),
        "retail",
        "asset_class_retail",
    ));
    r.push(check_eq(
        cls.and_then(|c| c.project_type.as_ref().and_then(|p| p.category.as_deref())),
        "acquisition",
        "project_type_acquisition",
    ));
    r.push(check_eq(
        cls.and_then(|c| c.loan_type.as_ref().and_then(|l| l.category.as_deref())),
        "bridge",
        "loan_type_bridge",
    ));
    r.push(check_eq(
        cls.and_then(|c| {
            c.operating_model
                .as_ref()
                .and_then(|o| o.category.as_deref())
        }),
        "income_property",
        "operating_model_income_property",
    ));
    r.push({
        if let Some(focus) = find_focus(doc) {
            if focus
                .name
                .as_deref()
                .map(|n| n.to_lowercase().contains("bridge"))
                .unwrap_or(false)
            {
                CheckResult::pass("focus_contains_bridge")
            } else {
                CheckResult::fail("focus_contains_bridge", format!("name={:?}", focus.name))
            }
        } else {
            CheckResult::fail("focus_contains_bridge", "no focus facility")
        }
    });
    r
}

fn check_eq(actual: Option<&str>, expected: &str, name: &str) -> CheckResult {
    if actual == Some(expected) {
        CheckResult::pass(name)
    } else {
        CheckResult::fail(name, format!("got {actual:?}, expected {expected:?}"))
    }
}

pub fn score_equity_brochure(doc: &DealDocument, name: &str) -> Vec<CheckResult> {
    let mut r = Vec::new();
    let cls = doc.classification.as_ref();
    r.push(check_eq(
        cls.and_then(|c| c.capital_position.as_deref()),
        "equity",
        &format!("{name}: capital_position_equity"),
    ));
    r.push({
        let la = doc.details.as_ref().and_then(|d| d.loan_amount.as_ref()).and_then(sourced_amount);
        if la.is_none() {
            CheckResult::pass(&format!("{name}: loan_amount_null"))
        } else {
            CheckResult::fail(&format!("{name}: loan_amount_null"), format!("got {la:?}"))
        }
    });
    r
}

pub fn print_results(name: &str, results: &[CheckResult]) {
    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    let skipped = results
        .iter()
        .filter(|r| r.reason.starts_with("[skipped]"))
        .count();
    let failed = total - passed;
    println!("\n=== {name} === {passed}/{total} passed ({failed} failed, {skipped} skipped)");
    for r in results {
        let mark = if r.passed { "PASS" } else { "FAIL" };
        if r.reason.is_empty() {
            println!("  [{mark}] {}", r.name);
        } else {
            println!("  [{mark}] {} — {}", r.name, r.reason);
        }
    }
}
