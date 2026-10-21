//! Tie checks — cross-field reconciliation of extracted figures. Every check
//! only runs when all its inputs are present; percentages are normalised
//! (4.5 and 0.045 both read as 4.5%).

use serde_json::Value;

#[derive(Debug, Clone)]
pub struct Tie {
    pub id: &'static str,
    pub status: &'static str, // "pass" | "fail"
    pub detail: String,
    pub fields: Vec<String>,
}

fn money(doc: &Value, path: &str) -> Option<f64> {
    let mut cur = doc;
    for seg in path.split('.') {
        cur = cur.get(seg)?;
    }
    let v = cur.get("value").unwrap_or(cur);
    let v = if v.is_object() && v.get("amount").is_some() {
        v.get("amount")?
    } else {
        v
    };
    v.as_f64()
        .or_else(|| {
            v.as_str()?
                .replace([',', '£', '$', '€'], "")
                .parse::<f64>()
                .ok()
        })
        .or_else(|| v.get("amount")?.as_f64())
}

/// Normalise a percentage field: 4.5 → 4.5, 0.045 → 4.5.
fn pct(doc: &Value, path: &str) -> Option<f64> {
    let mut cur = doc;
    for seg in path.split('.') {
        cur = cur.get(seg)?;
    }
    let v = cur.get("value").unwrap_or(cur);
    let raw = if v.is_object() {
        v.get("percentage")
            .or_else(|| v.get("value"))
            .or_else(|| v.get("yield"))
            .or_else(|| v.get("residential_and_parking_percentage"))
            .or_else(|| v.get("amount"))?
    } else {
        v
    };
    let n = raw
        .as_f64()
        .or_else(|| raw.as_str()?.replace('%', "").trim().parse::<f64>().ok())?;
    Some(if n > 0.0 && n < 1.0 { n * 100.0 } else { n })
}

/// The four reconciliation checks. Returns one Tie per check whose inputs
/// exist — a missing input means the check is skipped, not failed.
#[must_use]
pub fn tie_checks(doc: &Value) -> Vec<Tie> {
    let mut out = Vec::new();
    let d = |k: &str| money(doc, &format!("details.{k}"));

    // NOI ≈ (ERV or passing rent) − opex (amount, or % × rent).
    if let Some(noi) = d("net_operating_income") {
        let rent = d("estimated_rental_value").or_else(|| d("passing_rent_per_annum"));
        let opex_amt = d("operating_costs");
        let opex_pct = pct(doc, "details.operating_costs");
        if let Some(rent) = rent {
            if let Some(oamt) = opex_amt.filter(|a| *a > 0.0 && *a < rent) {
                let implied = rent - oamt;
                let ok = (noi - implied).abs() <= implied * 0.02;
                let gross = (noi - rent).abs() <= rent * 0.02;
                let detail = if ok {
                    format!("NOI {noi:.0} ≈ rent {rent:.0} − opex {oamt:.0}")
                } else if gross {
                    format!("NOI {noi:.0} equals gross rent {rent:.0}; opex {oamt:.0} not deducted")
                } else {
                    format!(
                        "NOI {noi:.0} differs from rent {rent:.0} − opex {oamt:.0} = {implied:.0} — check scope: fields may mix residential-only and whole-scheme figures"
                    )
                };
                out.push(Tie {
                    id: "noi_vs_rent_opex",
                    status: if ok { "pass" } else { "fail" },
                    detail,
                    fields: vec![
                        "details.net_operating_income".into(),
                        "details.operating_costs".into(),
                    ],
                });
            } else if let Some(op) = opex_pct.filter(|p| *p > 0.0 && *p < 80.0) {
                let implied = rent * (1.0 - op / 100.0);
                let ok = (noi - implied).abs() <= implied * 0.02;
                let gross = (noi - rent).abs() <= rent * 0.02;
                let detail = if ok {
                    format!("NOI {noi:.0} ≈ rent {rent:.0} × (1 − {op:.2}%)")
                } else if gross {
                    format!("NOI {noi:.0} equals gross rent {rent:.0}; opex {op:.2}% not deducted")
                } else {
                    format!(
                        "NOI {noi:.0} differs from rent {rent:.0} × (1 − {op:.2}%) = {implied:.0}"
                    )
                };
                out.push(Tie {
                    id: "noi_vs_rent_opex",
                    status: if ok { "pass" } else { "fail" },
                    detail,
                    fields: vec![
                        "details.net_operating_income".into(),
                        "details.operating_costs".into(),
                    ],
                });
            }
        }
    }

    // Implied cap = NOI ÷ (GDV ÷ (1 − purchaser's costs%)) vs stated yield.
    if let (Some(noi), Some(gdv), Some(stated)) = (
        d("net_operating_income"),
        d("gross_development_value"),
        pct(doc, "details.stabilised_yield"),
    ) {
        let pc = pct(doc, "details.purchaser_costs").unwrap_or(0.0) / 100.0;
        if gdv > 0.0 {
            let implied = noi / (gdv / (1.0 - pc)) * 100.0;
            // Only meaningful when the GDV is an INVESTMENT value (income /
            // BTR exit): on a unit-sales scheme the cap vs NOI comparison is
            // a category error (IH sales GDV £115.2m vs stated cap 5.2% on a
            // much smaller income stream). Investment-like = a stabilised
            // valuation exists, no residential unit-sales schedule, or the
            // stated cap actually explains the GDV (NOI ÷ cap ≈ GDV ±50%).
            let resi_units = d("residential_units").unwrap_or(0.0);
            let stabilised = doc
                .get("canonical_deal")
                .and_then(|c| c.get("valuations"))
                .and_then(Value::as_array)
                .is_some_and(|vs| vs.iter().any(|v| {
                    v.get("value_type").and_then(Value::as_str) == Some("stabilised_value")
                }));
            let investment_like = stabilised || resi_units <= 0.0 || (implied - stated).abs() <= 0.5;
            if !investment_like {
                // Sales GDV — a NOI/cap tie would misfire and burn a fix call.
            } else {
            let ok = (implied - stated).abs() <= 0.25;
            out.push(Tie {
                id: "cap_vs_noi_value",
                status: if ok { "pass" } else { "fail" },
                detail: format!("stated cap {stated:.2}% vs implied {implied:.2}% (NOI {noi:.0} ÷ GDV {gdv:.0} net of {:.1}% costs)", pc * 100.0),
                fields: vec!["details.stabilised_yield".into(), "details.net_operating_income".into(), "details.gross_development_value".into()],
            });
            }
        }
    }

    // TDC ≤ TIC; when finance costs are stated, TIC − TDC ≈ finance_costs.
    if let (Some(tdc), Some(tic)) = (d("total_development_cost"), d("total_investment_cost")) {
        let fin = d("finance_costs");
        let (ok, detail) = if tic < tdc {
            (false, format!("TIC {tic:.0} is below TDC {tdc:.0}"))
        } else if let Some(fin) = fin {
            let gap = tic - tdc;
            let ok = (gap - fin).abs() <= fin * 0.02;
            let detail = if !ok && gap > fin {
                // IH: Soller "TOTAL COSTS" 80.99m = TDC 65.80m + disposal
                // fees 2.07m + finance 13.12m — the residual is an ungeared
                // cost line dropped from TDC, not bad finance data.
                format!(
                    "TIC − TDC {gap:.0} exceeds stated finance costs {fin:.0} by {residual:.0} — TDC likely omits ungeared cost lines (disposal/selling fees, other costs)",
                    residual = gap - fin,
                )
            } else {
                format!("TIC − TDC {gap:.0} vs stated finance costs {fin:.0}")
            };
            (ok, detail)
        } else {
            (true, format!("TDC {tdc:.0} ≤ TIC {tic:.0}"))
        };
        out.push(Tie {
            id: "tdc_le_tic",
            status: if ok { "pass" } else { "fail" },
            detail,
            fields: vec![
                "details.total_development_cost".into(),
                "details.total_investment_cost".into(),
                "details.finance_costs".into(),
            ],
        });
    }

    // Stated profit ≈ GDV − TDC.
    if let (Some(profit), Some(gdv), Some(tdc)) = (
        d("profit"),
        d("gross_development_value"),
        d("total_development_cost"),
    ) {
        let implied = gdv - tdc;
        let ok = (profit - implied).abs() <= implied.abs().max(1.0) * 0.02;
        out.push(Tie {
            id: "profit_vs_gdv_cost",
            status: if ok { "pass" } else { "fail" },
            detail: format!(
                "stated profit {profit:.0} vs GDV {gdv:.0} − TDC {tdc:.0} = {implied:.0}"
            ),
            fields: vec![
                "details.profit".into(),
                "details.gross_development_value".into(),
                "details.total_development_cost".into(),
            ],
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn doc(fields: Value) -> Value {
        json!({ "details": fields })
    }
    #[allow(non_snake_case)]
    fn ws(v: Value) -> Value {
        json!({ "value": v, "source": "x.pdf", "source_type": "extracted" })
    }

    #[test]
    fn noi_tie_pass_fail_skip() {
        // pass: 800 = 1000 − 200
        let d = doc(json!({
            "net_operating_income": ws(json!({"amount": 800000})),
            "estimated_rental_value": ws(json!({"amount": 1000000})),
            "operating_costs": ws(json!({"amount": 200000})),
        }));
        assert_eq!(tie_checks(&d)[0].status, "pass");
        // fail: NOI equals gross rent while opex is stated
        let d2 = doc(json!({
            "net_operating_income": ws(json!({"amount": 1000000})),
            "estimated_rental_value": ws(json!({"amount": 1000000})),
            "operating_costs": ws(json!({"amount": 200000})),
        }));
        let t = &tie_checks(&d2)[0];
        assert_eq!(t.status, "fail");
        assert!(t.detail.contains("not deducted"));
        // pct opex form
        let d3 = doc(json!({
            "net_operating_income": ws(json!({"amount": 755000})),
            "estimated_rental_value": ws(json!({"amount": 1000000})),
            "operating_costs": ws(json!({"percentage": 24.5})),
        }));
        assert_eq!(tie_checks(&d3)[0].status, "pass");
        // skip: no opex evidence → check absent
        let d4 = doc(json!({
            "net_operating_income": ws(json!({"amount": 800000})),
            "estimated_rental_value": ws(json!({"amount": 1000000})),
        }));
        assert!(!tie_checks(&d4).iter().any(|t| t.id == "noi_vs_rent_opex"));
    }

    #[test]
    fn cap_tie() {
        // NOI 2.9m / (GDV 60m / 0.94) = 4.55% — stated 4.5% passes.
        let d = doc(json!({
            "net_operating_income": ws(json!({"amount": 2905000})),
            "gross_development_value": ws(json!({"amount": 60000000})),
            "stabilised_yield": ws(json!(0.045)),
            "purchaser_costs": ws(json!({"percentage": 6})),
        }));
        let t = tie_checks(&d)
            .into_iter()
            .find(|t| t.id == "cap_vs_noi_value")
            .unwrap();
        assert_eq!(t.status, "pass");
        let d2 = doc(json!({
            "net_operating_income": ws(json!({"amount": 2905000})),
            "gross_development_value": ws(json!({"amount": 60000000})),
            "stabilised_yield": ws(json!(5.5)),
            "purchaser_costs": ws(json!({"percentage": 6})),
        }));
        assert_eq!(
            tie_checks(&d2)
                .into_iter()
                .find(|t| t.id == "cap_vs_noi_value")
                .unwrap()
                .status,
            "fail"
        );
    }

    #[test]
    fn tdc_tic_tie() {
        let d = doc(json!({
            "total_development_cost": ws(json!({"amount": 5000000})),
            "total_investment_cost": ws(json!({"amount": 5500000})),
            "finance_costs": ws(json!({"amount": 500000})),
        }));
        assert_eq!(
            tie_checks(&d)
                .into_iter()
                .find(|t| t.id == "tdc_le_tic")
                .unwrap()
                .status,
            "pass"
        );
        let d2 = doc(json!({
            "total_development_cost": ws(json!({"amount": 6000000})),
            "total_investment_cost": ws(json!({"amount": 5500000})),
        }));
        assert_eq!(
            tie_checks(&d2)
                .into_iter()
                .find(|t| t.id == "tdc_le_tic")
                .unwrap()
                .status,
            "fail"
        );
    }

    #[test]
    fn tdc_gap_residual_names_ungeared_lines() {
        // IH shape: gap exceeds stated finance by an omitted disposal line.
        let d = doc(json!({
            "total_development_cost": ws(json!({"amount": 65795730})),
            "total_investment_cost": ws(json!({"amount": 80991075})),
            "finance_costs": ws(json!({"amount": 13121745})),
        }));
        let t = tie_checks(&d)
            .into_iter()
            .find(|t| t.id == "tdc_le_tic")
            .unwrap();
        assert_eq!(t.status, "fail");
        assert!(t.detail.contains("ungeared cost lines"));
        assert!(t.detail.contains("2073600"));
        // Gap fully explained by finance → ordinary fail text, no residual hint.
        let d2 = doc(json!({
            "total_development_cost": ws(json!({"amount": 6000000})),
            "total_investment_cost": ws(json!({"amount": 5000000})),
            "finance_costs": ws(json!({"amount": 100000})),
        }));
        let t2 = tie_checks(&d2)
            .into_iter()
            .find(|t| t.id == "tdc_le_tic")
            .unwrap();
        assert_eq!(t2.status, "fail");
        assert!(t2.detail.contains("is below TDC"));
    }

    #[test]
    fn profit_tie() {
        let d = doc(json!({
            "profit": ws(json!({"amount": 2700000})),
            "gross_development_value": ws(json!({"amount": 8200000})),
            "total_development_cost": ws(json!({"amount": 5500000})),
        }));
        assert_eq!(
            tie_checks(&d)
                .into_iter()
                .find(|t| t.id == "profit_vs_gdv_cost")
                .unwrap()
                .status,
            "pass"
        );
        // skip when no stated profit
        let d2 = doc(json!({
            "gross_development_value": ws(json!({"amount": 8200000})),
            "total_development_cost": ws(json!({"amount": 5500000})),
        }));
        assert!(!tie_checks(&d2).iter().any(|t| t.id == "profit_vs_gdv_cost"));
    }
}

#[cfg(test)]
mod extra_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn cap_tie_skipped_on_sales_gdv() {
        let d = json!({"details": {
            "net_operating_income": {"value": {"amount": 3340000}},
            "gross_development_value": {"value": {"amount": 115200000}},
            "stabilised_yield": {"value": 0.052},
            "residential_units": {"value": 108},
            "purchaser_costs": {"value": 0.0},
        }, "canonical_deal": {"valuations": []}});
        let ties = tie_checks(&d);
        assert!(!ties.iter().any(|t| t.id == "cap_vs_noi_value"), "{ties:?}");
    }

    #[test]
    fn cap_tie_fires_on_investment_gdv() {
        let d = json!({"details": {
            "net_operating_income": {"value": {"amount": 2888391}},
            "gross_development_value": {"value": {"amount": 59903986}},
            "stabilised_yield": {"value": 0.045},
            "residential_units": {"value": 183},
            "purchaser_costs": {"value": 0.0},
        }, "canonical_deal": {"valuations": []}});
        let ties = tie_checks(&d);
        assert!(ties.iter().any(|t| t.id == "cap_vs_noi_value"), "{ties:?}");
    }
}
