//! Multi-pass extraction pipeline — splits the single monolithic LLM call
//! into 5 focused passes, each with a smaller prompt and a specific mandate.
//! Deterministic post-processing computes derived fields (loan_amount,
//! gross_debt, LTV, etc.) from the facility graph, eliminating a class of
//! LLM errors entirely.

use serde_json::{json, Value};

use crate::conform;
use crate::corpus::Corpus;
use crate::coverage;
use crate::extract::Extractor;
use crate::prompts;
use crate::repair;
use crate::validate;

#[derive(Debug)]
pub struct PipelineResult {
    pub document: Value,
    pub diagnostics: PipelineDiagnostics,
}

#[derive(Debug)]
pub struct PipelineDiagnostics {
    pub pass_times_ms: Vec<(&'static str, u64)>,
    pub repair_rounds: usize,
    pub issues_initial: usize,
    pub issues_remaining: Vec<validate::Issue>,
}

pub async fn run_multi_pass(
    extractor: &Extractor,
    corpus: &Corpus,
    max_repair_rounds: usize,
    coverage_ex: Option<&Extractor>,
    structure_ex: Option<&Extractor>,
    details_ex: Option<&Extractor>,
) -> Result<PipelineResult, String> {
    let pipeline_start = std::time::Instant::now();
    let mut pass_times: Vec<(&'static str, u64)> = Vec::new();

    // ── Pass A: Structure (classification + facilities + focus) ──────────
    let t = std::time::Instant::now();
    eprintln!("[pipeline] pass A — structure");
    // Pass A (structure) runs on Sol when configured — it lays down the
    // facility graph and deal skeleton that every later pass builds on.
    let structure_model = structure_ex.unwrap_or(extractor);
    eprintln!("[pipeline] pass A model: {}", structure_model.model);
    let pass_a_text = structure_model.call(prompts::PASS_A_STRUCTURE, &corpus.render()).await?;
    let pass_a: Value = crate::extract::parse_json(&pass_a_text)?;
    let classification = pass_a.get("classification").cloned().unwrap_or(json!(null));
    let doc_annotations = pass_a.get("document_annotations").cloned().unwrap_or(json!([]));
    let facilities = pass_a.get("facilities").cloned().unwrap_or(json!([]));
    let capital_events = pass_a.get("capital_events").cloned().unwrap_or(json!([]));
    let capital_snapshots = pass_a.get("capital_snapshots").cloned().unwrap_or(json!([]));
    let focus_facility_id = pass_a.get("focus_facility_id").cloned().unwrap_or(json!(null));
    let focus_resolution_status = pass_a.get("focus_resolution_status").cloned().unwrap_or(json!("unresolved"));
    let candidate_facility_ids = pass_a.get("candidate_facility_ids").cloned().unwrap_or(json!([]));
    pass_times.push(("structure", t.elapsed().as_millis() as u64));
    eprintln!("[pipeline] pass A done — {} facilities, focus={}, {:.1}s",
        facilities.as_array().map_or(0, |a| a.len()),
        focus_facility_id.as_str().unwrap_or("none"),
        t.elapsed().as_secs_f64());

    // ── Pass B: Details (valuations + phases + details + KPIs) ───────────
    let t = std::time::Instant::now();
    eprintln!("[pipeline] pass B — details");
    let focus_summary = format_focus_summary(&focus_facility_id, &facilities);
    let classification_summary = format_classification_summary(&classification);
    let pass_b_prompt = prompts::PASS_B_DETAILS
        .replace("{focus_summary}", &focus_summary)
        .replace("{classification_summary}", &classification_summary)
        .replace("{deal_type_guidance}", &prompts::deal_type_guidance(&classification));
    let details_extractor = details_ex.unwrap_or(extractor);
    eprintln!("[pipeline] pass B model: {}", details_extractor.model);
    let pass_b_text = details_extractor.call(&pass_b_prompt, &corpus.render()).await?;
    let pass_b: Value = crate::extract::parse_json(&pass_b_text)?;
    let valuations = pass_b.get("valuations").cloned().unwrap_or(json!([]));
    let phases = pass_b.get("phases").cloned().unwrap_or(json!([]));
    let mut details = pass_b.get("details").cloned().unwrap_or(json!({}));
    let mut kpis = pass_b.get("kpis").cloned().unwrap_or(json!({}));
    pass_times.push(("details", t.elapsed().as_millis() as u64));
    eprintln!("[pipeline] pass B done — {} valuations, {} phases, {:.1}s",
        valuations.as_array().map_or(0, |a| a.len()),
        phases.as_array().map_or(0, |a| a.len()),
        t.elapsed().as_secs_f64());

    // ── Deterministic post-processing ───────────────────────────────────
    apply_deterministic_fields(
        &mut details,
        &mut kpis,
        &facilities,
        &focus_facility_id,
        &valuations,
        &classification,
    );

    // ── Assemble the full document ──────────────────────────────────────
    let document = json!({
        "canonical_deal": {
            "schema_version": "1",
            "facilities": facilities,
            "tranches": [],
            "facility_parties": [],
            "capital_events": capital_events,
            "capital_snapshots": capital_snapshots,
            "focus_facility_id": focus_facility_id,
            "focus_resolution_status": focus_resolution_status,
            "candidate_facility_ids": candidate_facility_ids,
            "assertions": [],
            "conflicts": [],
            "validation_issues": [],
            "deal_scope": null,
            "documents": doc_annotations,
            "assets": [],
            "phases": phases,
            "planning_applications": [],
            "valuations": valuations,
            "construction_budgets": [],
        },
        "details": details,
        "kpis": kpis,
        "classification": classification,
    });

    // ── Conform → validate & repair ─────────────────────────────────────
    let mut document = document;
    let fixes = conform::conform_document(&mut document);
    if fixes > 0 {
        eprintln!("[pipeline] mechanical conform applied {fixes} fixes");
    }
    let initial_issues = validate::validate(&document);
    let initial_errors = validate::error_count(&initial_issues);
    eprintln!("[pipeline] assembled — {initial_errors} validation errors");

    let (mut final_doc, _pre_coverage_issues, repair_rounds) =
        repair::repair_loop(extractor, corpus, document, max_repair_rounds).await;

    // ── Thin-doc escalation ──
    // When pass B produced a sparse document (many evidenced-but-empty fields),
    // re-run the details pass on Sol and merge — but ONLY into fields still
    // empty, never overwriting populated values.
    if let Some(cov_ex) = coverage_ex {
        let gaps = coverage::coverage_issues(&final_doc, corpus);
        let details_model = details_ex.map(|d| d.model.as_str()).unwrap_or(extractor.model.as_str());
        if gaps.len() >= 4 && cov_ex.model != details_model {
            eprintln!("[pipeline] thin document — {} evidenced gaps, escalating pass B to {}",
                gaps.len(), cov_ex.model);
            let pass_b_prompt = prompts::PASS_B_DETAILS
                .replace("{focus_summary}", &focus_summary)
                .replace("{classification_summary}", &classification_summary)
                .replace("{deal_type_guidance}", &prompts::deal_type_guidance(&classification));
            match cov_ex.call(&pass_b_prompt, &corpus.render()).await {
                Ok(text) => match crate::extract::parse_json(&text) {
                    Ok(esc) => {
                        let mut filled = 0usize;
                        for section in ["details", "kpis"] {
                            let (Some(base), Some(extra)) = (final_doc.get_mut(section), esc.get(section)) else { continue };
                            let (Some(base_o), Some(extra_o)) = (base.as_object_mut(), extra.as_object()) else { continue };
                            for (k, v) in extra_o {
                                let empty = base_o.get(k).map_or(true, |e| {
                                    e.is_null()
                                        || e.as_object().map(|o| o.is_empty() || o.get("value").map_or(false, |v| v.is_null())).unwrap_or(false)
                                        || e.as_array().map(|a| a.is_empty()).unwrap_or(false)
                                });
                                if empty {
                                    base_o.insert(k.clone(), v.clone());
                                    filled += 1;
                                }
                            }
                        }
                        // Canonical arrays the escalation produced that Luna lacked.
                        if let (Some(cd), Some(ecd)) = (final_doc.get_mut("canonical_deal").and_then(|v| v.as_object_mut()), esc.get("canonical_deal").and_then(|v| v.as_object())) {
                            for k in ["valuations", "phases", "planning_applications", "facility_parties", "construction_budgets"] {
                                let empty = cd.get(k).and_then(|v| v.as_array()).map_or(true, |a| a.is_empty());
                                if empty {
                                    if let Some(arr) = ecd.get(k).and_then(|v| v.as_array()).filter(|a| !a.is_empty()) {
                                        cd.insert(k.to_string(), Value::Array(arr.clone()));
                                        filled += 1;
                                    }
                                }
                            }
                        }
                        conform::conform_document(&mut final_doc);
                        eprintln!("[pipeline] escalation merged {filled} fields");
                    }
                    Err(e) => eprintln!("[pipeline] escalation parse failed: {e}"),
                },
                Err(e) => eprintln!("[pipeline] escalation call failed: {e}"),
            }
        }
    }

    // ── Coverage fill (Sol) ──
    // One small patch call per extraction, not folded into the repair loop:
    // the evidence travels inline so no tool loop is needed. Sol is better
    // at scoped entity fills and the patch output keeps latency low.
    if let Some(cov_ex) = coverage_ex {
        let missing = coverage::coverage_issues(&final_doc, corpus);
        if !missing.is_empty() {
            eprintln!("[coverage] {} missing fields with evidence — Sol patch call ({}): {}",
                missing.len(), cov_ex.model,
                missing.iter().map(|i| i.path.clone()).collect::<Vec<_>>().join(", "));
            let t = std::time::Instant::now();
            match coverage::fill_missing(cov_ex, &mut final_doc, &missing).await {
                Ok(paths) => eprintln!("[coverage] filled {} paths in {:.1}s", paths.len(), t.elapsed().as_secs_f64()),
                Err(e) => eprintln!("[coverage] fill FAILED: {e}"),
            }
        }
    }

    // ── Verify pass: cheap ground-check of extracted numbers against the
    // corpus, one cheap LLM call per flagged mismatch. Catches right-field /
    // wrong-cell errors (e.g. cap rate read as 0.04 vs a stated 0.045) that
    // conform can't see because they're shape-correct.
    let verify_model = details_ex.or(coverage_ex).unwrap_or(extractor);
    let (checked, fixes_applied) = crate::verify::verify_extracted_numbers(&mut final_doc, corpus, verify_model).await;
    if fixes_applied > 0 {
        eprintln!("[pipeline] verify — {} fields checked, {} corrections applied", checked, fixes_applied);
    }

    let remaining_issues = validate::validate(&final_doc);
    eprintln!("[pipeline] complete — {} repair rounds, {} issues, {:.1}s total",
        repair_rounds, remaining_issues.len(), pipeline_start.elapsed().as_secs_f64());

    Ok(PipelineResult {
        document: final_doc,
        diagnostics: PipelineDiagnostics {
            pass_times_ms: pass_times,
            repair_rounds,
            issues_initial: initial_errors,
            issues_remaining: remaining_issues,
        },
    })
}

fn format_classification_summary(classification: &Value) -> String {
    let cp = classification.get("capital_position").and_then(|v| v.as_str()).unwrap_or("unknown");
    let lt = classification.get("loan_type")
        .and_then(|v| v.get("category").and_then(|c| c.as_str()))
        .unwrap_or("unknown");
    let lr = classification.get("loan_rank")
        .and_then(|v| v.get("category").and_then(|c| c.as_str()))
        .unwrap_or("unknown");
    let pt = classification.get("project_type")
        .and_then(|v| v.get("category").and_then(|c| c.as_str()))
        .unwrap_or("unknown");
    format!("{cp} / {lt} / {lr} / {pt}")
}

fn format_focus_summary(focus_id: &Value, facilities: &Value) -> String {
    let fid = focus_id.as_str().unwrap_or("none");
    if fid == "none" { return "No focus facility resolved".into(); }
    let facility = facilities.as_array()
        .and_then(|a| a.iter().find(|f| f.get("id").and_then(|v| v.as_str()) == Some(fid)));
    match facility {
        Some(f) => {
            let name = f.get("name").and_then(|v| v.as_str()).unwrap_or(fid);
            let amt = f.get("commitment").and_then(|c| c.get("amount")).and_then(|v| v.as_str()).unwrap_or("?");
            let ccy = f.get("commitment").and_then(|c| c.get("currency")).and_then(|v| v.as_str()).unwrap_or("GBP");
            let role = f.get("role").and_then(|v| v.as_str()).unwrap_or("?");
            format!("{name} ({role}) — {ccy} {amt}")
        }
        None => format!("focus_facility_id={fid} (not found in facilities)"),
    }
}


fn amount_of(money: &Value) -> Option<f64> {
    money.get("amount")
        .and_then(|v| v.as_str().and_then(|s| s.parse::<f64>().ok()).or_else(|| v.as_f64()))
}

fn sourced_money(amount: f64, currency: &str, source: &str) -> Value {
    json!({
        "source": source,
        "source_type": "calculated",
        "notes": "derived from facility graph",
        "scope": "deal",
        "evidence_text": null,
        "value": { "amount": format!("{:.0}", amount), "currency": currency }
    })
}

fn sourced_percent(value: f64) -> Value {
    json!({
        "source": "",
        "source_type": "calculated",
        "notes": "computed from loan_amount and valuation",
        "scope": "deal",
        "evidence_text": null,
        "value": value
    })
}

fn apply_deterministic_fields(
    details: &mut Value,
    kpis: &mut Value,
    facilities: &Value,
    focus_id: &Value,
    valuations: &Value,
    _classification: &Value,
) {
    let facs = facilities.as_array().cloned().unwrap_or_default();
    let focus_fid = focus_id.as_str().unwrap_or("");

    let mut focus = facs.iter().find(|f| f.get("id").and_then(|v| v.as_str()) == Some(focus_fid));

    if let Some(fc) = focus {
        let focus_amt = fc.get("commitment").and_then(amount_of).unwrap_or(0.0);
        let focus_role = fc.get("role").and_then(|v| v.as_str()).unwrap_or("");
        let same_role_facilities: Vec<&Value> = facs.iter()
            .filter(|f| f.get("role").and_then(|v| v.as_str()) == Some(focus_role)
                || f.get("role").and_then(|v| v.as_str()) == Some("junior_debt") && focus_role == "mezzanine_debt"
                || f.get("role").and_then(|v| v.as_str()) == Some("mezzanine_debt") && focus_role == "junior_debt")
            .collect();
        for c in &same_role_facilities {
            let cid = c.get("id").and_then(|v| v.as_str()).unwrap_or("");
            if cid == focus_fid { continue; }
            let c_amt = c.get("commitment").and_then(amount_of).unwrap_or(0.0);
            if c_amt > focus_amt * 1.1 && c_amt < focus_amt * 5.0 {
                let other_facs: Vec<f64> = same_role_facilities.iter()
                    .filter(|x| x.get("id").and_then(|v| v.as_str()) != Some(cid))
                    .map(|x| x.get("commitment").and_then(amount_of).unwrap_or(0.0))
                    .filter(|a| *a > 0.0)
                    .collect();
                let other_sum: f64 = other_facs.iter().sum();
                if other_facs.len() >= 2 && other_sum > 0.0 && (c_amt - other_sum).abs() / c_amt < 0.05 {
                    focus = Some(c);
                    break;
                }
            }
        }
    }

    // If the focus facility is part of a facility group, sum all grouped
    // facilities' commitments — e.g. Edenbridge's two Shojin mezzanine
    // tranches share facility_group_id and should report as one loan_amount.
    let focus_commitment = focus
        .and_then(|f| f.get("facility_group_id"))
        .and_then(|gid| gid.as_str())
        .and_then(|gid| {
            let sum: f64 = facs.iter()
                .filter(|f| f.get("facility_group_id").and_then(|v| v.as_str()) == Some(gid))
                .filter_map(|f| f.get("commitment").and_then(amount_of))
                .sum();
            if sum > 0.0 { Some(sum) } else { None }
        })
        .or_else(|| focus.and_then(|f| f.get("commitment")).and_then(amount_of));
    let focus_currency = focus
        .and_then(|f| f.get("commitment").and_then(|c| c.get("currency")).and_then(|v| v.as_str()))
        .unwrap_or("gbp");

    if let Some(amt) = focus_commitment {
        if amt >= 100_000.0 {
            details["loan_amount"] = sourced_money(amt, focus_currency, "");
        }
    }

    // Whether the focus facility is project-scope. When it is, the whole
    // project is the deal, so project-scope facilities are included.
    let focus_is_project = focus
        .and_then(|f| f.get("scope_level"))
        .and_then(|v| v.as_str())
        == Some("project");
    let has_deal_scope_facility = facs.iter().any(|f| {
        f.get("scope_level").and_then(|v| v.as_str()) == Some("deal")
    });

    let mut senior_total = 0.0;
    let mut mezz_total = 0.0;
    let mut equity_total = 0.0;
    let mut gross_debt_total = 0.0;
    let mut debt_currency = "gbp".to_string();

    for f in &facs {
        let instrument = f.get("instrument_type").and_then(|v| v.as_str()).unwrap_or("");
        let role = f.get("role").and_then(|v| v.as_str()).unwrap_or("");
        let disposition = f.get("disposition").and_then(|v| v.as_str()).unwrap_or("");
        let scope = f.get("scope_level").and_then(|v| v.as_str());

        if disposition == "proposed" { continue; }
        if disposition == "repaid" { continue; }
        // Exclude project-scope facilities only when the deal itself is a
        // tranche within a project (i.e. there are deal-scope facilities and
        // the focus is not project-scope). When the deal is the project —
        // like Edenbridge — project-scope facilities are the deal.
        if scope == Some("project") && has_deal_scope_facility && !focus_is_project {
            continue;
        }

        let commitment = f.get("commitment").and_then(amount_of).unwrap_or(0.0);
        if commitment < 100_000.0 { continue; }

        if let Some(c) = f.get("commitment").and_then(|c| c.get("currency")).and_then(|v| v.as_str()) {
            debt_currency = c.to_string();
        }

        if instrument == "debt_facility" {
            // Only recognised debt roles count toward gross/senior/mezzanine.
            // Instruments like "other" (e.g. the Oakley co-investment bond) are
            // not debt of the underlying facility and must be excluded.
            match role {
                "senior_debt" | "whole_loan" => {
                    gross_debt_total += commitment;
                    senior_total += commitment;
                }
                "mezzanine_debt" | "junior_debt" => {
                    gross_debt_total += commitment;
                    mezz_total += commitment;
                }
                _ => {}
            }
        } else if instrument == "ordinary_equity" || instrument == "preferred_equity" {
            equity_total += commitment;
        }
    }

    if gross_debt_total > 0.0 {
        details["gross_debt"] = sourced_money(gross_debt_total, &debt_currency, "");
    }
    if senior_total > 0.0 {
        details["senior_debt"] = sourced_money(senior_total, &debt_currency, "");
    }
    if mezz_total > 0.0 {
        details["mezzanine_debt"] = sourced_money(mezz_total, &debt_currency, "");
    }
    if equity_total > 0.0 {
        details["developer_equity"] = sourced_money(equity_total, &debt_currency, "");
    }

    let loan_amount = focus_commitment.filter(|a| *a >= 100_000.0);

    let market_value = valuations.as_array()
        .and_then(|vs| vs.iter()
            .find(|v| v.get("value_type").and_then(|t| t.as_str()) == Some("market_value"))
            .and_then(|v| v.get("amount").and_then(amount_of)))
        .or_else(|| detail_money(details, "market_value"))
        .or_else(|| detail_money(details, "day_1_value"))
        .or_else(|| detail_money(details, "stabilised_value"))
        .or_else(|| detail_money(details, "purchase_price"));

    let gdv = valuations.as_array()
        .and_then(|vs| vs.iter()
            .filter(|v| v.get("value_type").and_then(|t| t.as_str()) == Some("gross_development_value"))
            .filter(|v| v.get("phase_id").is_none() || v.get("phase_id") == Some(&json!(null)))
            .map(|v| v.get("amount").and_then(amount_of).unwrap_or(0.0))
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)));

    if let (Some(la), Some(mv)) = (loan_amount, market_value) {
        if mv > 0.0 {
            let ltv = la / mv * 100.0;
            kpis["loan_to_value"] = sourced_percent((ltv * 10.0).round() / 10.0);
        }
    }
    if let (Some(la), Some(g)) = (loan_amount, gdv) {
        if g > 0.0 {
            let ltgdv = la / g * 100.0;
            kpis["loan_to_gdv"] = sourced_percent((ltgdv * 10.0).round() / 10.0);
        }
    }

    let tdc = detail_money(details, "total_development_cost");
    let tic = detail_money(details, "total_investment_cost");
    let construction_costs = detail_money(details, "construction_costs");
    let cost_basis = tic.or(tdc);
    let detail_gdv = detail_money(details, "gross_development_value").or(gdv);

    if let (Some(la), Some(cb)) = (loan_amount, cost_basis) {
        if cb > 0.0 {
            let ltc = la / cb * 100.0;
            kpis["loan_to_cost"] = sourced_percent((ltc * 100.0).round() / 100.0);
        }
    }

    if let (Some(g), Some(cb)) = (detail_gdv, tdc.or(tic)) {
        if cb > 0.0 {
            let poc = (g - cb) / cb * 100.0;
            kpis["profit_on_cost"] = sourced_percent((poc * 100.0).round() / 100.0);
            let profit = g - cb;
            details["profit"] = sourced_money(profit, focus_currency, "");
        }
    }

    if equity_total > 0.0 {
        if let Some(cb) = tdc.or(tic) {
            if cb > 0.0 {
                let eq_tdc = equity_total / cb * 100.0;
                kpis["borrower_equity_on_tdc"] = sourced_percent((eq_tdc * 100.0).round() / 100.0);
            }
        }

        let value_basis = market_value.or(detail_money(details, "day_1_value")).or(gdv);
        if let Some(vb) = value_basis {
            if vb > 0.0 {
                let eq_value = equity_total / vb * 100.0;
                kpis["borrower_equity_on_value"] = sourced_percent((eq_value * 100.0).round() / 100.0);
            }
        }
    }

    let focus_has_interest = focus
        .and_then(|f| f.get("interest_terms"))
        .and_then(|v| v.as_array())
        .map_or(false, |a| !a.is_empty());

    if !focus_has_interest {
        if let Some(ir) = details.get("interest_rate") {
            let notes = ir.get("notes").and_then(|v| v.as_str()).unwrap_or("");
            let evidence = ir.get("evidence_text").and_then(|v| v.as_str()).unwrap_or("");
            let combined = format!("{} {}", notes, evidence).to_lowercase();
            let is_valuation_rate = combined.contains("finance rate")
                || combined.contains("residual appraisal")
                || combined.contains("valuation finance")
                || combined.contains("allowed a finance")
                || combined.contains("we have allowed");
            if is_valuation_rate {
                details.as_object_mut().unwrap().remove("interest_rate");
            }
        }
    }

    let gia = details.get("gross_internal_area")
        .and_then(|v| v.get("value"))
        .and_then(|v| v.get("value"))
        .and_then(|v| v.as_f64());
    let build_costs = construction_costs.or(tdc);
    if let (Some(bc), Some(area)) = (build_costs, gia) {
        if area > 0.0 {
            let per_sqft = bc / area;
            if per_sqft >= 50.0 && per_sqft <= 1500.0 {
                kpis["build_cost_per_area_unit"] = json!({
                    "source": "",
                    "source_type": "calculated",
                    "notes": "computed from construction costs / GIA",
                    "scope": "deal",
                    "evidence_text": null,
                    "value": { "amount": (per_sqft * 100.0).round() / 100.0, "currency": focus_currency, "area_unit": "sqft" }
                });
            }
        }
    }
}

fn detail_money(details: &Value, field: &str) -> Option<f64> {
    details.get(field)
        .and_then(|v| v.get("value"))
        .and_then(|v| {
            v.get("amount")
                .and_then(|a| a.as_f64().or_else(|| a.as_str().and_then(|s| s.parse::<f64>().ok())))
        })
        .filter(|a| *a >= 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_loan_amount_from_focus() {
        let facilities = json!([
            { "id": "bridge", "commitment": { "amount": "25750000", "currency": "gbp" },
              "instrument_type": "debt_facility", "role": "senior_debt",
              "disposition": "requested" },
            { "id": "mezz", "commitment": { "amount": "5000000", "currency": "gbp" },
              "instrument_type": "debt_facility", "role": "mezzanine_debt",
              "disposition": "requested" },
        ]);
        let focus_id = json!("bridge");
        let valuations = json!([
            { "id": "v1", "value_type": "market_value", "amount": { "amount": "32500000", "currency": "gbp" } },
            { "id": "v2", "value_type": "gross_development_value", "amount": { "amount": "50000000", "currency": "gbp" } },
        ]);
        let classification = json!({ "capital_position": "debt" });
        let mut details = json!({});
        let mut kpis = json!({});

        apply_deterministic_fields(&mut details, &mut kpis, &facilities, &focus_id, &valuations, &classification);

        let la = details["loan_amount"]["value"]["amount"].as_str().unwrap();
        assert_eq!(la, "25750000");

        let gd = details["gross_debt"]["value"]["amount"].as_str().unwrap();
        assert_eq!(gd, "30750000");

        let sd = details["senior_debt"]["value"]["amount"].as_str().unwrap();
        assert_eq!(sd, "25750000");

        let md = details["mezzanine_debt"]["value"]["amount"].as_str().unwrap();
        assert_eq!(md, "5000000");

        let ltv = kpis["loan_to_value"]["value"].as_f64().unwrap();
        assert!((ltv - 79.2).abs() < 0.5, "LTV was {ltv}");

        let ltgdv = kpis["loan_to_gdv"]["value"].as_f64().unwrap();
        assert!((ltgdv - 51.5).abs() < 0.5, "LTGDV was {ltgdv}");
    }

    #[test]
    fn repaid_facilities_excluded_from_gross_debt() {
        let facilities = json!([
            { "id": "bridge", "commitment": { "amount": "25750000", "currency": "gbp" },
              "instrument_type": "debt_facility", "role": "senior_debt",
              "disposition": "requested" },
            { "id": "old_loan", "commitment": { "amount": "10000000", "currency": "gbp" },
              "instrument_type": "debt_facility", "role": "senior_debt",
              "disposition": "repaid", "status": "repaid" },
        ]);
        let focus_id = json!("bridge");
        let mut details = json!({});
        let mut kpis = json!({});

        apply_deterministic_fields(&mut details, &mut kpis, &facilities, &focus_id, &json!([]), &json!({}));

        let gd = details["gross_debt"]["value"]["amount"].as_str().unwrap();
        assert_eq!(gd, "25750000");
    }
}
