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
use crate::recall::corpus_has_amount;
use crate::repair;
use crate::tie;
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
    pub recall: RecallDiagnostics,
}

/// What the recall machinery saw and did — returned to the caller so missed
/// fields are visible in traces instead of only on the service's stderr.
#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct RecallDiagnostics {
    /// Pass retries: "<pass>: <first error>" (truncation / bad JSON).
    pub pass_retries: Vec<String>,
    /// Pass-A self-consistency: how many structure votes were cast.
    pub pass_a_votes: usize,
    /// Points the two pass-A votes disagreed on (empty = votes agreed).
    pub pass_a_disagreements: Vec<String>,
    /// Whether a disagreeing pass A was adjudicated by a third call.
    pub pass_a_adjudicated: bool,
    /// Facilities the adjudicator dropped that were restored because their
    /// commitment is stated in the corpus: "<name> <amount>".
    pub adjudication_restored: Vec<String>,
    /// Vote-2 or adjudication failure (truncated); pass A still succeeded.
    pub pass_a_error: Option<String>,
    /// Outcome of the "debt deal but zero facilities" pass-A retry, e.g.
    /// "added 2 facilities" / "no evidenced facilities".
    pub pass_a_empty_retry: Option<String>,
    /// Evidenced-but-empty field paths after the repair loop.
    pub coverage_gaps_initial: Vec<String>,
    pub escalated: bool,
    pub escalation_filled: usize,
    pub escalation_error: Option<String>,
    pub coverage_filled: Vec<String>,
    pub coverage_rejected: Vec<String>,
    /// First new validation issue per rejected fill path:
    /// "path: code message" — fills used to die silently on nullable
    /// required keys.
    pub coverage_rejected_reasons: Vec<String>,
    pub coverage_error: Option<String>,
    /// Corpus figures (labelled money / percentages) found nowhere in the
    /// output, before the tool fill: "<doc> p<page>: <label> <figure>".
    pub unaccounted_figures: Vec<String>,
    pub tool_fill_merged: Vec<String>,
    pub tool_fill_rejected: Vec<String>,
    pub tool_fill_rejected_reasons: Vec<String>,
    pub tool_fill_error: Option<String>,
    /// Evidenced-but-empty paths still empty at the end.
    pub coverage_gaps_remaining: Vec<String>,
    /// Corpus figures still unaccounted for at the end.
    pub unaccounted_remaining: Vec<String>,
    pub verify_checked: usize,
    pub verify_fixes: usize,
    /// Fields flagged by verify (missing from corpus or contradicting their
    /// own evidence text) — the inputs to the LLM fix call.
    #[serde(default)]
    pub verify_flagged: Vec<String>,
    /// Tie checks that failed before the fix call: "id: detail".
    pub ties_initial: Vec<String>,
    /// details.* paths merged by the tie-fix call.
    pub tie_fixes: Vec<String>,
    /// Tie checks still failing after fixes were applied.
    pub ties_remaining: Vec<String>,
    /// Tie-fix call failure (truncated); checks still ran.
    pub tie_error: Option<String>,
}

/// Model routing shared by the service, the CLI and the scoring tests, so
/// every caller exercises the production pipeline. Each model resolves from
/// the explicit override, then `DEAL_EXTRACT_{STRUCTURE,DETAILS,COVERAGE}_MODEL`,
/// then the default (Sol for structure + coverage, `extractor.model` for details).
pub async fn run_routed(
    extractor: &Extractor,
    corpus: &Corpus,
    max_repair_rounds: usize,
    structure_model: Option<String>,
    details_model: Option<String>,
    coverage_model: Option<String>,
) -> Result<PipelineResult, String> {
    let resolve = |explicit: Option<String>, env: &str, default: &str| {
        explicit
            .or_else(|| std::env::var(env).ok())
            .unwrap_or_else(|| default.to_string())
    };
    let distinct = |m: String| -> Option<Extractor> {
        (m != extractor.model)
            .then(|| Extractor::new(&m).ok())
            .flatten()
    };
    let cov_model = resolve(coverage_model, "DEAL_EXTRACT_COVERAGE_MODEL", "gpt-5.6-sol");
    let coverage_owned = distinct(cov_model.clone());
    // Coverage on the same model reuses the main extractor; structure and
    // details on the same model fall back to it inside run_multi_pass.
    let coverage_ex = coverage_owned
        .as_ref()
        .or_else(|| (cov_model == extractor.model).then_some(extractor));
    // The structure extractor always gets its own instance: pass A runs at a
    // raised reasoning effort (DEAL_EXTRACT_STRUCTURE_EFFORT, default
    // "medium") because it lays down the facility graph every later pass
    // builds on — even when the structure model name matches `model`.
    let structure_effort =
        std::env::var("DEAL_EXTRACT_STRUCTURE_EFFORT").unwrap_or_else(|_| "medium".to_string());
    let structure_owned = Extractor::new(&resolve(
        structure_model,
        "DEAL_EXTRACT_STRUCTURE_MODEL",
        "gpt-5.6-sol",
    ))
    .ok()
    .map(|e| e.with_effort(&structure_effort));
    let details_owned = distinct(resolve(
        details_model,
        "DEAL_EXTRACT_DETAILS_MODEL",
        &extractor.model,
    ));
    run_multi_pass(
        extractor,
        corpus,
        max_repair_rounds,
        coverage_ex,
        structure_owned.as_ref(),
        details_owned.as_ref(),
    )
    .await
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
    let mut recall = RecallDiagnostics::default();

    // ── Pass A: Structure (classification + facilities + focus) ──────────
    let t = std::time::Instant::now();
    eprintln!("[pipeline] pass A — structure");
    // Pass A (structure) runs on Sol when configured — it lays down the
    // facility graph and deal skeleton that every later pass builds on.
    let structure_model = structure_ex.unwrap_or(extractor);
    eprintln!("[pipeline] pass A model: {}", structure_model.model);

    // Self-consistency: pass A runs N votes concurrently (default 3; N>3 is
    // clamped to 3 — the crate has no `futures` dep, so the concurrency uses
    // fixed tokio::join! arms) on independent extractors. When the votes
    // disagree on the facility graph or classification, an adjudication call
    // decides the contested points from the evidence — a single sloppy read
    // can no longer set the deal's skeleton. DEAL_EXTRACT_PASS_A_VOTES=1
    // restores single-vote behaviour.
    let pass_a_votes = std::env::var("DEAL_EXTRACT_PASS_A_VOTES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(3)
        .min(3);
    let pass_a = if pass_a_votes < 2 {
        recall.pass_a_votes = 1;
        let (pass_a, retry) = structure_model
            .call_json(prompts::PASS_A_STRUCTURE, &corpus.render())
            .await
            .map_err(|e| format!("pass A (structure): {e}"))?;
        if let Some(r) = retry {
            recall.pass_retries.push(format!("structure: {r}"));
        }
        pass_a
    } else {
        let vote2_ex = details_ex.unwrap_or(extractor);
        eprintln!("[pipeline] pass A vote 2 model: {}", vote2_ex.model);
        let corpus_text = corpus.render();
        let raw: Vec<Result<(Value, Option<String>), String>> = if pass_a_votes >= 3 {
            let (r1, r2, r3) = tokio::join!(
                structure_model.call_json(prompts::PASS_A_STRUCTURE, &corpus_text),
                vote2_ex.call_json(prompts::PASS_A_STRUCTURE, &corpus_text),
                structure_model.call_json(prompts::PASS_A_STRUCTURE, &corpus_text),
            );
            vec![r1, r2, r3]
        } else {
            let (r1, r2) = tokio::join!(
                structure_model.call_json(prompts::PASS_A_STRUCTURE, &corpus_text),
                vote2_ex.call_json(prompts::PASS_A_STRUCTURE, &corpus_text),
            );
            vec![r1, r2]
        };
        recall.pass_a_votes = raw.len();
        let mut votes: Vec<Value> = Vec::new();
        let mut first_err: Option<String> = None;
        for (i, res) in raw.into_iter().enumerate() {
            match res {
                Ok((v, retry)) => {
                    if let Some(r) = retry {
                        recall
                            .pass_retries
                            .push(format!("structure vote {}: {r}", i + 1));
                    }
                    votes.push(v);
                }
                Err(e) => {
                    eprintln!("[pipeline] pass A vote {} FAILED: {e}", i + 1);
                    if first_err.is_none() {
                        first_err = Some(e.clone());
                    }
                    recall.pass_a_error = Some(e.chars().take(300).collect());
                }
            }
        }
        if votes.is_empty() {
            return Err(format!(
                "pass A (structure): {} (all {pass_a_votes} votes failed)",
                first_err.unwrap_or_else(|| "unknown error".to_string())
            ));
        }
        let mut disagreements = Vec::new();
        for (k, other) in votes.iter().enumerate().skip(1) {
            for d in pass_a_disagreements(&votes[0], other) {
                disagreements.push(d.replace("vote 2", &format!("vote {}", k + 1)));
            }
        }
        disagreements.sort();
        disagreements.dedup();
        let mut chosen = if disagreements.is_empty() {
            votes[0].clone()
        } else {
            recall.pass_a_disagreements = disagreements;
            eprintln!(
                "[pipeline] pass A votes disagree — adjudicating: {}",
                recall.pass_a_disagreements.join("; ")
            );
            let mut adj_prompt = format!(
                "{}\n\n## DISAGREEMENTS\n{}",
                prompts::PASS_A_ADJUDICATE,
                recall.pass_a_disagreements.join("\n"),
            );
            for (i, v) in votes.iter().enumerate() {
                adj_prompt.push_str(&format!(
                    "\n\n## EXTRACTION {}\n{}",
                    i + 1,
                    serde_json::to_string(v).unwrap_or_default()
                ));
            }
            match structure_model.call_json(&adj_prompt, &corpus_text).await {
                Ok((adj, retry)) => {
                    if let Some(r) = retry {
                        recall.pass_retries.push(format!("adjudication: {r}"));
                    }
                    recall.pass_a_adjudicated = true;
                    adj
                }
                Err(e) => {
                    eprintln!("[pipeline] pass A adjudication FAILED: {e}");
                    recall.pass_a_error = Some(e.chars().take(300).collect());
                    votes[0].clone()
                }
            }
        };
        // Union guard: a facility present in any vote but absent from the
        // chosen result is restored iff its commitment is corpus-evidenced.
        let vote_refs: Vec<&Value> = votes.iter().collect();
        restore_dropped_facilities(&mut chosen, &vote_refs, corpus, &mut recall);
        dedupe_facilities(&mut chosen);
        merge_facility_amounts(&mut chosen, &votes);
        merge_parties(&mut chosen, &votes);
        // Focus adopt: the adjudicator can clear focus_facility_id even when
        // a vote pointed at a facility that (possibly via the union) is in
        // the chosen set — adopt the first vote focus that resolves.
        let chosen_ids: Vec<&str> = chosen
            .get("facilities")
            .and_then(Value::as_array)
            .map(|fs| fs.iter().filter_map(|f| f.get("id").and_then(Value::as_str)).collect())
            .unwrap_or_default();
        let chosen_focus = chosen
            .get("focus_facility_id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty());
        if chosen_focus.is_none() || !chosen_ids.contains(&chosen_focus.unwrap()) {
            if let Some(fid) = votes.iter().filter_map(|v| {
                v.get("focus_facility_id")
                    .and_then(Value::as_str)
                    .filter(|id| chosen_ids.contains(id))
            }).next() {
                if let Some(obj) = chosen.as_object_mut() {
                    obj.insert("focus_facility_id".to_string(), json!(fid));
                }
            }
        }
        // Empty retry: a debt deal with zero facilities is almost always a
        // missed extraction (Luton: one of three runs produced no facility).
        let no_facilities = chosen
            .get("facilities")
            .and_then(Value::as_array)
            .is_none_or(Vec::is_empty);
        let debt_deal = std::iter::once(&chosen).chain(votes.iter()).any(|v| {
            v.get("classification")
                .and_then(|c| c.get("capital_position"))
                .and_then(Value::as_str)
                == Some("debt")
        });
        if no_facilities && debt_deal {
            eprintln!("[pipeline] pass A returned no facilities on a debt deal — retrying");
            let retry_prompt = format!(
                "{}\n\nPREVIOUS ATTEMPTS RETURNED NO FACILITIES. This is a debt deal: list every existing and requested/proposed facility; each commitment must be an amount quoted in the documents.",
                prompts::PASS_A_STRUCTURE
            );
            match structure_model.call_json(&retry_prompt, &corpus_text).await {
                Ok((retry_doc, retry_note)) => {
                    if let Some(r) = retry_note {
                        recall
                            .pass_retries
                            .push(format!("structure empty retry: {r}"));
                    }
                    let accepted = evidenced_retry_facilities(&retry_doc, corpus);
                    if accepted.is_empty() {
                        recall.pass_a_empty_retry = Some("no evidenced facilities".into());
                    } else {
                        let accepted_ids: Vec<&str> = accepted
                            .iter()
                            .filter_map(|f| f.get("id").and_then(Value::as_str))
                            .collect();
                        let n = accepted.len();
                        if let Some(obj) = chosen.as_object_mut() {
                            obj.insert("facilities".to_string(), json!(accepted));
                            if let Some(fid) =
                                retry_doc.get("focus_facility_id").and_then(Value::as_str)
                            {
                                if accepted_ids.contains(&fid) {
                                    obj.insert("focus_facility_id".to_string(), json!(fid));
                                }
                            }
                        }
                        recall.pass_a_empty_retry = Some(format!("added {n} facilities"));
                    }
                }
                Err(e) => {
                    eprintln!("[pipeline] pass A empty retry FAILED: {e}");
                    recall.pass_a_error = Some(e.chars().take(300).collect());
                    recall.pass_a_empty_retry = Some("retry call failed".into());
                }
            }
        }
        chosen
    };
    let classification = pass_a.get("classification").cloned().unwrap_or(json!(null));
    let doc_annotations = pass_a
        .get("document_annotations")
        .cloned()
        .unwrap_or(json!([]));
    let facilities = pass_a.get("facilities").cloned().unwrap_or(json!([]));
    let capital_events = pass_a.get("capital_events").cloned().unwrap_or(json!([]));
    let capital_snapshots = pass_a
        .get("capital_snapshots")
        .cloned()
        .unwrap_or(json!([]));
    let focus_facility_id = pass_a
        .get("focus_facility_id")
        .cloned()
        .unwrap_or(json!(null));
    let focus_resolution_status = pass_a
        .get("focus_resolution_status")
        .cloned()
        .unwrap_or(json!("unresolved"));
    let candidate_facility_ids = pass_a
        .get("candidate_facility_ids")
        .cloned()
        .unwrap_or(json!([]));
    pass_times.push(("structure", t.elapsed().as_millis() as u64));
    eprintln!(
        "[pipeline] pass A done — {} facilities, focus={}, {:.1}s",
        facilities.as_array().map_or(0, |a| a.len()),
        focus_facility_id.as_str().unwrap_or("none"),
        t.elapsed().as_secs_f64()
    );

    // ── Pass B: Details (valuations + phases + details + KPIs) ───────────
    let t = std::time::Instant::now();
    eprintln!("[pipeline] pass B — details");
    let focus_summary = format_focus_summary(&focus_facility_id, &facilities);
    let classification_summary = format_classification_summary(&classification);
    let guidance = prompts::deal_type_guidance(&classification);
    let details_extractor = details_ex.unwrap_or(extractor);
    eprintln!("[pipeline] pass B model: {}", details_extractor.model);
    let (pass_b, b_notes) = run_pass_b(
        details_extractor,
        corpus,
        &focus_summary,
        &classification_summary,
        &guidance,
    )
    .await
    .map_err(|e| format!("pass B (details): {e}"))?;
    recall.pass_retries.extend(b_notes);
    let valuations = pass_b.get("valuations").cloned().unwrap_or(json!([]));
    let phases = pass_b.get("phases").cloned().unwrap_or(json!([]));
    let mut details = pass_b.get("details").cloned().unwrap_or(json!({}));
    let mut kpis = pass_b.get("kpis").cloned().unwrap_or(json!({}));
    pass_times.push(("details", t.elapsed().as_millis() as u64));
    eprintln!(
        "[pipeline] pass B done — {} valuations, {} phases, {:.1}s",
        valuations.as_array().map_or(0, |a| a.len()),
        phases.as_array().map_or(0, |a| a.len()),
        t.elapsed().as_secs_f64()
    );

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
            "parties": pass_a.get("parties").cloned().unwrap_or(json!([])),
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
    recall.coverage_gaps_initial = coverage::coverage_issues(&final_doc, corpus)
        .into_iter()
        .map(|i| i.path)
        .collect();
    if let Some(cov_ex) = coverage_ex {
        let gaps = &recall.coverage_gaps_initial;
        let details_model = details_ex
            .map(|d| d.model.as_str())
            .unwrap_or(extractor.model.as_str());
        if gaps.len() >= 4 && cov_ex.model != details_model {
            recall.escalated = true;
            eprintln!(
                "[pipeline] thin document — {} evidenced gaps, escalating pass B to {}",
                gaps.len(),
                cov_ex.model
            );
            match run_pass_b(
                cov_ex,
                corpus,
                &focus_summary,
                &classification_summary,
                &guidance,
            )
            .await
            {
                Ok((esc, b_notes)) => {
                    recall
                        .pass_retries
                        .extend(b_notes.into_iter().map(|n| format!("escalation {n}")));
                    {
                        let mut filled = 0usize;
                        for section in ["details", "kpis"] {
                            let (Some(base), Some(extra)) =
                                (final_doc.get_mut(section), esc.get(section))
                            else {
                                continue;
                            };
                            let (Some(base_o), Some(extra_o)) =
                                (base.as_object_mut(), extra.as_object())
                            else {
                                continue;
                            };
                            for (k, v) in extra_o {
                                let empty = base_o.get(k).map_or(true, |e| {
                                    e.is_null()
                                        || e.as_object()
                                            .map(|o| {
                                                o.is_empty()
                                                    || o.get("value").map_or(false, |v| v.is_null())
                                            })
                                            .unwrap_or(false)
                                        || e.as_array().map(|a| a.is_empty()).unwrap_or(false)
                                });
                                if empty {
                                    base_o.insert(k.clone(), v.clone());
                                    filled += 1;
                                }
                            }
                        }
                        // Canonical arrays the escalation produced that Luna lacked.
                        if let (Some(cd), Some(ecd)) = (
                            final_doc
                                .get_mut("canonical_deal")
                                .and_then(|v| v.as_object_mut()),
                            esc.get("canonical_deal").and_then(|v| v.as_object()),
                        ) {
                            for k in [
                                "valuations",
                                "phases",
                                "planning_applications",
                                "facility_parties",
                                "construction_budgets",
                            ] {
                                let empty = cd
                                    .get(k)
                                    .and_then(|v| v.as_array())
                                    .map_or(true, |a| a.is_empty());
                                if empty {
                                    if let Some(arr) = ecd
                                        .get(k)
                                        .and_then(|v| v.as_array())
                                        .filter(|a| !a.is_empty())
                                    {
                                        cd.insert(k.to_string(), Value::Array(arr.clone()));
                                        filled += 1;
                                    }
                                }
                            }
                        }
                        conform::conform_document(&mut final_doc);
                        recall.escalation_filled = filled;
                        eprintln!("[pipeline] escalation merged {filled} fields");
                    }
                }
                Err(e) => {
                    eprintln!("[pipeline] escalation failed: {e}");
                    recall.escalation_error = Some(e.chars().take(300).collect());
                }
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
            eprintln!(
                "[coverage] {} missing fields with evidence — Sol patch call ({}): {}",
                missing.len(),
                cov_ex.model,
                missing
                    .iter()
                    .map(|i| i.path.clone())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            let t = std::time::Instant::now();
            match coverage::fill_missing(cov_ex, &mut final_doc, &missing).await {
                Ok((merged, rejected, reasons)) => {
                    eprintln!(
                        "[coverage] filled {} paths in {:.1}s",
                        merged.len(),
                        t.elapsed().as_secs_f64()
                    );
                    recall.coverage_filled = merged;
                    recall.coverage_rejected = rejected;
                    recall.coverage_rejected_reasons = reasons;
                }
                Err(e) => {
                    eprintln!("[coverage] fill FAILED: {e}");
                    recall.coverage_error = Some(e.chars().take(300).collect());
                }
            }
        }
    }

    // ── Recall check + tool fill ──
    // Reverse of verify: every labelled money/percent figure in the corpus
    // must be accounted for somewhere in the output. Unaccounted figures and
    // still-empty evidenced fields go to one tool-equipped fill round that
    // can read the actual pages, and may only write empty paths.
    let unaccounted = crate::recall::unaccounted_figures(&final_doc, corpus);
    recall.unaccounted_figures = unaccounted.iter().map(|f| f.summary()).collect();
    let mut fill_issues = coverage::coverage_issues(&final_doc, corpus);
    fill_issues.extend(
        unaccounted
            .iter()
            .map(crate::recall::UnaccountedFigure::issue),
    );
    // A debt deal with no live requested/proposed facility is a structural
    // miss — both pass-A votes can drop the ask and the figure-level recall
    // check can't see it when the amount collides with a cost line (Luton:
    // £35.81m facility == £35.81m TDC). Hand it to the tool fill: the model
    // can only append a facility the corpus actually evidences.
    {
        let is_debt = final_doc
            .pointer("/classification/capital_position")
            .and_then(Value::as_str)
            .is_some_and(|c| c == "debt");
        let facs = final_doc
            .pointer("/canonical_deal/facilities")
            .and_then(Value::as_array);
        let has_ask = facs.is_some_and(|fs| {
            fs.iter().any(|f| {
                matches!(
                    f.get("disposition").and_then(Value::as_str),
                    Some("requested" | "proposed")
                ) && !matches!(
                    f.get("status").and_then(Value::as_str),
                    Some("repaid" | "cancelled" | "superseded")
                )
            })
        });
        if is_debt && !has_ask {
            let next_idx = facs.map_or(0, |fs| fs.len());
            fill_issues.push(validate::Issue {
                code: "no_requested_facility",
                path: "canonical_deal.facilities".into(),
                message: format!("Debt deal extracted with no facility marked disposition \"requested\"/\"proposed\" and no focus facility. Search the corpus for the loan request (facility amount, term, purpose); if the documents evidence one, return it under path \"canonical_deal.facilities[{next_idx}]\" as a facility object (id, name, instrument_type \"debt_facility\", role, commitment {{ amount, currency }}, term_months, disposition \"requested\", status \"candidate\", evidence) and set \"canonical_deal.focus_facility_id\" to its id. Do NOT fabricate — extension/variation deals legitimately have no new-money ask."),
                severity: validate::Severity::Error,
            });
        }
    }
    if !fill_issues.is_empty() {
        let fill_model = coverage_ex.or(details_ex).unwrap_or(extractor);
        eprintln!(
            "[recall] tool fill ({}) — {} unaccounted figures, {} empty fields",
            fill_model.model,
            unaccounted.len(),
            fill_issues.len() - unaccounted.len()
        );
        match repair::tool_fill(fill_model, corpus, &mut final_doc, &fill_issues).await {
            Ok((merged, rejected, reasons)) => {
                eprintln!(
                    "[recall] merged {} paths, rejected {}",
                    merged.len(),
                    rejected.len()
                );
                recall.tool_fill_merged = merged;
                recall.tool_fill_rejected_reasons = reasons;
                recall.tool_fill_rejected = rejected;
            }
            Err(e) => {
                eprintln!("[recall] tool fill FAILED: {e}");
                recall.tool_fill_error = Some(e.chars().take(300).collect());
            }
        }
    }

    // ── Tie checks: figures that don't reconcile with each other (NOI =
    // gross rent while opex is stated, cap ≠ NOI ÷ value, TIC < TDC, stated
    // profit ≠ GDV − TDC) get ONE targeted fix call restricted to the
    // implicated paths — a wrong-but-plausible cell is a misread, not a new
    // field.
    let ties = tie::tie_checks(&final_doc);
    recall.ties_initial = ties
        .iter()
        .filter(|t| t.status == "fail")
        .map(|t| format!("{}: {}", t.id, t.detail))
        .collect();
    if !recall.ties_initial.is_empty() {
        let allowed: std::collections::HashSet<String> = ties
            .iter()
            .filter(|t| t.status == "fail")
            .flat_map(|t| t.fields.iter().cloned())
            .collect();
        let tie_ex = details_ex.or(coverage_ex).unwrap_or(extractor);
        let prompt = format!(
            "{}\n\n## FAILED CHECKS\n{}",
            prompts::TIE_FIX_PROMPT,
            recall.ties_initial.join("\n")
        );
        eprintln!(
            "[ties] {} failed checks — fix call ({}): {}",
            recall.ties_initial.len(),
            tie_ex.model,
            recall.ties_initial.join("; ")
        );
        match tie_ex.call_json(&prompt, &corpus.render()).await {
            Ok((fix_obj, _)) => {
                let mut scoped = serde_json::Map::new();
                if let Some(fixes) = fix_obj.get("fixes").and_then(Value::as_object) {
                    for (p, v) in fixes {
                        if allowed.contains(p) {
                            scoped.insert(p.clone(), v.clone());
                        }
                    }
                }
                let (merged, _rejected, _reasons) =
                    coverage::merge_patch_guarded(&mut final_doc, &scoped, |_| None, false);
                recall.tie_fixes = merged;
            }
            Err(e) => {
                eprintln!("[ties] fix call FAILED: {e}");
                recall.tie_error = Some(e.chars().take(300).collect());
            }
        }
        recall.ties_remaining = tie::tie_checks(&final_doc)
            .iter()
            .filter(|t| t.status == "fail")
            .map(|t| format!("{}: {}", t.id, t.detail))
            .collect();
    }

    // ── Verify pass: cheap ground-check of extracted numbers against the
    // corpus, one cheap LLM call per flagged mismatch. Catches right-field /
    // wrong-cell errors (e.g. cap rate read as 0.04 vs a stated 0.045) that
    // conform can't see because they're shape-correct.
    let verify_model = details_ex.or(coverage_ex).unwrap_or(extractor);
    let (checked, fixes_applied, flagged) =
        crate::verify::verify_extracted_numbers(&mut final_doc, corpus, verify_model).await;
    recall.verify_checked = checked;
    recall.verify_fixes = fixes_applied;
    recall.verify_flagged = flagged;
    if fixes_applied > 0 {
        eprintln!(
            "[pipeline] verify — {} fields checked, {} corrections applied",
            checked, fixes_applied
        );
    }

    recall.coverage_gaps_remaining = coverage::coverage_issues(&final_doc, corpus)
        .into_iter()
        .map(|i| i.path)
        .collect();
    recall.unaccounted_remaining = crate::recall::unaccounted_figures(&final_doc, corpus)
        .iter()
        .map(crate::recall::UnaccountedFigure::summary)
        .collect();
    if !recall.coverage_gaps_remaining.is_empty() || !recall.unaccounted_remaining.is_empty() {
        eprintln!(
            "[recall] still missing — {} evidenced fields, {} corpus figures",
            recall.coverage_gaps_remaining.len(),
            recall.unaccounted_remaining.len()
        );
    }

    let remaining_issues = validate::validate(&final_doc);
    eprintln!(
        "[pipeline] complete — {} repair rounds, {} issues, {:.1}s total",
        repair_rounds,
        remaining_issues.len(),
        pipeline_start.elapsed().as_secs_f64()
    );

    Ok(PipelineResult {
        document: final_doc,
        diagnostics: PipelineDiagnostics {
            pass_times_ms: pass_times,
            repair_rounds,
            issues_initial: initial_errors,
            issues_remaining: remaining_issues,
            recall,
        },
    })
}

fn facility_commitment(f: &Value) -> Option<f64> {
    f.get("commitment")
        .and_then(|c| c.get("amount"))
        .and_then(|v| v.as_f64().or_else(|| v.as_str()?.parse::<f64>().ok()))
}

/// Facilities from a "no facilities" pass-A retry we accept: either the
/// commitment amount is quoted in the corpus, or the facility carries no
/// amount but is marked existing/repaid (an already-in-place facility the
/// model summarised without a figure). Everything else is dropped — the
/// retry exists to recover evidenced omissions, not to admit new guesses.
fn evidenced_retry_facilities(doc: &Value, corpus: &Corpus) -> Vec<Value> {
    doc.get("facilities")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter(|f| match facility_commitment(f) {
            Some(amt) => corpus_has_amount(corpus, amt) && facility_evidence_is_funding(f),
            None => matches!(
                f.get("disposition").and_then(Value::as_str),
                Some("existing") | Some("repaid")
            ),
        })
        .collect()
}

/// Does the facility's own evidence quote describe a financing commitment?
/// A requested/proposed debt facility must quote funding language — a bare
/// cost/budget line ("Land Refinance Cost 5078772", "Total Construction cost
/// 29006250") is a use-of-funds figure, not a loan, even though the amount is
/// literally in the corpus (Luton: pass A promoted cost lines into requested
/// facilities and the implied £35.81m ask never materialised).
/// Equity/existing/repaid facilities carry no commitment claim — always true.
fn facility_evidence_is_funding(f: &Value) -> bool {
    let requested = matches!(
        f.get("disposition").and_then(Value::as_str),
        Some("requested") | Some("proposed")
    );
    let equity = matches!(
        f.get("instrument_type").and_then(Value::as_str),
        Some("ordinary_equity") | Some("preferred_equity")
    );
    if !requested || equity {
        return true;
    }
    const FUNDING: &[&str] = &[
        "loan", "facilit", "debt", "drawdown", "drawn", "borrow", "lender",
        "mortgage", "mezz", "tranche", "advance", "senior", "funding",
    ];
    const SOFT: &[&str] = &["refinanc"];
    const COST: &[&str] = &[
        "cost", "contract", "contingency", "fee", "price", "budget", "expense",
    ];
    let quotes: Vec<String> = f
        .get("evidence")
        .and_then(Value::as_array)
        .map(|evs| {
            evs.iter()
                .filter_map(|e| {
                    e.get("quote")
                        .and_then(Value::as_str)
                        .map(str::to_lowercase)
                })
                .collect()
        })
        .unwrap_or_default();
    !quotes.is_empty()
        && quotes.iter().any(|q| {
            FUNDING.iter().any(|k| q.contains(k))
                || (SOFT.iter().any(|k| q.contains(k))
                    && !COST.iter().any(|k| q.contains(k)))
        })
}

/// Adjudication (or a single agreeing vote) can lack a facility some vote
/// found — but when an evidenced facility is dropped the deal can be left
/// with no debt facility at all (Luton: the £35.81m dev facility vanished,
/// focus resolved to nothing). A dropped facility whose commitment is
/// literally stated in the corpus is evidence-backed, not a vote
/// hallucination — restore it and let downstream validation weigh it.
fn restore_dropped_facilities(
    adj: &mut Value,
    votes: &[&Value],
    corpus: &Corpus,
    recall: &mut RecallDiagnostics,
) {
    let facs_of = |v: &Value| -> Vec<Value> {
        v.get("facilities")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
    };
    let close = |x: f64, y: f64| (x - y).abs() <= 0.005 * x.abs().max(y.abs());
    let kept = facs_of(adj);
    let kept_ids: Vec<&str> = kept
        .iter()
        .filter_map(|f| f.get("id").and_then(Value::as_str))
        .collect();
    let kept_amts: Vec<f64> = kept.iter().filter_map(facility_commitment).collect();
    let mut restored = Vec::new();
    for vote in votes {
        for f in facs_of(vote) {
            let Some(amt) = facility_commitment(&f) else {
                continue;
            };
            if amt < 100_000.0 {
                continue;
            }
            let id = f.get("id").and_then(Value::as_str);
            if id.is_some_and(|i| kept_ids.contains(&i))
                || kept_amts.iter().any(|&g| close(amt, g))
                || kept.iter().any(|g| same_facility(&f, g))
                || restored
                    .iter()
                    .any(|r: &Value| facility_commitment(r).is_some_and(|g| close(amt, g)))
            {
                continue;
            }
            if corpus_has_amount(corpus, amt) && facility_evidence_is_funding(&f) {
                let name = f
                    .get("name")
                    .or_else(|| f.get("id"))
                    .and_then(Value::as_str)
                    .unwrap_or("?");
                recall.adjudication_restored.push(format!("{name} {amt}"));
                restored.push(f);
            }
        }
    }
    if restored.is_empty() {
        return;
    }
    let arr = adj
        .as_object_mut()
        .expect("pass A json")
        .entry("facilities")
        .or_insert_with(|| json!([]));
    if let Some(list) = arr.as_array_mut() {
        list.extend(restored);
    }
}

/// Normalised amount-bearing quotes a facility's evidence carries — the
/// same cost/loan line quoted by two facilities is the fingerprint of an
/// extraction duplicate ("Land Refinance Cost 5078772.32" produced both a
/// "leg" and a "facility" on Luton6).
fn facility_quotes(f: &Value) -> Vec<String> {
    f.get("evidence")
        .and_then(Value::as_array)
        .map(|evs| {
            evs.iter()
                .filter_map(|e| e.get("quote").and_then(Value::as_str))
                .map(|q| q.to_lowercase().split_whitespace().collect::<Vec<_>>().join(" "))
                .filter(|q| !q.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// Two facilities are the same extraction when their commitments match
/// (±0.5%) AND they share at least one identical evidence quote — the
/// adjudication-restored twin that rebuilt Luton6's £39.16m group.
fn same_facility(a: &Value, b: &Value) -> bool {
    let (Some(x), Some(y)) = (facility_commitment(a), facility_commitment(b)) else {
        return false;
    };
    if (x - y).abs() > 0.005 * x.abs().max(y.abs()) {
        return false;
    }
    let bq = facility_quotes(b);
    facility_quotes(a).iter().any(|q| bq.contains(q))
}

/// Final dedupe after restore/adjudication: drop the SECOND facility of
/// every same-amount same-quote pair, keeping the richer entry.
fn dedupe_facilities(chosen: &mut Value) {
    let Some(list) = chosen.get_mut("facilities").and_then(Value::as_array_mut) else {
        return;
    };
    let snapshot = list.clone();
    let mut keep_idx: Vec<usize> = Vec::new();
    for (i, f) in snapshot.iter().enumerate() {
        if let Some(pos) = keep_idx
            .iter()
            .position(|&j| same_facility(f, &snapshot[j]))
        {
            // Keep whichever carries more evidence entries.
            let fe = f.get("evidence").and_then(Value::as_array).map_or(0, Vec::len);
            let j = keep_idx[pos];
            let ge = snapshot[j]
                .get("evidence")
                .and_then(Value::as_array)
                .map_or(0, Vec::len);
            if fe > ge {
                keep_idx[pos] = i;
            }
            continue;
        }
        keep_idx.push(i);
    }
    if keep_idx.len() != snapshot.len() {
        let kept: Vec<Value> = keep_idx.iter().map(|&i| snapshot[i].clone()).collect();
        *list = kept;
    }
}

/// A facility's stated commitment survives a disagreeing vote: when the
/// chosen (adjudicated or vote-1) result has a facility with a null
/// commitment but another vote carries the same facility id with an amount,
/// the stated amount is adopted. A null never overwrites a figure.
fn merge_facility_amounts(chosen: &mut Value, votes: &[Value]) {
    let Some(list) = chosen.get_mut("facilities").and_then(Value::as_array_mut) else {
        return;
    };
    for f in list.iter_mut() {
        if facility_commitment(f).is_some() {
            continue;
        }
        let Some(id) = f.get("id").and_then(Value::as_str) else {
            continue;
        };
        for v in votes {
            let donor = v
                .get("facilities")
                .and_then(Value::as_array)
                .and_then(|fs| {
                    fs.iter().find(|x| {
                        x.get("id").and_then(Value::as_str) == Some(id)
                            && facility_commitment(x).is_some()
                    })
                });
            if let Some(d) = donor {
                if let Some(obj) = f.as_object_mut() {
                    obj.insert(
                        "commitment".to_string(),
                        d.get("commitment").cloned().unwrap_or(Value::Null),
                    );
                }
                break;
            }
        }
    }
}

/// Union of the vote/adjudication party lists by normalised name —
/// adjudication returns PASS A's full format but routinely drops parties
/// the votes found (canonical parties was empty on every prod run).
fn merge_parties(chosen: &mut Value, votes: &[Value]) {
    let norm = |s: &str| s.to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ");
    let mut merged: Vec<Value> = chosen
        .get("parties")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for vote in votes {
        for p in vote.get("parties").and_then(Value::as_array).cloned().unwrap_or_default() {
            let name = p.get("name").and_then(Value::as_str).map(norm);
            let Some(name) = name.filter(|n| !n.is_empty()) else {
                continue;
            };
            if let Some(existing) = merged.iter_mut().find(|m| {
                m.get("name").and_then(Value::as_str).map(norm).as_deref() == Some(name.as_str())
            }) {
                // Merge roles (union) and backfill missing scalar fields.
                if let (Some(er), Some(vr)) = (
                    existing.get_mut("roles").and_then(Value::as_array_mut),
                    p.get("roles").and_then(Value::as_array),
                ) {
                    for r in vr {
                        if !er.contains(r) {
                            er.push(r.clone());
                        }
                    }
                }
                for key in ["company_number", "jurisdiction", "entity_type", "address"] {
                    if existing.get(key).is_none_or(Value::is_null) {
                        if let Some(v) = p.get(key).filter(|v| !v.is_null()) {
                            existing.as_object_mut().unwrap().insert(key.to_string(), v.clone());
                        }
                    }
                }
            } else {
                merged.push(p);
            }
        }
    }
    if !merged.is_empty() || chosen.get("parties").is_some() {
        if let Some(obj) = chosen.as_object_mut() {
            obj.insert("parties".to_string(), Value::Array(merged));
        }
    }
}

/// Compare two pass-A structure extractions on the points that decide the
/// deal's skeleton. Returns human-readable disagreement descriptions; an
/// empty vec means the votes agree and no adjudication is needed.
/// Ids, names and array order are ignored.
pub fn pass_a_disagreements(a: &Value, b: &Value) -> Vec<String> {
    let mut out = Vec::new();

    let close = |x: f64, y: f64| (x - y).abs() <= 0.005 * x.abs().max(y.abs());
    let facs = |v: &Value| -> Vec<Value> {
        v.get("facilities")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
    };
    fn commitment(f: &Value) -> Option<f64> {
        f.get("commitment")
            .and_then(|c| c.get("amount"))
            .and_then(|v| v.as_f64().or_else(|| v.as_str()?.parse::<f64>().ok()))
    }

    // (a) focus facility commitment.
    let focus_amt = |v: &Value| -> Option<f64> {
        let fid = v.get("focus_facility_id")?.as_str()?;
        facs(v)
            .iter()
            .find(|f| f.get("id").and_then(Value::as_str) == Some(fid))
            .and_then(commitment)
    };
    match (focus_amt(a), focus_amt(b)) {
        (Some(x), Some(y)) if !close(x, y) => {
            out.push(format!("focus facility commitment differs: {x} vs {y}"));
        }
        (Some(x), None) => out.push(format!("focus facility commitment {x} only in vote 1")),
        (None, Some(y)) => out.push(format!("focus facility commitment {y} only in vote 2")),
        _ => {}
    }

    // (b) facility commitment multiset: every material facility in one vote
    // must have a within-0.5% counterpart in the other.
    for (n, mine, theirs) in [(1usize, a, b), (2usize, b, a)] {
        for f in facs(mine) {
            let Some(amt) = commitment(&f) else { continue };
            if amt < 100_000.0 {
                continue;
            }
            let matched = facs(theirs)
                .iter()
                .filter_map(commitment)
                .any(|g| close(amt, g));
            if !matched {
                let name = f
                    .get("name")
                    .or_else(|| f.get("id"))
                    .and_then(Value::as_str)
                    .unwrap_or("?");
                out.push(format!("facility {name} {amt} only in vote {n}"));
            }
        }
    }

    // (c)-(e) classification dimensions.
    let cls = |v: &Value, path: &[&str]| -> Option<String> {
        let mut cur = v.get("classification")?;
        for seg in path {
            cur = cur.get(*seg)?;
        }
        cur.as_str().map(str::to_string)
    };
    for (label, path) in [
        ("capital_position", &["capital_position"][..]),
        ("loan_type", &["loan_type", "category"][..]),
        ("project_type", &["project_type", "category"][..]),
    ] {
        match (cls(a, path), cls(b, path)) {
            (Some(x), Some(y)) if x != y => {
                out.push(format!("classification.{label} differs: {x} vs {y}"));
            }
            (Some(x), None) => out.push(format!("classification.{label} {x} only in vote 1")),
            (None, Some(y)) => out.push(format!("classification.{label} {y} only in vote 2")),
            _ => {}
        }
    }
    out
}

/// One pass-B sub-pass: shared header (context + sourced-value rules) plus
/// the pass-specific section. Returns the parsed JSON.
fn pass_b_prompt(header_args: &PassBContext, section: &str, keys: &str) -> String {
    let mut prompt = format!("{}\n\n{}", prompts::PASS_B_HEADER, section);
    prompt = prompt
        .replace("{focus_summary}", header_args.focus_summary)
        .replace(
            "{classification_summary}",
            header_args.classification_summary,
        )
        .replace("{deal_type_guidance}", header_args.guidance);
    prompt.push_str(&format!("\n\nReturn ONLY valid JSON: {keys}"));
    prompt
}

struct PassBContext<'a> {
    focus_summary: &'a str,
    classification_summary: &'a str,
    guidance: &'a str,
}

/// True when a `details` field value is empty under the same rules the
/// coverage pass uses (null / empty array / empty object / blank string,
/// seeing through the WithSource `{value: …}` envelope).
fn details_value_empty(v: &Value) -> bool {
    let doc = json!({ "x": v });
    coverage::path_is_empty(&doc, "x")
}

/// Pass B, split into three concurrent focused calls on one extractor:
/// economics (valuations/phases/economics details/KPIs), income (tenancy
/// schedule, rents, yields, lease metrics) and parties (narrative +
/// descriptive fields). The sub-pass results merge into one
/// {"valuations","phases","details","kpis"} value; a failing sub-pass is
/// recorded in the notes and the others still merge — only all three failing
/// fails the pass. Returns the merged value plus retry/failure notes.
async fn run_pass_b(
    ex: &Extractor,
    corpus: &Corpus,
    focus_summary: &str,
    classification_summary: &str,
    guidance: &str,
) -> Result<(Value, Vec<String>), String> {
    let ctx = PassBContext {
        focus_summary,
        classification_summary,
        guidance,
    };
    let corpus_text = corpus.render();
    let econ_prompt = pass_b_prompt(
        &ctx,
        prompts::PASS_B_ECONOMICS,
        "{ \"valuations\": [...], \"phases\": [...], \"details\": {...}, \"kpis\": {...} }",
    );
    let income_prompt = pass_b_prompt(&ctx, prompts::PASS_B_INCOME, "{ \"details\": {...} }");
    let parties_prompt = pass_b_prompt(&ctx, prompts::PASS_B_PARTIES, "{ \"details\": {...} }");

    let (econ, income, parties) = tokio::join!(
        ex.call_json(&econ_prompt, &corpus_text),
        ex.call_json(&income_prompt, &corpus_text),
        ex.call_json(&parties_prompt, &corpus_text),
    );

    let mut notes: Vec<String> = Vec::new();
    let mut results: Vec<(&str, Option<Value>)> = Vec::new();
    for (name, res) in [
        ("economics", econ),
        ("income", income),
        ("parties", parties),
    ] {
        match res {
            Ok((v, retry)) => {
                if let Some(r) = retry {
                    notes.push(format!("details/{name}: {r}"));
                }
                results.push((name, Some(v)));
            }
            Err(e) => {
                let short: String = e.chars().take(300).collect();
                eprintln!("[pipeline] pass B/{name} FAILED: {short}");
                notes.push(format!("details/{name} FAILED: {short}"));
                results.push((name, None));
            }
        }
    }
    if results.iter().all(|(_, r)| r.is_none()) {
        return Err("all three pass-B sub-passes failed".to_string());
    }

    let merged = merge_pass_b(
        results[0].1.as_ref(),
        results[1].1.as_ref(),
        results[2].1.as_ref(),
    );
    Ok((merged, notes))
}

/// Merge the three pass-B sub-pass results into one
/// {"valuations","phases","details","kpis"} value. Canonical arrays and KPIs
/// come from economics; `details` is the union where, for a key several
/// sub-passes returned, the first non-empty value wins in
/// economics → income → parties order.
fn merge_pass_b(econ: Option<&Value>, income: Option<&Value>, parties: Option<&Value>) -> Value {
    let passes = [econ, income, parties];
    let mut merged = json!({
        "valuations": econ.and_then(|v| v.get("valuations")).cloned().unwrap_or(json!([])),
        "phases": econ.and_then(|v| v.get("phases")).cloned().unwrap_or(json!([])),
        "kpis": econ.and_then(|v| v.get("kpis")).cloned().unwrap_or(json!({})),
    });
    let mut details = serde_json::Map::new();
    let mut keys: Vec<String> = Vec::new();
    for r in passes.iter().flatten() {
        if let Some(o) = r.get("details").and_then(Value::as_object) {
            for k in o.keys() {
                if !keys.contains(k) {
                    keys.push(k.clone());
                }
            }
        }
    }
    for k in keys {
        for r in passes.iter().flatten() {
            let Some(v) = r.get("details").and_then(|d| d.get(&k)) else {
                continue;
            };
            if !details_value_empty(v) {
                details.insert(k.clone(), v.clone());
                break;
            }
            details.entry(k.clone()).or_insert_with(|| v.clone());
        }
    }
    merged["details"] = Value::Object(details);
    merged
}

fn format_classification_summary(classification: &Value) -> String {
    let cp = classification
        .get("capital_position")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let lt = classification
        .get("loan_type")
        .and_then(|v| v.get("category").and_then(|c| c.as_str()))
        .unwrap_or("unknown");
    let lr = classification
        .get("loan_rank")
        .and_then(|v| v.get("category").and_then(|c| c.as_str()))
        .unwrap_or("unknown");
    let pt = classification
        .get("project_type")
        .and_then(|v| v.get("category").and_then(|c| c.as_str()))
        .unwrap_or("unknown");
    format!("{cp} / {lt} / {lr} / {pt}")
}

fn format_focus_summary(focus_id: &Value, facilities: &Value) -> String {
    let fid = focus_id.as_str().unwrap_or("none");
    if fid == "none" {
        return "No focus facility resolved".into();
    }
    let facility = facilities.as_array().and_then(|a| {
        a.iter()
            .find(|f| f.get("id").and_then(|v| v.as_str()) == Some(fid))
    });
    match facility {
        Some(f) => {
            let name = f.get("name").and_then(|v| v.as_str()).unwrap_or(fid);
            let amt = f
                .get("commitment")
                .and_then(|c| c.get("amount"))
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let ccy = f
                .get("commitment")
                .and_then(|c| c.get("currency"))
                .and_then(|v| v.as_str())
                .unwrap_or("GBP");
            let role = f.get("role").and_then(|v| v.as_str()).unwrap_or("?");
            format!("{name} ({role}) — {ccy} {amt}")
        }
        None => format!("focus_facility_id={fid} (not found in facilities)"),
    }
}

fn amount_of(money: &Value) -> Option<f64> {
    money.get("amount").and_then(|v| {
        v.as_str()
            .and_then(|s| s.parse::<f64>().ok())
            .or_else(|| v.as_f64())
    })
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

    let mut focus = facs
        .iter()
        .find(|f| f.get("id").and_then(|v| v.as_str()) == Some(focus_fid));

    if let Some(fc) = focus {
        let focus_amt = fc.get("commitment").and_then(amount_of).unwrap_or(0.0);
        let focus_role = fc.get("role").and_then(|v| v.as_str()).unwrap_or("");
        let same_role_facilities: Vec<&Value> = facs
            .iter()
            .filter(|f| {
                f.get("role").and_then(|v| v.as_str()) == Some(focus_role)
                    || f.get("role").and_then(|v| v.as_str()) == Some("junior_debt")
                        && focus_role == "mezzanine_debt"
                    || f.get("role").and_then(|v| v.as_str()) == Some("mezzanine_debt")
                        && focus_role == "junior_debt"
            })
            .collect();
        for c in &same_role_facilities {
            let cid = c.get("id").and_then(|v| v.as_str()).unwrap_or("");
            if cid == focus_fid {
                continue;
            }
            let c_amt = c.get("commitment").and_then(amount_of).unwrap_or(0.0);
            if c_amt > focus_amt * 1.1 && c_amt < focus_amt * 5.0 {
                let other_facs: Vec<f64> = same_role_facilities
                    .iter()
                    .filter(|x| x.get("id").and_then(|v| v.as_str()) != Some(cid))
                    .map(|x| x.get("commitment").and_then(amount_of).unwrap_or(0.0))
                    .filter(|a| *a > 0.0)
                    .collect();
                let other_sum: f64 = other_facs.iter().sum();
                if other_facs.len() >= 2
                    && other_sum > 0.0
                    && (c_amt - other_sum).abs() / c_amt < 0.05
                {
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
            let sum: f64 = facs
                .iter()
                .filter(|f| f.get("facility_group_id").and_then(|v| v.as_str()) == Some(gid))
                .filter_map(|f| f.get("commitment").and_then(amount_of))
                .sum();
            if sum > 0.0 {
                Some(sum)
            } else {
                None
            }
        })
        .or_else(|| focus.and_then(|f| f.get("commitment")).and_then(amount_of));
    let focus_currency = focus
        .and_then(|f| {
            f.get("commitment")
                .and_then(|c| c.get("currency"))
                .and_then(|v| v.as_str())
        })
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
    let has_deal_scope_facility = facs
        .iter()
        .any(|f| f.get("scope_level").and_then(|v| v.as_str()) == Some("deal"));

    let mut senior_total = 0.0;
    let mut mezz_total = 0.0;
    let mut equity_total = 0.0;
    let mut gross_debt_total = 0.0;
    let mut debt_currency = "gbp".to_string();

    for f in &facs {
        let instrument = f
            .get("instrument_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let role = f.get("role").and_then(|v| v.as_str()).unwrap_or("");
        let disposition = f.get("disposition").and_then(|v| v.as_str()).unwrap_or("");
        let scope = f.get("scope_level").and_then(|v| v.as_str());

        if disposition == "proposed" {
            continue;
        }
        if disposition == "repaid" {
            continue;
        }
        // Exclude project-scope facilities only when the deal itself is a
        // tranche within a project (i.e. there are deal-scope facilities and
        // the focus is not project-scope). When the deal is the project —
        // like Edenbridge — project-scope facilities are the deal.
        if scope == Some("project") && has_deal_scope_facility && !focus_is_project {
            continue;
        }

        let commitment = f.get("commitment").and_then(amount_of).unwrap_or(0.0);
        if commitment < 100_000.0 {
            continue;
        }

        if let Some(c) = f
            .get("commitment")
            .and_then(|c| c.get("currency"))
            .and_then(|v| v.as_str())
        {
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

    let market_value = valuations
        .as_array()
        .and_then(|vs| {
            vs.iter()
                .find(|v| v.get("value_type").and_then(|t| t.as_str()) == Some("market_value"))
                .and_then(|v| v.get("amount").and_then(amount_of))
        })
        .or_else(|| detail_money(details, "market_value"))
        .or_else(|| detail_money(details, "day_1_value"))
        .or_else(|| detail_money(details, "stabilised_value"))
        .or_else(|| detail_money(details, "purchase_price"));

    let gdv = valuations.as_array().and_then(|vs| {
        vs.iter()
            .filter(|v| {
                v.get("value_type").and_then(|t| t.as_str()) == Some("gross_development_value")
            })
            .filter(|v| v.get("phase_id").is_none() || v.get("phase_id") == Some(&json!(null)))
            .map(|v| v.get("amount").and_then(amount_of).unwrap_or(0.0))
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    });

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

        let value_basis = market_value
            .or(detail_money(details, "day_1_value"))
            .or(gdv);
        if let Some(vb) = value_basis {
            if vb > 0.0 {
                let eq_value = equity_total / vb * 100.0;
                kpis["borrower_equity_on_value"] =
                    sourced_percent((eq_value * 100.0).round() / 100.0);
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
            let evidence = ir
                .get("evidence_text")
                .and_then(|v| v.as_str())
                .unwrap_or("");
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

    let gia = details
        .get("gross_internal_area")
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
    details
        .get(field)
        .and_then(|v| v.get("value"))
        .and_then(|v| {
            v.get("amount").and_then(|a| {
                a.as_f64()
                    .or_else(|| a.as_str().and_then(|s| s.parse::<f64>().ok()))
            })
        })
        .filter(|a| *a >= 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pass_a(focus: &str, facilities: Value, classification: Value) -> Value {
        json!({
            "classification": classification,
            "facilities": facilities,
            "focus_facility_id": focus,
        })
    }
    fn fac(id: &str, amount: &str) -> Value {
        json!({ "id": id, "name": id, "commitment": { "amount": amount, "currency": "gbp" } })
    }

    fn corpus_with(text: &str) -> Corpus {
        Corpus::from_pages(vec![crate::documents::DocPage {
            doc_name: "pack.pdf".into(),
            page_label: "1".into(),
            text: text.into(),
        }])
    }

    #[test]
    fn adjudication_restores_evidenced_dropped_facility() {
        // Vote 2 found the £35.81m dev facility; the adjudicator dropped it.
        // Its commitment is stated in the pack → restore.
        let mut adj = pass_a("land", json!([fac("land", "5078772.32")]), json!(null));
        let v1 = pass_a("land", json!([fac("land", "5078772.32")]), json!(null));
        let v2 = pass_a(
            "dev",
            json!([fac("land", "5078772.32"), fac("dev", "35810099.96")]),
            json!(null),
        );
        let corpus = corpus_with("Total facility £35.81m against costs of £35,810,099.96");
        let mut recall = RecallDiagnostics::default();
        restore_dropped_facilities(&mut adj, &[&v1, &v2], &corpus, &mut recall);
        let ids: Vec<&str> = adj["facilities"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|f| f["id"].as_str())
            .collect();
        assert!(ids.contains(&"dev"), "{ids:?}");
        assert_eq!(recall.adjudication_restored.len(), 1);
    }

    #[test]
    fn adjudication_drop_stands_for_unevidenced_facility() {
        // A facility whose commitment appears nowhere in the corpus is a vote
        // hallucination — the adjudicator's drop stands.
        let mut adj = pass_a("land", json!([fac("land", "5078772.32")]), json!(null));
        let v1 = pass_a("land", json!([fac("land", "5078772.32")]), json!(null));
        let v2 = pass_a(
            "dev",
            json!([fac("land", "5078772.32"), fac("dev", "99999999")]),
            json!(null),
        );
        let corpus = corpus_with("Total costs £35,810,099.96");
        let mut recall = RecallDiagnostics::default();
        restore_dropped_facilities(&mut adj, &[&v1, &v2], &corpus, &mut recall);
        assert_eq!(adj["facilities"].as_array().unwrap().len(), 1);
        assert!(recall.adjudication_restored.is_empty());
    }

    #[test]
    fn union_restores_facility_only_in_third_vote() {
        // Votes 1 and 2 agree and are chosen without adjudication; vote 3
        // alone found the evidenced £35.81m dev facility → union restores it.
        let mut chosen = pass_a("land", json!([fac("land", "5078772.32")]), json!(null));
        let v1c = chosen.clone();
        let v2 = pass_a("land", json!([fac("land", "5078772.32")]), json!(null));
        let v3 = pass_a(
            "dev",
            json!([fac("land", "5078772.32"), fac("dev", "35810099.96")]),
            json!(null),
        );
        let corpus = corpus_with("Total facility £35.81m against costs of £35,810,099.96");
        let mut recall = RecallDiagnostics::default();
        restore_dropped_facilities(&mut chosen, &[&v1c, &v2, &v3], &corpus, &mut recall);
        let ids: Vec<&str> = chosen["facilities"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|f| f["id"].as_str())
            .collect();
        assert!(ids.contains(&"dev"), "{ids:?}");
        assert_eq!(recall.adjudication_restored.len(), 1);
    }

    #[test]
    fn union_drop_stands_for_unevidenced_third_vote_facility() {
        let mut chosen = pass_a("land", json!([fac("land", "5078772.32")]), json!(null));
        let v1c = chosen.clone();
        let v2 = pass_a("land", json!([fac("land", "5078772.32")]), json!(null));
        let v3 = pass_a(
            "dev",
            json!([fac("land", "5078772.32"), fac("dev", "99999999")]),
            json!(null),
        );
        let corpus = corpus_with("Total costs £35,810,099.96");
        let mut recall = RecallDiagnostics::default();
        restore_dropped_facilities(&mut chosen, &[&v1c, &v2, &v3], &corpus, &mut recall);
        assert_eq!(chosen["facilities"].as_array().unwrap().len(), 1);
        assert!(recall.adjudication_restored.is_empty());
    }

    #[test]
    fn union_does_not_duplicate_present_facility() {
        // Same facility under a different id in vote 2 — amount matches within
        // tolerance, so it must not be appended a second time.
        let mut chosen = pass_a("dev", json!([fac("dev", "35810099.96")]), json!(null));
        let v2 = pass_a(
            "facility_1",
            json!([fac("facility_1", "35810099.96")]),
            json!(null),
        );
        let corpus = corpus_with("Total facility £35.81m");
        let mut recall = RecallDiagnostics::default();
        restore_dropped_facilities(&mut chosen, &[&v2], &corpus, &mut recall);
        assert_eq!(chosen["facilities"].as_array().unwrap().len(), 1);
        assert!(recall.adjudication_restored.is_empty());
    }

    #[test]
    fn empty_retry_filter_keeps_evidenced_and_null_existing() {
        let retry = json!({
            "facilities": [
                fac("dev", "35810099.96"),
                { "id": "old", "name": "old", "commitment": { "amount": null },
                  "disposition": "existing" },
                { "id": "mezz", "name": "mezz",
                  "commitment": { "amount": "99999999" } },
                { "id": "ghost", "name": "ghost", "disposition": "requested" }
            ]
        });
        let corpus = corpus_with("Total facility £35.81m against costs of £35,810,099.96");
        let kept = evidenced_retry_facilities(&retry, &corpus);
        let ids: Vec<&str> = kept.iter().filter_map(|f| f["id"].as_str()).collect();
        assert_eq!(ids, ["dev", "old"]);
    }

    #[test]
    fn pass_a_agreeing_votes_have_no_disagreements() {
        let cls = json!({ "capital_position": "debt",
            "loan_type": {"category": "bridge"}, "project_type": {"category": "refinance"} });
        let a = pass_a(
            "f1",
            json!([fac("f1", "25750000"), fac("f2", "5000000")]),
            cls.clone(),
        );
        // Order and ids may differ; amounts and classification agree.
        let b = pass_a(
            "g1",
            json!([fac("g2", "5000000"), fac("g1", "25750000")]),
            cls,
        );
        assert!(pass_a_disagreements(&a, &b).is_empty());
    }

    #[test]
    fn pass_a_focus_amount_difference_flagged() {
        let cls = json!({ "capital_position": "debt" });
        let a = pass_a("f1", json!([fac("f1", "25750000")]), cls.clone());
        let b = pass_a("f1", json!([fac("f1", "30000000")]), cls.clone());
        let d = pass_a_disagreements(&a, &b);
        assert!(d.iter().any(|s| s.contains("focus facility commitment")));
        // Within 0.5% is agreement.
        let c = pass_a("f1", json!([fac("f1", "25800000")]), cls.clone());
        assert!(pass_a_disagreements(&a, &c).is_empty());
    }

    #[test]
    fn pass_a_facility_only_in_one_vote_flagged() {
        let cls = json!({ "capital_position": "debt" });
        let a = pass_a(
            "f1",
            json!([fac("f1", "25750000"), fac("extra", "3000000")]),
            cls.clone(),
        );
        let b = pass_a("f1", json!([fac("f1", "25750000")]), cls.clone());
        let d = pass_a_disagreements(&a, &b);
        assert_eq!(d.len(), 1);
        assert!(d[0].contains("extra") && d[0].contains("vote 1"));
        // Sub-100k facilities don't count.
        let small = pass_a(
            "f1",
            json!([fac("f1", "25750000"), fac("fee", "50000")]),
            cls,
        );
        assert!(pass_a_disagreements(&a, &small)
            .iter()
            .all(|s| !s.contains("fee")));
    }

    #[test]
    fn pass_a_classification_difference_flagged() {
        let f = json!([fac("f1", "25750000")]);
        let a = pass_a("f1", f.clone(), json!({ "capital_position": "debt" }));
        let b = pass_a("f1", f, json!({ "capital_position": "equity" }));
        let d = pass_a_disagreements(&a, &b);
        assert!(d.iter().any(|s| s.contains("capital_position")));
    }

    #[test]
    fn pass_b_merge_first_non_empty_wins() {
        let econ = json!({
            "valuations": [{"id": "v1"}],
            "phases": [{"id": "p1"}],
            "kpis": {"k": 1},
            "details": {
                "market_value": {"value": {"amount": "100"}, "source": "e.pdf"},
                "tenure": {"value": null, "source": "e.pdf"},
            },
        });
        let income = json!({
            "details": {
                "tenure": {"value": "freehold", "source": "i.pdf"},
                "market_value": {"value": {"amount": "999"}, "source": "i.pdf"},
            },
        });
        let parties = json!({
            "details": {"tenure": {"value": "leasehold", "source": "p.pdf"}},
        });
        let m = merge_pass_b(Some(&econ), Some(&income), Some(&parties));
        assert_eq!(m["valuations"].as_array().unwrap().len(), 1);
        assert_eq!(m["details"]["market_value"]["value"]["amount"], "100");
        // economics had tenure empty → income's non-empty wins over parties
        assert_eq!(m["details"]["tenure"]["value"], "freehold");
    }

    #[test]
    fn pass_b_merge_tolerates_failed_subpass() {
        let econ = json!({"valuations": [], "phases": [], "kpis": {}, "details": {}});
        let parties = json!({"details": {"address": {"value": {"line1": "x"}, "source": "p.pdf"}}});
        let m = merge_pass_b(Some(&econ), None, Some(&parties));
        assert_eq!(m["details"]["address"]["value"]["line1"], "x");
    }

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

        apply_deterministic_fields(
            &mut details,
            &mut kpis,
            &facilities,
            &focus_id,
            &valuations,
            &classification,
        );

        let la = details["loan_amount"]["value"]["amount"].as_str().unwrap();
        assert_eq!(la, "25750000");

        let gd = details["gross_debt"]["value"]["amount"].as_str().unwrap();
        assert_eq!(gd, "30750000");

        let sd = details["senior_debt"]["value"]["amount"].as_str().unwrap();
        assert_eq!(sd, "25750000");

        let md = details["mezzanine_debt"]["value"]["amount"]
            .as_str()
            .unwrap();
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

        apply_deterministic_fields(
            &mut details,
            &mut kpis,
            &facilities,
            &focus_id,
            &json!([]),
            &json!({}),
        );

        let gd = details["gross_debt"]["value"]["amount"].as_str().unwrap();
        assert_eq!(gd, "25750000");
    }
}

#[cfg(test)]
mod extra_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn restore_skips_and_dedupe_drops_shared_quote_twin() {
        let leg = json!({"id":"leg","name":"Luton land refinance leg",
            "commitment":{"amount":"5078772.32","currency":"GBP"},
            "disposition":"requested","loan_type":"development",
            "evidence":[{"quote":"Land Refinance Cost 5078772.32"}]});
        let facq = json!({"id":"fac","name":"Luton Phase 2 Land Refinance Facility",
            "commitment":{"amount":"5078772.32","currency":"GBP"},
            "disposition":"requested","loan_type":"development",
            "evidence":[{"quote":"Land Refinance Cost 5078772.32"}]});
        let mut chosen = json!({"facilities":[leg.clone(), facq.clone()]});
        dedupe_facilities(&mut chosen);
        assert_eq!(chosen["facilities"].as_array().unwrap().len(), 1);
        let mut adj = json!({"facilities":[leg.clone()]});
        let mut recall = RecallDiagnostics::default();
        let vote = json!({"facilities":[leg, facq]});
        let corpus = Corpus::from_pages(vec![crate::documents::DocPage {
            doc_name: "pack.pdf".into(), page_label: "1".into(),
            text: "Land Refinance Cost 5078772.32".into(),
        }]);
        restore_dropped_facilities(&mut adj, &[&vote], &corpus, &mut recall);
        assert!(recall.adjudication_restored.is_empty());
        assert_eq!(adj["facilities"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn merge_parties_unions_roles_and_fields() {
        let p = |name: &str, roles: &[&str], extra: Value| {
            let mut o = json!({"id": name, "name": name, "entity_type": "company",
                "roles": roles, "evidence": [{"file_id":"f1","document_type":"brochure","quote":"q"}]});
            o.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            o
        };
        let v1 = json!({"parties":[
            p("SSRE Investment 2 Limited", &["borrower"], json!({"jurisdiction":"Guernsey","company_number":"63888"})),
            p("Strawberry Star Investment Manager Guernsey Limited", &["investment_manager"], json!({})),
        ]});
        let v2 = json!({"parties":[
            p("SSRE Investment 2 Limited", &["borrower","sponsor"], json!({})),
            p("Strawberry Star Real Estate PCC Limited", &["parent"], json!({})),
            p("Deep Group London", &["contractor"], json!({})),
        ]});
        let mut chosen = json!({"parties":[]});
        merge_parties(&mut chosen, &[v1, v2]);
        let parties = chosen["parties"].as_array().unwrap();
        assert_eq!(parties.len(), 4);
        let ssre = parties.iter().find(|x| x["name"] == "SSRE Investment 2 Limited").unwrap();
        assert!(ssre["roles"].as_array().unwrap().contains(&json!("sponsor")));
        assert_eq!(ssre["jurisdiction"], "Guernsey");
        let deep = parties.iter().find(|x| x["name"] == "Deep Group London").unwrap();
        assert_eq!(deep["roles"], json!(["contractor"]));
        // conform fills the party-level keys the model omits.
        let mut doc = json!({"canonical_deal":{"parties": parties.clone()}});
        crate::conform::conform_document(&mut doc);
        let c = &doc["canonical_deal"]["parties"][0];
        for key in ["address", "company_number", "evidence_assertion_ids", "evidence"] {
            assert!(c.get(key).is_some(), "missing {key}");
        }
    }
    #[test]
    fn merge_facility_amounts_keeps_stated_amount_over_null_vote() {
        let mut chosen = json!({
            "facilities": [
                {"id": "land_refinance_leg", "name": "Land refinance leg", "disposition": "requested", "commitment": null},
                {"id": "dev_facility", "name": "Dev facility", "disposition": "requested", "commitment": {"amount": "35810099.96", "currency": "GBP"}},
            ]
        });
        let votes = vec![
            chosen.clone(),
            json!({"facilities": [
                {"id": "land_refinance_leg", "name": "Land refinance leg", "disposition": "requested", "commitment": {"amount": "5078772.32", "currency": "GBP"}},
            ]}),
        ];
        merge_facility_amounts(&mut chosen, &votes);
        assert_eq!(
            chosen["facilities"][0]["commitment"]["amount"],
            "5078772.32",
        );
        // An existing commitment is never overwritten.
        assert_eq!(
            chosen["facilities"][1]["commitment"]["amount"],
            "35810099.96",
        );
    }

}
