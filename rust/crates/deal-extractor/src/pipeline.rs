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
    /// Vote-2 or adjudication failure (truncated); pass A still succeeded.
    pub pass_a_error: Option<String>,
    /// Evidenced-but-empty field paths after the repair loop.
    pub coverage_gaps_initial: Vec<String>,
    pub escalated: bool,
    pub escalation_filled: usize,
    pub escalation_error: Option<String>,
    pub coverage_filled: Vec<String>,
    pub coverage_rejected: Vec<String>,
    pub coverage_error: Option<String>,
    /// Corpus figures (labelled money / percentages) found nowhere in the
    /// output, before the tool fill: "<doc> p<page>: <label> <figure>".
    pub unaccounted_figures: Vec<String>,
    pub tool_fill_merged: Vec<String>,
    pub tool_fill_rejected: Vec<String>,
    pub tool_fill_error: Option<String>,
    /// Evidenced-but-empty paths still empty at the end.
    pub coverage_gaps_remaining: Vec<String>,
    /// Corpus figures still unaccounted for at the end.
    pub unaccounted_remaining: Vec<String>,
    pub verify_checked: usize,
    pub verify_fixes: usize,
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

    // Self-consistency: pass A runs twice concurrently on independent
    // extractors. When the votes disagree on the facility graph or
    // classification, a third adjudication call decides the contested points
    // from the evidence — a single sloppy read can no longer set the deal's
    // skeleton. DEAL_EXTRACT_PASS_A_VOTES=1 restores single-vote behaviour.
    let pass_a_votes = std::env::var("DEAL_EXTRACT_PASS_A_VOTES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(2);
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
        recall.pass_a_votes = 2;
        let vote2_ex = details_ex.unwrap_or(extractor);
        eprintln!("[pipeline] pass A vote 2 model: {}", vote2_ex.model);
        let corpus_text = corpus.render();
        let (r1, r2) = tokio::join!(
            structure_model.call_json(prompts::PASS_A_STRUCTURE, &corpus_text),
            vote2_ex.call_json(prompts::PASS_A_STRUCTURE, &corpus_text),
        );
        for (label, res) in [("structure", &r1), ("structure vote 2", &r2)] {
            if let Ok((_, Some(r))) = res {
                recall.pass_retries.push(format!("{label}: {r}"));
            }
        }
        match (r1, r2) {
            (Ok((v1, _)), Ok((v2, _))) => {
                let disagreements = pass_a_disagreements(&v1, &v2);
                if disagreements.is_empty() {
                    v1
                } else {
                    recall.pass_a_disagreements = disagreements;
                    eprintln!(
                        "[pipeline] pass A votes disagree — adjudicating: {}",
                        recall.pass_a_disagreements.join("; ")
                    );
                    let adj_prompt = format!(
                        "{}\n\n## DISAGREEMENTS\n{}\n\n## EXTRACTION 1\n{}\n\n## EXTRACTION 2\n{}",
                        prompts::PASS_A_ADJUDICATE,
                        recall.pass_a_disagreements.join("\n"),
                        serde_json::to_string(&v1).unwrap_or_default(),
                        serde_json::to_string(&v2).unwrap_or_default(),
                    );
                    match structure_model
                        .call_json(&adj_prompt, &corpus.render())
                        .await
                    {
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
                            v1
                        }
                    }
                }
            }
            (Ok((v1, _)), Err(e2)) => {
                eprintln!("[pipeline] pass A vote 2 FAILED: {e2}");
                recall.pass_a_error = Some(e2.chars().take(300).collect());
                v1
            }
            (Err(e1), Ok((v2, _))) => {
                eprintln!("[pipeline] pass A vote 1 FAILED: {e1} — using vote 2");
                recall.pass_a_error = Some(e1.chars().take(300).collect());
                v2
            }
            (Err(e1), Err(e2)) => {
                return Err(format!(
                    "pass A (structure): {e1} (vote 2 also failed: {e2})"
                ));
            }
        }
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
                Ok((merged, rejected)) => {
                    eprintln!(
                        "[coverage] filled {} paths in {:.1}s",
                        merged.len(),
                        t.elapsed().as_secs_f64()
                    );
                    recall.coverage_filled = merged;
                    recall.coverage_rejected = rejected;
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
    if !fill_issues.is_empty() {
        let fill_model = coverage_ex.or(details_ex).unwrap_or(extractor);
        eprintln!(
            "[recall] tool fill ({}) — {} unaccounted figures, {} empty fields",
            fill_model.model,
            unaccounted.len(),
            fill_issues.len() - unaccounted.len()
        );
        match repair::tool_fill(fill_model, corpus, &mut final_doc, &fill_issues).await {
            Ok((merged, rejected)) => {
                eprintln!(
                    "[recall] merged {} paths, rejected {}",
                    merged.len(),
                    rejected.len()
                );
                recall.tool_fill_merged = merged;
                recall.tool_fill_rejected = rejected;
            }
            Err(e) => {
                eprintln!("[recall] tool fill FAILED: {e}");
                recall.tool_fill_error = Some(e.chars().take(300).collect());
            }
        }
    }

    // ── Verify pass: cheap ground-check of extracted numbers against the
    // corpus, one cheap LLM call per flagged mismatch. Catches right-field /
    // wrong-cell errors (e.g. cap rate read as 0.04 vs a stated 0.045) that
    // conform can't see because they're shape-correct.
    let verify_model = details_ex.or(coverage_ex).unwrap_or(extractor);
    let (checked, fixes_applied) =
        crate::verify::verify_extracted_numbers(&mut final_doc, corpus, verify_model).await;
    recall.verify_checked = checked;
    recall.verify_fixes = fixes_applied;
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
