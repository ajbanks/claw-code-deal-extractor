//! Repair loop: the agent is shown its candidate plus the issues it
//! violated, and asked to return the corrected complete document. On repair
//! turns it gets read-only document tools so it can verify values against
//! the source rather than guessing.

use std::collections::BTreeMap;

use api::{
    InputContentBlock, InputMessage, MessageRequest, OutputContentBlock, ToolChoice,
    ToolDefinition, ToolResultContentBlock,
};
use serde_json::Value;

use crate::corpus::Corpus;
use crate::extract::{self, Extractor};
use crate::prompts;
use crate::validate::{self, Issue, Severity};

// 8 turns was too few for a large repair issue list — the round aborted
// mid-fix and discarded all progress. 16 keeps the bound while giving a
// ~100-issue round room to finish.
const MAX_TOOL_TURNS: usize = 16;

/// The repair-turn tool surface. Read-only corpus access.
fn repair_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "list_documents".into(),
            description: Some("List the documents in this deal with page counts.".into()),
            input_schema: serde_json::json!({
                "type": "object", "properties": {}, "additionalProperties": false,
            }),
        },
        ToolDefinition {
            name: "search_documents".into(),
            description: Some(
                "Case-insensitive substring search across all documents. \
                 Returns matching snippets with document name and page label."
                    .to_string(),
            ),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "text to search for"},
                    "max_results": {"type": "integer", "description": "cap on hits (default 10)"},
                },
                "required": ["query"],
                "additionalProperties": false,
            }),
        },
        ToolDefinition {
            name: "read_page".into(),
            description: Some("Return the full text of one document page.".into()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "document": {"type": "string"},
                    "page": {"type": "string"},
                },
                "required": ["document", "page"],
                "additionalProperties": false,
            }),
        },
    ]
}

fn dispatch_tool(corpus: &Corpus, name: &str, input: &Value) -> String {
    match name {
        "list_documents" => {
            let docs = corpus.documents();
            serde_json::to_string_pretty(&docs).unwrap_or_else(|_| "[]".into())
        }
        "search_documents" => {
            let q = input.get("query").and_then(|v| v.as_str()).unwrap_or("");
            let max = input
                .get("max_results")
                .and_then(|v| v.as_u64())
                .unwrap_or(10) as usize;
            let hits = corpus.search(q, max);
            if hits.is_empty() {
                format!("no matches for \"{q}\"")
            } else {
                hits.iter()
                    .map(|h| format!("[{} p{}] {}", h.doc_name, h.page_label, h.snippet))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        }
        "read_page" => {
            let doc = input.get("document").and_then(|v| v.as_str()).unwrap_or("");
            let page = input.get("page").and_then(|v| v.as_str()).unwrap_or("");
            corpus
                .page(doc, page)
                .map(|p| format!("=== {} [{}] ===\n{}", p.doc_name, p.page_label, p.text))
                .unwrap_or_else(|| format!("error: no page \"{page}\" in document \"{doc}\""))
        }
        other => format!("error: unknown tool \"{other}\""),
    }
}

fn format_issues(issues: &[Issue]) -> String {
    issues
        .iter()
        .filter(|i| i.severity == Severity::Error)
        .map(|i| format!("- [{}] {} — {}", i.code, i.path, i.message))
        .collect::<Vec<_>>()
        .join("\n")
}

/// One repair round: show the agent its candidate + the issues, let it use
/// tools to verify, and return the corrected document.
pub(crate) async fn repair_round(
    ex: &Extractor,
    corpus: &Corpus,
    candidate: &Value,
    issues: &[Issue],
) -> Result<Value, String> {
    let issues_text = format_issues(issues);
    let candidate_text = serde_json::to_string_pretty(candidate).unwrap_or_default();

    let user_text = format!(
        "{}\n\n## ISSUES TO FIX\n{}\n\n## YOUR PREVIOUS OUTPUT\n{}\n",
        prompts::REPAIR_PROMPT,
        issues_text,
        candidate_text,
    );

    let fixes = run_tool_loop(ex, corpus, user_text, "repair").await?;
    if fixes.is_empty() {
        return Err("repair returned an empty patch".to_string());
    }
    let mut next = candidate.clone();
    for (path, value) in &fixes {
        crate::coverage::set_path(&mut next, path, value.clone());
    }
    eprintln!("[repair] applied {} patch entries", fixes.len());
    Ok(next)
}

/// Recall fill: the issues name evidenced-but-empty fields and corpus
/// figures missing from the output. The model reads the pages with tools and
/// returns a patch; only paths that are currently empty (or array appends)
/// and keep the document valid are merged. Returns (merged, rejected).
pub(crate) async fn tool_fill(
    ex: &Extractor,
    corpus: &Corpus,
    doc: &mut Value,
    issues: &[Issue],
) -> Result<(Vec<String>, Vec<String>), String> {
    let user_text = format!(
        "{}\n\n## GAPS\n{}\n\n## CURRENT EXTRACTION\n{}\n",
        prompts::RECALL_FILL_PROMPT,
        issues
            .iter()
            .map(|i| format!("- [{}] {} — {}", i.code, i.path, i.message))
            .collect::<Vec<_>>()
            .join("\n"),
        serde_json::to_string(doc).unwrap_or_default(),
    );
    let fixes = run_tool_loop(ex, corpus, user_text, "recall").await?;
    Ok(crate::coverage::merge_patch_guarded(
        doc,
        &fixes,
        |_| None,
        true,
    ))
}

/// Drive a read-only tool loop until the model answers with a
/// `{"fixes": {path: value}}` patch; returns the patch map.
async fn run_tool_loop(
    ex: &Extractor,
    corpus: &Corpus,
    user_text: String,
    label: &str,
) -> Result<serde_json::Map<String, Value>, String> {
    let mut messages = vec![InputMessage::user_text(user_text)];
    let tools = repair_tools();

    for _turn in 0..MAX_TOOL_TURNS {
        let mut extra_body = BTreeMap::new();
        extra_body.insert(
            "response_format".into(),
            serde_json::json!({ "type": "json_object" }),
        );
        let req = MessageRequest {
            model: ex.model.clone(),
            max_tokens: ex.max_tokens,
            messages: messages.clone(),
            system: Some(prompts::SYSTEM.to_string()),
            tools: Some(tools.clone()),
            tool_choice: Some(ToolChoice::Auto),
            stream: false,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            reasoning_effort: Some("none".to_string()),
            extra_body,
        };

        let r = ex
            .client
            .send_message(&req)
            .await
            .map_err(|e| format!("API error: {e:?}"))?;
        ex.usage.record(label, &r.usage);
        extract::check_truncated(&r, ex.max_tokens)?;

        // Collect tool uses.
        let tool_uses: Vec<(String, String, Value)> = r
            .content
            .iter()
            .filter_map(|b| match b {
                OutputContentBlock::ToolUse { id, name, input } => {
                    Some((id.clone(), name.clone(), input.clone()))
                }
                _ => None,
            })
            .collect();

        if tool_uses.is_empty() || r.stop_reason.as_deref() != Some("tool_use") {
            // Final answer — a {"fixes": {path: value}} patch, applied to the
            // candidate. Patch mode means the model cannot drop data.
            let text =
                extract::response_text(&r).ok_or_else(|| format!("no text in {label} response"))?;
            let raw = extract::parse_json(&text)?;
            return Ok(raw
                .get("fixes")
                .and_then(|f| f.as_object())
                .cloned()
                .unwrap_or_else(|| raw.as_object().cloned().unwrap_or_default()));
        }

        // Echo the assistant turn, then feed tool results.
        messages.push(InputMessage {
            role: "assistant".into(),
            content: r
                .content
                .iter()
                .filter_map(|b| match b {
                    OutputContentBlock::Text { text } => {
                        Some(InputContentBlock::Text { text: text.clone() })
                    }
                    OutputContentBlock::ToolUse { id, name, input } => {
                        Some(InputContentBlock::ToolUse {
                            id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                        })
                    }
                    _ => None,
                })
                .collect(),
        });
        let results: Vec<InputContentBlock> = tool_uses
            .iter()
            .map(|(id, name, input)| {
                let text = dispatch_tool(corpus, name, input);
                InputContentBlock::ToolResult {
                    tool_use_id: id.clone(),
                    content: vec![ToolResultContentBlock::Text { text }],
                    is_error: false,
                }
            })
            .collect();
        messages.push(InputMessage {
            role: "user".into(),
            content: results,
        });
    }

    Err(format!(
        "{label} round exceeded {MAX_TOOL_TURNS} tool turns"
    ))
}

/// The repair loop: validate → if issues, repair with tools → re-validate.
/// Keeps the best candidate; stops early when the issue count stops falling.
/// Returns the best document and the remaining issues.
pub async fn repair_loop(
    ex: &Extractor,
    corpus: &Corpus,
    mut candidate: Value,
    max_rounds: usize,
) -> (Value, Vec<Issue>, usize) {
    let mut issues = validate::validate(&candidate);
    let mut best_errors = validate::error_count(&issues);
    let mut rounds_used = 0;

    for _ in 0..max_rounds {
        if best_errors == 0 {
            break;
        }
        rounds_used += 1;
        eprintln!("[repair] round {rounds_used} — {best_errors} errors, calling LLM...");
        match repair_round(ex, corpus, &candidate, &issues).await {
            Ok(mut next) => {
                // Mechanical conformance before scoring — don't let shape
                // errors count for (or against) the model's repair.
                crate::conform::conform_document(&mut next);
                let next_issues = validate::validate(&next);
                let next_errors = validate::error_count(&next_issues);
                eprintln!(
                    "[repair] round {rounds_used} done — {next_errors} errors (was {best_errors})"
                );
                if next_errors >= best_errors {
                    break;
                }
                candidate = next;
                issues = next_issues;
                best_errors = next_errors;
            }
            Err(e) => {
                eprintln!("[repair] round {rounds_used} FAILED: {e}");
                break;
            }
        }
    }

    (candidate, issues, rounds_used)
}

/// Canonical arrays a full-document repair may never shrink — retained for
/// the legacy full-doc path; patch-mode repair cannot drop data.
#[allow(dead_code)]
const PRESERVED_ARRAYS: &[&str] = &[
    "facilities",
    "valuations",
    "phases",
    "capital_events",
    "capital_snapshots",
];

/// Restore canonical arrays the repair shrank, a dropped focus id, and
/// details keys that vanished. Returns the restored paths.
#[allow(dead_code)]
pub(crate) fn restore_dropped_structure(prev: &Value, next: &mut Value) -> Vec<String> {
    let mut restored = Vec::new();
    let len = |v: Option<&Value>| v.and_then(Value::as_array).map_or(0, Vec::len);

    if let (Some(p_cd), Some(n_cd)) = (prev.get("canonical_deal"), next.get_mut("canonical_deal")) {
        for key in PRESERVED_ARRAYS {
            let prev_len = len(p_cd.get(*key));
            if prev_len > 0 && len(n_cd.get(*key)) < prev_len {
                n_cd[*key] = p_cd[*key].clone();
                restored.push(format!("canonical_deal.{key}"));
            }
        }
        let had_focus = p_cd.get("focus_facility_id").is_some_and(|v| !v.is_null());
        let lost_focus = n_cd.get("focus_facility_id").map_or(true, Value::is_null);
        if had_focus && lost_focus {
            n_cd["focus_facility_id"] = p_cd["focus_facility_id"].clone();
            if let Some(s) = p_cd.get("focus_resolution_status") {
                n_cd["focus_resolution_status"] = s.clone();
            }
            restored.push("canonical_deal.focus_facility_id".into());
        }
    }

    if let (Some(p_d), Some(n_d)) = (
        prev.get("details").and_then(Value::as_object),
        next.get_mut("details").and_then(Value::as_object_mut),
    ) {
        for (k, v) in p_d {
            if !n_d.contains_key(k) {
                n_d.insert(k.clone(), v.clone());
                restored.push(format!("details.{k}"));
            }
        }
    }
    restored
}

#[cfg(test)]
mod restore_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn restores_deleted_facilities_and_focus() {
        let prev = json!({
            "canonical_deal": {"facilities": [{"id": "a"}, {"id": "b"}], "focus_facility_id": "a", "phases": []},
            "details": {"loan_amount": {"value": 1}}
        });
        let mut next = json!({
            "canonical_deal": {"facilities": [], "focus_facility_id": null, "phases": [{"id": "p"}]},
            "details": {}
        });
        let r = restore_dropped_structure(&prev, &mut next);
        assert_eq!(
            next["canonical_deal"]["facilities"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(next["canonical_deal"]["focus_facility_id"], "a");
        assert_eq!(
            next["canonical_deal"]["phases"].as_array().unwrap().len(),
            1
        );
        assert!(next["details"].get("loan_amount").is_some());
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn keeps_edits_that_do_not_shrink() {
        let prev =
            json!({"canonical_deal": {"facilities": [{"id": "a", "amount": "1"}]}, "details": {}});
        let mut next = json!({"canonical_deal": {"facilities": [{"id": "a", "amount": "2"}]}, "details": {"x": 1}});
        assert!(restore_dropped_structure(&prev, &mut next).is_empty());
        assert_eq!(next["canonical_deal"]["facilities"][0]["amount"], "2");
    }
}
