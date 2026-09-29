//! HTTP extraction service: accepts extracted document pages from the TS
//! backend, runs whole-corpus LLM extraction, validates against the shared
//! JSON Schema + domain invariants, and drives a tool-assisted repair loop.
//! The agent owns correctness — the service never coerces values itself.

use std::time::Instant;

use base64::Engine as _;
use deal_extractor::corpus::Corpus;
use deal_extractor::documents::{self, DocPage};
use deal_extractor::extract::Extractor;
use deal_extractor::pipeline;
use deal_extractor::repair;
use deal_extractor::validate::{self, Issue};
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ─── Request / response types ───────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ExtractRequest {
    /// Optional deal identifier, echoed back in diagnostics.
    #[serde(default)]
    pub deal_id: Option<String>,
    /// OpenAI model to use. Defaults to gpt-5.4.
    #[serde(default = "default_model")]
    pub model: String,
    /// Max validate→repair rounds after round 0. Default 3.
    #[serde(default = "default_max_repair_rounds")]
    pub max_repair_rounds: usize,
    /// Use multi-pass pipeline (5 focused passes). Default true.
    #[serde(default = "default_multi_pass")]
    pub multi_pass: bool,
    /// Documents to extract from.
    pub documents: Vec<InputDocument>,
}

fn default_model() -> String {
    "gpt-5.4".to_string()
}
fn default_max_repair_rounds() -> usize {
    3
}
fn default_multi_pass() -> bool {
    true
}

#[derive(Debug, Deserialize)]
pub struct InputDocument {
    /// File name as it appears to the agent (evidence citations use this).
    pub name: String,
    /// "pages" | "xlsx_base64" | "pdf_base64" | "text"
    pub kind: String,
    /// For kind = "pages": extracted page text.
    #[serde(default)]
    pub pages: Vec<InputPage>,
    /// For kind = "*_base64": the raw document bytes, base64-encoded.
    #[serde(default)]
    pub data: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct InputPage {
    /// Page label (e.g. "1", "Sheet: Cash Flow").
    pub label: String,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct ExtractResponse {
    pub canonical_deal: Option<Value>,
    pub details: Option<Value>,
    pub kpis: Option<Value>,
    pub classification: Option<Value>,
    pub diagnostics: Diagnostics,
}

#[derive(Debug, Serialize)]
pub struct Diagnostics {
    pub rounds_used: usize,
    pub issues_initial: usize,
    pub issues_remaining: Vec<IssueView>,
    pub elapsed_ms: u64,
    pub corpus: CorpusStats,
}

#[derive(Debug, Serialize)]
pub struct IssueView {
    pub code: &'static str,
    pub path: String,
    pub message: String,
    pub severity: &'static str,
}

impl From<&Issue> for IssueView {
    fn from(i: &Issue) -> Self {
        Self {
            code: i.code,
            path: i.path.clone(),
            message: i.message.clone(),
            severity: match i.severity {
                validate::Severity::Error => "error",
                validate::Severity::Warn => "warn",
            },
        }
    }
}

#[derive(Debug, Serialize)]
pub struct CorpusStats {
    pub documents: usize,
    pub pages: usize,
    pub chars: usize,
}

#[derive(Debug, Deserialize)]
pub struct ValidateRequest {
    pub document: Value,
}

#[derive(Debug, Serialize)]
pub struct ValidateResponse {
    pub valid: bool,
    pub issues: Vec<IssueView>,
}

#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub error: String,
}

// ─── Core logic ─────────────────────────────────────────────────────────────

/// Build the corpus from request documents.
pub fn build_corpus(documents: &[InputDocument]) -> Result<Corpus, String> {
    let mut pages: Vec<DocPage> = Vec::new();
    for doc in documents {
        match doc.kind.as_str() {
            "pages" | "text" => {
                if doc.kind == "text" {
                    pages.push(DocPage {
                        doc_name: doc.name.clone(),
                        page_label: "1".into(),
                        text: doc.data.clone().unwrap_or_default(),
                    });
                } else {
                    for p in &doc.pages {
                        pages.push(DocPage {
                            doc_name: doc.name.clone(),
                            page_label: p.label.clone(),
                            text: p.text.clone(),
                        });
                    }
                }
            }
            "xlsx_base64" | "xls_base64" | "pdf_base64" => {
                let data = doc.data.as_deref().ok_or_else(|| {
                    format!(
                        "document \"{}\" kind {} requires `data`",
                        doc.name, doc.kind
                    )
                })?;
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(data)
                    .map_err(|e| format!("document \"{}\": bad base64: {e}", doc.name))?;
                pages.extend(documents::load_document_bytes(&doc.name, &doc.kind, &bytes));
            }
            other => {
                return Err(format!(
                    "document \"{}\": unknown kind \"{other}\"",
                    doc.name
                ));
            }
        }
    }
    Ok(Corpus::from_pages(pages))
}

/// Run the full extraction pipeline.
pub async fn run_extraction(req: &ExtractRequest) -> Result<ExtractResponse, String> {
    let start = Instant::now();
    let corpus = build_corpus(&req.documents)?;
    let stats = CorpusStats {
        documents: corpus.document_names().len(),
        pages: corpus.pages.len(),
        chars: corpus.char_count(),
    };
    if corpus.pages.is_empty() {
        return Err("no document pages provided".into());
    }

    eprintln!("[extract] deal={} model={} docs={} pages={} chars={} multi_pass={}",
        req.deal_id.as_deref().unwrap_or("-"), req.model,
        stats.documents, stats.pages, stats.chars, req.multi_pass);

    let extractor = Extractor::new(&req.model)?;

    if req.multi_pass {
        let result = pipeline::run_multi_pass(&extractor, &corpus, req.max_repair_rounds).await?;
        let diag = result.diagnostics;
        Ok(ExtractResponse {
            canonical_deal: result.document.get("canonical_deal").cloned(),
            details: result.document.get("details").cloned(),
            kpis: result.document.get("kpis").cloned(),
            classification: result.document.get("classification").cloned(),
            diagnostics: Diagnostics {
                rounds_used: diag.repair_rounds,
                issues_initial: diag.issues_initial,
                issues_remaining: diag.issues_remaining.iter().map(IssueView::from).collect(),
                elapsed_ms: start.elapsed().as_millis() as u64,
                corpus: stats,
            },
        })
    } else {
        eprintln!("[extract] single-pass mode — round 0 calling LLM...");
        let candidate = extractor.extract(&corpus).await.map_err(|e| {
            eprintln!("[extract] round 0 FAILED: {e}");
            e
        })?;
        let initial_issues = validate::validate(&candidate);
        let initial_errors = validate::error_count(&initial_issues);
        eprintln!("[extract] round 0 done — {initial_errors} validation errors");
        let (doc, issues, rounds) =
            repair::repair_loop(&extractor, &corpus, candidate, req.max_repair_rounds).await;
        eprintln!("[extract] repair done — {} rounds, {} issues remaining, {:.1}s",
            rounds, issues.len(), start.elapsed().as_secs_f64());

        Ok(ExtractResponse {
            canonical_deal: doc.get("canonical_deal").cloned(),
            details: doc.get("details").cloned(),
            kpis: doc.get("kpis").cloned(),
            classification: doc.get("classification").cloned(),
            diagnostics: Diagnostics {
                rounds_used: rounds,
                issues_initial: initial_errors,
                issues_remaining: issues.iter().map(IssueView::from).collect(),
                elapsed_ms: start.elapsed().as_millis() as u64,
                corpus: stats,
            },
        })
    }
}

/// Validate a deal document without any LLM call.
pub fn run_validation(req: &ValidateRequest) -> ValidateResponse {
    let issues = validate::validate(&req.document);
    let valid = validate::error_count(&issues) == 0;
    ValidateResponse {
        valid,
        issues: issues.iter().map(IssueView::from).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_corpus_from_pages() {
        let req_docs = vec![InputDocument {
            name: "a.pdf".into(),
            kind: "pages".into(),
            pages: vec![
                InputPage {
                    label: "1".into(),
                    text: "page one".into(),
                },
                InputPage {
                    label: "2".into(),
                    text: "page two".into(),
                },
            ],
            data: None,
        }];
        let c = build_corpus(&req_docs).unwrap();
        assert_eq!(c.pages.len(), 2);
        assert_eq!(c.page("a.pdf", "2").unwrap().text, "page two");
    }

    #[test]
    fn unknown_kind_rejected() {
        let req_docs = vec![InputDocument {
            name: "x".into(),
            kind: "bogus".into(),
            pages: vec![],
            data: None,
        }];
        assert!(build_corpus(&req_docs).is_err());
    }

    #[test]
    fn validate_endpoint_shape() {
        let resp = run_validation(&ValidateRequest {
            document: serde_json::json!({"canonical_deal": {}}),
        });
        // Missing required fields → schema violations + no facilities.
        assert!(!resp.issues.is_empty());
    }
}
