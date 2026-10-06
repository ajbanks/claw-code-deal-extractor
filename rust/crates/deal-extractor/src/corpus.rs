//! In-memory document corpus — the unit the extractor works over.
//! Pages come either from local files (the CLI) or from the service
//! request (TS-extracted PDF pages, or base64 attachments).

use std::path::Path;

use crate::documents::{self, DocPage};

#[derive(Debug, Clone)]
pub struct Corpus {
    pub pages: Vec<DocPage>,
}

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub doc_name: String,
    pub page_label: String,
    pub snippet: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct DocumentSummary {
    pub name: String,
    pub pages: usize,
    pub chars: usize,
}

impl Corpus {
    pub fn from_dir(dir: &Path) -> Self {
        Self {
            pages: documents::load_documents(dir),
        }
    }

    #[must_use]
    pub fn from_pages(pages: Vec<DocPage>) -> Self {
        Self { pages }
    }

    /// Render the whole corpus for a prompt. Same delimiter format the
    /// extractor was built on.
    #[must_use]
    pub fn render(&self) -> String {
        documents::render_corpus(&self.pages)
    }

    #[must_use]
    pub fn char_count(&self) -> usize {
        self.pages.iter().map(|p| p.text.len()).sum()
    }

    /// Unique document names in insertion order.
    #[must_use]
    pub fn document_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        for p in &self.pages {
            if !names.contains(&p.doc_name) {
                names.push(p.doc_name.clone());
            }
        }
        names
    }

    /// Per-document summaries for the `list_documents` tool.
    #[must_use]
    pub fn documents(&self) -> Vec<DocumentSummary> {
        let mut out: Vec<DocumentSummary> = Vec::new();
        for p in &self.pages {
            match out.iter_mut().find(|d| d.name == p.doc_name) {
                Some(d) => {
                    d.pages += 1;
                    d.chars += p.text.len();
                }
                None => out.push(DocumentSummary {
                    name: p.doc_name.clone(),
                    pages: 1,
                    chars: p.text.len(),
                }),
            }
        }
        out
    }

    /// Case-insensitive substring search across pages. Each hit returns the
    /// document, page label, and a window of context around the match.
    #[must_use]
    pub fn search(&self, query: &str, max_results: usize) -> Vec<SearchHit> {
        let needle = query.to_lowercase();
        if needle.is_empty() {
            return Vec::new();
        }
        let mut hits = Vec::new();
        for p in &self.pages {
            let hay = p.text.to_lowercase();
            let mut start = 0;
            while let Some(pos) = hay[start..].find(&needle) {
                let abs = start + pos;
                // Snap the context window to UTF-8 char boundaries — byte
                // offsets can land inside multi-byte chars (e.g. '£').
                let mut lo = abs.saturating_sub(160);
                while lo > 0 && !p.text.is_char_boundary(lo) {
                    lo -= 1;
                }
                let mut hi = (abs + needle.len() + 160).min(p.text.len());
                while hi < p.text.len() && !p.text.is_char_boundary(hi) {
                    hi += 1;
                }
                hits.push(SearchHit {
                    doc_name: p.doc_name.clone(),
                    page_label: p.page_label.clone(),
                    snippet: p.text[lo..hi].replace('\n', " "),
                });
                if hits.len() >= max_results {
                    return hits;
                }
                start = abs + needle.len();
            }
        }
        hits
    }

    /// Fetch a single page by document name + page label.
    #[must_use]
    pub fn page(&self, doc_name: &str, page_label: &str) -> Option<&DocPage> {
        self.pages
            .iter()
            .find(|p| p.doc_name == doc_name && p.page_label == page_label)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> Corpus {
        Corpus::from_pages(vec![
            DocPage {
                doc_name: "a.pdf".into(),
                page_label: "1".into(),
                text: "The bridge loan is £25.75m".into(),
            },
            DocPage {
                doc_name: "a.pdf".into(),
                page_label: "2".into(),
                text: "Shojin senior debt repaid".into(),
            },
            DocPage {
                doc_name: "b.xlsx".into(),
                page_label: "Sheet1".into(),
                text: "Units 189".into(),
            },
        ])
    }

    #[test]
    fn search_is_case_insensitive() {
        let c = corpus();
        let hits = c.search("BRIDGE LOAN", 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_name, "a.pdf");
        assert!(hits[0].snippet.contains("£25.75m"));
    }

    #[test]
    fn search_respects_max_results() {
        let c = corpus();
        assert!(c.search("e", 1).len() == 1);
    }

    #[test]
    fn page_lookup() {
        let c = corpus();
        assert!(c.page("a.pdf", "2").unwrap().text.contains("Shojin"));
        assert!(c.page("a.pdf", "9").is_none());
    }

    #[test]
    fn document_summaries() {
        let c = corpus();
        let docs = c.documents();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].pages, 2);
    }
}
