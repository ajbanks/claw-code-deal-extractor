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

    /// Whole-corpus render with non-subject fenced regions masked — the
    /// deterministic ground-truth view for verify/recall.
    #[must_use]
    pub fn render_subject_only(&self) -> String {
        subject_only_text(&self.render())
    }
}

/// Byte spans the TS scope-gatekeeper fenced as NON-subject content:
///
/// * `<SCOPE_FENCE scope="..." subject_deal="false"> ... </SCOPE_FENCE_*>`
///   — dedicated non-subject pages (catchment, market context, sponsor CV)
/// * `<TABLE_FENCE role="..." subject_related="false"> ... </TABLE_FENCE>`
///   — market/context tables that are NOT the subject's tenancy schedule
///   or valuation
/// * `[EXCLUDED_NON_SUBJECT_CONTEXT: ...]` — an inline line marker the
///   gatekeeper substitutes for a verified non-subject line on a mixed page
///
/// Everything inside these spans is readable context for the agent but is
/// not evidence for subject-deal fields — deterministic passes (verify
/// ground-check, recall unaccounted-figure scan) must not treat figures
/// inside them as subject-deal figures.
///
/// Returns `(start, end)` byte spans over `text`.
#[must_use]
pub fn non_subject_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut pos = 0usize;
    while let Some(off) = text[pos..].find('<') {
        let start = pos + off;
        let rest = &text[start..];
        let (flag, close) = if rest.starts_with("<SCOPE_FENCE") {
            ("subject_deal=\"false\"", "</SCOPE_FENCE_")
        } else if rest.starts_with("<TABLE_FENCE") {
            ("subject_related=\"false\"", "</TABLE_FENCE>")
        } else {
            pos = start + 1;
            continue;
        };
        let Some(hdr_end) = rest.find('>') else { break };
        let header = &rest[..hdr_end];
        if header.contains(flag) {
            // Unterminated fence: mask to end of text — safer to drop the
            // tail than treat fenced non-subject content as subject.
            let end = rest
                .find(close)
                .map(|c| {
                    // The SCOPE close tag carries a scope suffix
                    // (</SCOPE_FENCE_CATCHMENT_DEMOGRAPHICS>) — scan from the
                    // close-tag start to its '>' for both fence kinds.
                    let tag_start = start + c;
                    match text[tag_start..].find('>') {
                        Some(gt) => tag_start + gt + 1,
                        None => text.len(),
                    }
                })
                .unwrap_or(text.len());
            spans.push((start, end));
            pos = end;
        } else {
            pos = start + hdr_end + 1;
        }
    }
    // Inline excluded-line markers: the marker wraps the original line
    // (`[EXCLUDED_NON_SUBJECT_CONTEXT: <original text>]`) — mask the whole line.
    let mut line_start = 0usize;
    for line in text.split_inclusive('\n') {
        if line.contains("EXCLUDED_NON_SUBJECT_CONTEXT") {
            spans.push((line_start, line_start + line.len()));
        }
        line_start += line.len();
    }
    spans.sort_unstable();
    spans
}

/// `text` with every non-subject fenced span replaced by spaces — byte
/// offsets are preserved so callers can map positions back to the original.
#[must_use]
pub fn subject_only_text(text: &str) -> String {
    let spans = non_subject_spans(text);
    if spans.is_empty() {
        return text.to_string();
    }
    // Replace every non-newline byte inside a non-subject span with a space —
    // the string keeps the same length so byte offsets stay valid.
    let mut bytes = text.as_bytes().to_vec();
    for (s, e) in spans {
        let end = e.min(bytes.len());
        for b in &mut bytes[s..end] {
            if *b != b'\n' {
                *b = b' ';
            }
        }
    }
    // Only ASCII spaces were written; the result is still valid UTF-8.
    String::from_utf8(bytes).unwrap_or_else(|_| text.to_string())
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

    #[test]
    fn scope_fence_masks_non_subject_page() {
        let text = "Subject value £10m\n\
                    <SCOPE_FENCE scope=\"catchment_demographics\" subject_deal=\"false\">\n\
                    [DO NOT EXTRACT — context]\n\
                    Catchment population 42,000\n\
                    </SCOPE_FENCE_CATCHMENT_DEMOGRAPHICS>\n\
                    Subject rent £500,000";
        let masked = subject_only_text(text);
        assert_eq!(masked.len(), text.len());
        assert!(masked.contains("£10m"));
        assert!(masked.contains("£500,000"));
        assert!(!masked.contains("42,000"));
        assert!(!masked.contains("SCOPE_FENCE"));
    }

    #[test]
    fn table_fence_masks_only_non_subject_tables() {
        let text = "<TABLE_FENCE role=\"market_comparable\" subject_related=\"false\">\n\
                    Comp sale £2,000,000\n\
                    </TABLE_FENCE>\n\
                    <TABLE_FENCE role=\"sources_and_uses\" subject_related=\"true\">\n\
                    Total £5,000,000\n\
                    </TABLE_FENCE>";
        let masked = subject_only_text(text);
        assert!(!masked.contains("2,000,000"));
        assert!(masked.contains("5,000,000"));
    }

    #[test]
    fn excluded_line_marker_masks_that_line() {
        let text = "Passing rent £100,000\n\
                    [EXCLUDED_NON_SUBJECT_CONTEXT: Regional comp £9,999,999]\n\
                    Passing yield 6%";
        let masked = subject_only_text(text);
        assert!(masked.contains("£100,000"));
        assert!(!masked.contains("9,999,999"));
        assert!(masked.contains("6%"));
    }

    #[test]
    fn unfenced_text_is_untouched() {
        let text = "Everything £1,234 here stays";
        assert_eq!(subject_only_text(text), text);
    }
}
