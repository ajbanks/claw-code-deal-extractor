//! Per-page OCR for image-only pages. The TS side drops PDF pages with no
//! text layer, so appraisal screenshots and unit schedules inside an
//! otherwise-textual PDF used to vanish from the corpus entirely. For every
//! page whose extracted text is thin we render the page to PNG with PyMuPDF
//! and transcribe it with a vision model, then splice the transcription back
//! into the corpus at the right page label.

use std::process::Command;

use serde_json::{json, Value};

use crate::corpus::Corpus;
use crate::documents::DocPage;

/// Pages thinner than this are treated as image-only and sent to vision.
const THIN_CHARS: usize = 80;
/// A transcription shorter than this is treated as "no text on the page".
const MIN_TRANSCRIPTION: usize = 20;
const OCR_CONCURRENCY: usize = 4;
const DEFAULT_MAX_PAGES: usize = 40;

const OCR_PROMPT: &str = "Transcribe ALL text on this page of a property-finance document. Preserve tables as tab-separated rows (one row per line, header row first). Keep every number exactly as printed, with currency symbols, units and % signs. Output only the transcription, no commentary. If the page has no text, output nothing.";

/// A PDF source for page OCR: `(doc_name, pdf_bytes, explicit_pages)`.
/// `Some(pages)` skips thin-page detection; `None` means detect by rendering.
pub type PdfSource = (String, Vec<u8>, Option<Vec<usize>>);

/// One rendered page: `(1-based page number, png bytes)`.
type RenderedPage = (usize, Vec<u8>);

#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct OcrStats {
    /// Thin/explicit pages found before the per-extraction cap.
    pub pages_considered: usize,
    /// Pages whose transcription was spliced into the corpus.
    pub pages_transcribed: usize,
    pub pages_failed: usize,
    /// First ≤5 failures, truncated.
    pub errors: Vec<String>,
    /// Pages dropped because the extraction hit DEAL_EXTRACT_OCR_MAX_PAGES.
    pub skipped_over_cap: usize,
}

/// 1-based page numbers in `1..=page_count` whose DocPage is missing or
/// carries under `THIN_CHARS` of text — the candidates for vision OCR.
#[must_use]
pub fn thin_pages(corpus: &Corpus, doc: &str, page_count: usize) -> Vec<usize> {
    (1..=page_count)
        .filter(|n| {
            corpus
                .page(doc, &n.to_string())
                .is_none_or(|p| p.text.trim().chars().count() < THIN_CHARS)
        })
        .collect()
}

/// Render `pages` (1-based) of `pdf` to PNG via PyMuPDF. Returns the PDF's
/// total page count plus one PNG buffer per rendered page. An empty `pages`
/// is valid — callers use it to learn the page count.
pub fn render_pages(pdf: &[u8], pages: &[usize]) -> Result<(usize, Vec<RenderedPage>), String> {
    let dir = std::env::temp_dir().join(format!(
        "deal-extractor-ocr-{}-{:x}",
        std::process::id(),
        pdf.as_ptr() as usize
    ));
    std::fs::create_dir_all(&dir).map_err(|e| format!("ocr temp dir: {e}"))?;
    let pdf_path = dir.join("doc.pdf");
    let result = (|| -> Result<(usize, Vec<RenderedPage>), String> {
        std::fs::write(&pdf_path, pdf).map_err(|e| format!("ocr temp pdf: {e}"))?;
        let script = "import sys,os,fitz
d=fitz.open(sys.argv[1]);out=sys.argv[2]
print(d.page_count)
for n in (int(x) for x in sys.argv[3:]):
    if 1<=n<=d.page_count:
        d.load_page(n-1).get_pixmap(dpi=150).save(os.path.join(out,f'{n}.png'))
d.close()";
        let mut args = vec![
            "-c".to_string(),
            script.to_string(),
            pdf_path.to_string_lossy().into_owned(),
            dir.to_string_lossy().into_owned(),
        ];
        args.extend(pages.iter().map(usize::to_string));
        let out = Command::new("python3")
            .args(&args)
            .output()
            .map_err(|e| format!("ocr render spawn: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "ocr render failed: {}",
                String::from_utf8_lossy(&out.stderr)
                    .chars()
                    .take(200)
                    .collect::<String>()
            ));
        }
        let stdout = String::from_utf8_lossy(&out.stdout);
        let page_count: usize = stdout
            .lines()
            .next()
            .and_then(|l| l.trim().parse().ok())
            .ok_or_else(|| "ocr render: no page count in output".to_string())?;
        let mut pngs = Vec::new();
        for &n in pages {
            if !(1..=page_count).contains(&n) {
                continue;
            }
            let path = dir.join(format!("{n}.png"));
            match std::fs::read(&path) {
                Ok(b) => pngs.push((n, b)),
                Err(e) => return Err(format!("ocr render: read page {n} png: {e}")),
            }
        }
        Ok((page_count, pngs))
    })();
    let _ = std::fs::remove_dir_all(&dir);
    result
}

/// The JSON body sent to the vision model for one page image.
fn ocr_request_body(model: &str, png: &[u8]) -> Value {
    use base64::Engine as _;
    let b64 = base64::engine::general_purpose::STANDARD.encode(png);
    json!({
        "model": model,
        "max_completion_tokens": 4000,
        "messages": [{
            "role": "user",
            "content": [
                { "type": "text", "text": OCR_PROMPT },
                { "type": "image_url",
                  "image_url": { "url": format!("data:image/png;base64,{b64}") } },
            ],
        }],
    })
}

/// Transcribe one rendered page image with the vision model.
pub async fn transcribe_page(model: &str, png: &[u8]) -> Result<String, String> {
    let base = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let key = std::env::var("OPENAI_API_KEY").map_err(|_| "OPENAI_API_KEY not set".to_string())?;
    let body = ocr_request_body(model, png);
    let resp = reqwest::Client::new()
        .post(format!("{}/chat/completions", base.trim_end_matches('/')))
        .bearer_auth(key)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("ocr request: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text: String = resp
            .text()
            .await
            .unwrap_or_default()
            .chars()
            .take(200)
            .collect();
        return Err(format!("ocr request failed: {status} {text}"));
    }
    let json: Value = resp
        .json()
        .await
        .map_err(|e| format!("ocr response parse: {e}"))?;
    json.get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| "ocr response had no message content".to_string())
}

/// OCR every thin page of each PDF source back into the corpus.
///
/// `pdfs` entries are `(doc_name, pdf_bytes, explicit_pages)` — an explicit
/// `Some(pages)` list (from the TS side, which already knows which labels
/// were thin) skips the thin-page scan; `None` means detect thin pages by
/// rendering the PDF for its page count. Disabled entirely with
/// `DEAL_EXTRACT_OCR=false`. Capped at `DEAL_EXTRACT_OCR_MAX_PAGES` pages
/// per extraction (default 40), transcription concurrency 4.
pub async fn ocr_thin_pages(corpus: &mut Corpus, pdfs: &[PdfSource], model: &str) -> OcrStats {
    let mut stats = OcrStats::default();
    if std::env::var("DEAL_EXTRACT_OCR").ok().as_deref() == Some("false") || pdfs.is_empty() {
        return stats;
    }
    let max_pages = std::env::var("DEAL_EXTRACT_OCR_MAX_PAGES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_PAGES);

    // Resolve the page list per document (explicit or thin-detected) first so
    // the global cap applies across documents, then render.
    let mut budget = max_pages;
    let mut rendered: Vec<(String, Vec<RenderedPage>)> = Vec::new();
    for (name, bytes, explicit) in pdfs {
        let pages: Vec<usize> = match explicit {
            Some(p) => p.iter().copied().filter(|n| *n >= 1).collect(),
            None => match render_pages(bytes, &[]) {
                Ok((count, _)) => thin_pages(corpus, name, count),
                Err(e) => {
                    push_error(&mut stats, format!("{name}: {e}"));
                    continue;
                }
            },
        };
        stats.pages_considered += pages.len();
        let todo: Vec<usize> = if pages.len() <= budget {
            budget -= pages.len();
            pages
        } else {
            stats.skipped_over_cap += pages.len() - budget;
            let todo = pages.into_iter().take(budget).collect();
            budget = 0;
            todo
        };
        if todo.is_empty() {
            continue;
        }
        match render_pages(bytes, &todo) {
            Ok((_, pngs)) => rendered.push((name.clone(), pngs)),
            Err(e) => {
                stats.pages_failed += todo.len();
                push_error(&mut stats, format!("{name}: {e}"));
            }
        }
    }

    // Transcribe rendered pages, concurrency 4.
    let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(OCR_CONCURRENCY));
    let mut set = tokio::task::JoinSet::new();
    for (name, pngs) in rendered {
        for (n, png) in pngs {
            let sem = sem.clone();
            let model = model.to_string();
            let doc = name.clone();
            set.spawn(async move {
                let _permit = sem.acquire_owned().await;
                let text = transcribe_page(&model, &png).await;
                (doc, n, text)
            });
        }
    }
    let mut transcriptions: Vec<(String, usize, String)> = Vec::new();
    while let Some(joined) = set.join_next().await {
        match joined {
            Ok((doc, n, Ok(text))) => {
                if text.trim().chars().count() >= MIN_TRANSCRIPTION {
                    stats.pages_transcribed += 1;
                    transcriptions.push((doc, n, text));
                }
            }
            Ok((doc, n, Err(e))) => {
                stats.pages_failed += 1;
                push_error(&mut stats, format!("{doc} p{n}: {e}"));
            }
            Err(e) => {
                stats.pages_failed += 1;
                push_error(&mut stats, format!("ocr task: {e}"));
            }
        }
    }

    // Splice transcriptions into the corpus at the right page labels —
    // replacing an existing thin DocPage or inserting the missing one — then
    // keep each document's pages in page order.
    let mut touched: Vec<String> = Vec::new();
    for (doc, n, text) in transcriptions {
        let label = n.to_string();
        if let Some(p) = corpus
            .pages
            .iter_mut()
            .find(|p| p.doc_name == doc && p.page_label == label)
        {
            p.text = text;
        } else {
            corpus.pages.push(DocPage {
                doc_name: doc.clone(),
                page_label: label,
                text,
            });
        }
        if !touched.contains(&doc) {
            touched.push(doc);
        }
    }
    for doc in touched {
        let mut order: Vec<usize> = (0..corpus.pages.len())
            .filter(|&i| corpus.pages[i].doc_name == doc)
            .collect();
        order.sort_by_key(|&i| {
            corpus.pages[i]
                .page_label
                .parse::<usize>()
                .unwrap_or(usize::MAX)
        });
        let sorted: Vec<DocPage> = order.iter().map(|&i| corpus.pages[i].clone()).collect();
        for (slot, page) in order.into_iter().zip(sorted) {
            corpus.pages[slot] = page;
        }
    }
    stats
}

fn push_error(stats: &mut OcrStats, e: String) {
    if stats.errors.len() < 5 {
        stats.errors.push(e.chars().take(300).collect());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus_with(doc: &str, pages: &[(&str, &str)]) -> Corpus {
        Corpus::from_pages(
            pages
                .iter()
                .map(|(label, text)| DocPage {
                    doc_name: doc.to_string(),
                    page_label: (*label).to_string(),
                    text: (*text).to_string(),
                })
                .collect(),
        )
    }

    #[test]
    fn thin_pages_flags_missing_and_thin() {
        let c = corpus_with(
            "a.pdf",
            &[
                ("1", &"x".repeat(200)),
                ("2", "tiny"),
                ("4", &"y".repeat(100)),
            ],
        );
        // page 3 missing, page 2 thin, pages 1/4 full, page 5 beyond count
        assert_eq!(thin_pages(&c, "a.pdf", 4), vec![2, 3]);
    }

    #[test]
    fn thin_pages_empty_when_all_full() {
        let c = corpus_with("a.pdf", &[("1", &"x".repeat(200)), ("2", &"y".repeat(200))]);
        assert!(thin_pages(&c, "a.pdf", 2).is_empty());
    }

    #[test]
    fn ocr_body_shape() {
        let body = ocr_request_body("gpt-5.4-2026-03-05", b"pngbytes");
        assert_eq!(body["model"], "gpt-5.4-2026-03-05");
        assert_eq!(body["max_completion_tokens"], 4000);
        let content = &body["messages"][0]["content"];
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image_url");
        let url = content[1]["image_url"]["url"].as_str().unwrap();
        assert!(url.starts_with("data:image/png;base64,"));
        assert!(url.contains("cG5nYnl0ZXM="));
    }
}
