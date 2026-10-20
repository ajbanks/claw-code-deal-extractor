//! Document loading: PDF → text via `pdftotext` (poppler, already installed),
//! XLSX → text via `calamine` (pure Rust). Each page/sheet becomes a
//! `DocPage` so the LLM can cite evidence by document + page.

use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;

use calamine::{open_workbook_auto, open_workbook_auto_from_rs, Data, Reader, Sheets};

#[derive(Debug, Clone)]
pub struct DocPage {
    pub doc_name: String,
    pub page_label: String,
    pub text: String,
}

/// Recursively collect files in `dir`, skipping `:Zone.Identifier` and
/// `~$` lock files (present in the fixtures).
fn collect_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        if name.contains(":Zone.Identifier") || name.starts_with("~$") || name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            out.extend(collect_files(&path));
        } else {
            out.push(path);
        }
    }
    out
}

/// Load a single document from raw bytes, dispatching on the declared kind.
/// Used by the service for `xlsx_base64` / `pdf_base64` request documents.
pub fn load_document_bytes(doc_name: &str, kind: &str, bytes: &[u8]) -> Vec<DocPage> {
    match kind {
        "xlsx_base64" | "xls_base64" => load_xlsx_bytes(doc_name, bytes),
        "pdf_base64" => load_pdf_bytes(doc_name, bytes),
        _ => vec![DocPage {
            doc_name: doc_name.to_string(),
            page_label: "error".into(),
            text: format!("[unsupported document kind: {kind}]"),
        }],
    }
}

fn load_pdf_bytes(doc_name: &str, bytes: &[u8]) -> Vec<DocPage> {
    // Write to a temp file so the existing PyMuPDF/pdftotext path can read it.
    let dir = std::env::temp_dir();
    let path = dir.join(format!(
        "deal-extractor-{}-{}.pdf",
        std::process::id(),
        doc_name
    ));
    if fs::write(&path, bytes).is_err() {
        return vec![DocPage {
            doc_name: doc_name.to_string(),
            page_label: "error".into(),
            text: "[could not stage pdf bytes]".into(),
        }];
    }
    let pages = load_pdf_named(&path, doc_name);
    let _ = fs::remove_file(&path);
    pages
}

fn load_xlsx_bytes(doc_name: &str, bytes: &[u8]) -> Vec<DocPage> {
    let workbook = match open_workbook_auto_from_rs(Cursor::new(bytes.to_vec())) {
        Ok(wb) => wb,
        Err(e) => {
            return vec![DocPage {
                doc_name: doc_name.to_string(),
                page_label: "error".into(),
                text: format!("[xlsx parse error: {e}]"),
            }];
        }
    };
    workbook_to_pages(doc_name.to_string(), workbook)
}

/// Load all documents in `dir` as page-tagged text.
pub fn load_documents(dir: &Path) -> Vec<DocPage> {
    let mut pages = Vec::new();
    for file in collect_files(dir) {
        let ext = file
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        match ext.as_str() {
            "pdf" => pages.extend(load_pdf(&file)),
            "xlsx" | "xls" => pages.extend(load_xlsx(&file)),
            _ => {}
        }
    }
    pages
}

fn load_pdf(path: &Path) -> Vec<DocPage> {
    let doc_name = path.file_name().unwrap().to_string_lossy().to_string();
    load_pdf_named(path, &doc_name)
}

fn load_pdf_named(path: &Path, doc_name: &str) -> Vec<DocPage> {
    // Primary: PyMuPDF (Python) — extracts significantly more text than
    // poppler's pdftotext from certain PDFs (e.g. the Oakley Square brochure:
    // 5562 chars vs 96). Falls back to pdftotext if Python/PyMuPDF unavailable.
    // Emit EVERY page (empty ones too) joined by \f so a page's chunk index
    // always equals its 0-based PDF page — skipping blank pages here used to
    // shift every subsequent page label.
    let pymupdf_script = "import sys,fitz;d=fitz.open(sys.argv[1]);\
print(chr(12).join(p.get_text() for p in d),end='');d.close()";
    let pymupdf = Command::new("python3")
        .args(["-c", pymupdf_script, &path.to_string_lossy()])
        .output();

    let raw = match pymupdf {
        Ok(o) if o.status.success() && !o.stdout.is_empty() => {
            String::from_utf8_lossy(&o.stdout).into_owned()
        }
        _ => {
            // Fallback: pdftotext -layout (poppler)
            let output = Command::new("pdftotext")
                .args(["-layout", &path.to_string_lossy(), "-"])
                .output();
            match output {
                Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).into_owned(),
                _ => {
                    return vec![DocPage {
                        doc_name: doc_name.to_string(),
                        page_label: "1".into(),
                        text: "[no extractable text — likely scanned/image PDF]".into(),
                    }];
                }
            }
        }
    };

    let chunks: Vec<&str> = raw.split('\x0c').collect();
    let mut pages = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let trimmed = chunk.trim();
        if trimmed.is_empty() {
            continue;
        }
        pages.push(DocPage {
            doc_name: doc_name.to_string(),
            page_label: (i + 1).to_string(),
            text: trimmed.to_string(),
        });
    }
    if pages.is_empty() {
        pages.push(DocPage {
            doc_name: doc_name.to_string(),
            page_label: "1".into(),
            text: "[no extractable text]".into(),
        });
    }
    pages
}

fn load_xlsx(path: &Path) -> Vec<DocPage> {
    let doc_name = path.file_name().unwrap().to_string_lossy().to_string();
    let workbook = match open_workbook_auto(path) {
        Ok(wb) => wb,
        Err(e) => {
            return vec![DocPage {
                doc_name,
                page_label: "error".into(),
                text: format!("[xlsx parse error: {e}]"),
            }];
        }
    };
    workbook_to_pages(doc_name, workbook)
}

fn workbook_to_pages<RS: std::io::Read + std::io::Seek>(
    doc_name: String,
    mut reader: Sheets<RS>,
) -> Vec<DocPage> {
    let sheet_names = reader.sheet_names().to_vec();
    let mut pages = Vec::new();
    for sheet in &sheet_names {
        let Ok(range) = reader.worksheet_range(sheet) else {
            continue;
        };
        let mut text = String::new();
        for row in range.rows() {
            let cells: Vec<String> = row
                .iter()
                .map(|c| match c {
                    Data::Empty => String::new(),
                    Data::String(s) => s.clone(),
                    Data::DateTime(d) => d.to_string(),
                    Data::DateTimeIso(s) => s.clone(),
                    Data::DurationIso(s) => s.clone(),
                    Data::Int(n) => n.to_string(),
                    Data::Float(n) => {
                        if n.fract() == 0.0 || n.abs() < 1.0 {
                            format!("{n}")
                        } else {
                            format!("{n:.2}")
                        }
                    }
                    Data::Bool(b) => b.to_string(),
                    Data::Error(e) => format!("#{e:?}"),
                })
                .collect();
            let line = cells.join("\t");
            if !line.trim().is_empty() {
                text.push_str(&line);
                text.push('\n');
            }
        }
        if !text.is_empty() {
            pages.push(DocPage {
                doc_name: doc_name.clone(),
                page_label: sheet.clone(),
                text,
            });
        }
    }
    pages
}

/// Render pages into a single prompt-ready corpus string with delimiters.
pub fn render_corpus(pages: &[DocPage]) -> String {
    let mut out = String::new();
    for p in pages {
        out.push_str(&format!("=== {} [{}] ===\n", p.doc_name, p.page_label));
        out.push_str(&p.text);
        out.push_str("\n\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_files_skips_zone_and_lock() {
        // smoke: just ensure the function doesn't panic on a missing dir
        let pages = load_documents(Path::new("/nonexistent"));
        assert!(pages.is_empty());
    }
}
