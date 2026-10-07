//! Coverage pass — the "missing field with evidence" check.
//!
//! The two whole-corpus passes occasionally leave a populated-document gap:
//! a contract field that is null/empty even though the corpus plainly
//! contains the answer (unit schedules, named principals, planning refs).
//! Whole-corpus repair rounds are too blunt to fix that — this module
//! produces targeted `Issue`s naming the field and the evidence, which the
//! existing tool-equipped repair round (search_documents / read_page) can
//! fill cheaply.

use serde_json::Value;

use crate::corpus::Corpus;
use crate::validate::{Issue, Severity};

/// High-value fields where an empty value plus corpus evidence should
/// trigger a targeted fill. Each entry: document path, keywords that count
/// as evidence, and a hint about what to extract.
const FIELD_SPECS: &[(&str, &[&str], &str)] = &[
    ("details.sponsor_principals", &["director", "shareholder", "track record"], "every named principal/director/shareholder with a per-person track_record list"),
    ("details.sponsor_track_record", &["track record", "previous projects", "delivered"], "the sponsor's delivered-scheme track record"),
    ("details.residential_units", &["unit mix", "bedroom", "apartments", "units"], "the per-type unit schedule (bedrooms, quantity, area, GDV)"),
    ("details.gross_internal_area", &["gia", "sq ft", "sqft"], "gross internal area as { value, unit }"),
    ("details.net_internal_area", &["nia", "net internal"], "net internal area as { value, unit }"),
    ("details.market_value", &["market value", "valuation"], "the current market value (day-1 / existing value), distinct from GDV"),
    ("details.gross_development_value", &["gdv", "gross development value"], "the scheme GDV"),
    ("details.total_development_cost", &["total cost", "development cost", "build cost"], "total development cost"),
    ("details.address", &["street", "postcode"], "the site address incl. line1, city, postcode"),
    ("details.exit_plan", &["exit", "sale", "dispose", "refinanc"], "the asset-level exit (unit sales / disposal), not facility repayment"),
    ("details.tenure", &["freehold", "leasehold", "tenure"], "tenure type and length"),
    ("details.purchase_price", &["price", "asking", "offers in excess"], "purchase price / asking price"),
    ("details.planning_status", &["planning", "consent", "prior approval"], "planning status description"),
    ("details.commercial_units", &["tenant", "lease", "rent", "occupier"], "the commercial tenancy schedule"),
    ("details.developer_equity", &["equity", "sponsor", "inject"], "sponsor/developer equity contribution"),
    ("canonical_deal.planning_applications", &["planning", "application ref", "prior approval", "pacbsd", "ful"], "every planning application referenced (ref, status, local authority)"),
    ("canonical_deal.phases", &["phase"], "scheme phases with unit_count and status"),
    ("canonical_deal.facility_parties", &["lender", "borrower", "spv"], "borrower and lender parties per facility"),
    ("canonical_deal.valuations", &["valuation", "valuer", "report"], "valuation entries (value, basis, valuer, date)"),
];

const MAX_ISSUES: usize = 12;

/// True when the field is absent or carries no usable value. Understands the
/// WithSource wrapper: `{ value: null }` / `{ value: [] }` count as empty.
fn field_is_empty(doc: &Value, path: &str) -> bool {
    let mut cur = doc;
    for seg in path.split('.') {
        match cur.get(seg) {
            Some(v) => cur = v,
            None => return true,
        }
    }
    // Unwrap WithSource envelopes to inspect the payload.
    if let Some(inner) = cur.get("value").filter(|_| {
        cur.get("source").is_some() || cur.get("source_type").is_some() || cur.get("evidence_text").is_some()
    }) {
        cur = inner;
    }
    match cur {
        Value::Null => true,
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
        Value::String(s) => s.trim().is_empty(),
        _ => false,
    }
}

/// Emit an `Issue` per empty field that has corpus evidence, naming the
/// source documents so the repair model knows where to look.
#[must_use]
pub fn coverage_issues(doc: &Value, corpus: &Corpus) -> Vec<Issue> {
    let mut issues = Vec::new();
    for (path, keywords, hint) in FIELD_SPECS {
        if issues.len() >= MAX_ISSUES {
            break;
        }
        if !field_is_empty(doc, path) {
            continue;
        }
        // Evidence = keyword hits; keep the strongest snippets inline so the
        // fill call can extract without needing document tools.
        let mut hits = Vec::new();
        let mut doc_names: Vec<String> = Vec::new();
        for kw in *keywords {
            for hit in corpus.search(kw, 3) {
                if !doc_names.contains(&hit.doc_name) {
                    doc_names.push(hit.doc_name.clone());
                }
                hits.push(format!("{} ({}): {}", hit.doc_name, hit.page_label, hit.snippet.trim()));
            }
            if !doc_names.is_empty() && hits.len() >= 4 {
                break;
            }
        }
        if doc_names.is_empty() {
            continue;
        }
        issues.push(Issue {
            code: "missing_with_evidence",
            path: (*path).to_string(),
            message: format!(
                "{path} — extract {hint}. Evidence:\n{}",
                hits.iter().take(4).cloned().collect::<Vec<_>>().join("\n"),
            ),
            severity: Severity::Error,
        });
    }
    issues
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn corpus_with(text: &str) -> Corpus {
        Corpus::from_pages(vec![crate::documents::DocPage {
            doc_name: "doc.pdf".to_string(),
            page_label: "p1".to_string(),
            text: text.to_string(),
        }])
    }

    #[test]
    fn empty_field_with_evidence_emits_issue() {
        let doc = json!({ "details": {}, "canonical_deal": {} });
        let c = corpus_with("Unit mix: 24 studio apartments, 68 one-bed units. GIA 194,871 sq ft.");
        let issues = coverage_issues(&doc, &c);
        let paths: Vec<&str> = issues.iter().map(|i| i.path.as_str()).collect();
        assert!(paths.contains(&"details.residential_units"));
        assert!(paths.contains(&"details.gross_internal_area"));
    }

    #[test]
    fn populated_field_is_not_flagged() {
        let doc = json!({
            "details": { "residential_units": { "value": [{"bedrooms": 1, "quantity": 68}], "source": "x.pdf" } },
            "canonical_deal": {},
        });
        let c = corpus_with("unit mix bedrooms apartments");
        let issues = coverage_issues(&doc, &c);
        assert!(!issues.iter().any(|i| i.path == "details.residential_units"));
    }

    #[test]
    fn empty_field_without_evidence_is_not_flagged() {
        let doc = json!({ "details": {} });
        let c = corpus_with("unrelated prose with no keywords at all");
        let issues = coverage_issues(&doc, &c);
        assert!(issues.is_empty());
    }
}

const PATCH_PROMPT: &str = "You are filling specific gaps in an extracted deal document. For each field below, the extraction left it empty but the quoted evidence shows the value. Return a JSON object mapping each path to its extracted value.\n\nSHAPE RULES:\n- paths under `details.` → a WithSource object   `{\"value\": <value>, \"source\": \"<document name>\", \"source_type\": \"extracted\", \"evidence_text\": \"<short quote>\", \"notes\": null, \"scope\": null}`\n- paths under `canonical_deal.` → the raw value only (array or object), matching   the shapes used elsewhere in the document\n- extract only what the evidence supports; do not guess or round to invented precision\n- omit a path entirely if the evidence does not answer it\n\nFIELDS TO FILL:\n";

/// Ask the coverage model for a `{path: value}` patch and merge the paths
/// that don't introduce validation errors. Returns the merged paths.
pub async fn fill_missing(
    ex: &crate::extract::Extractor,
    doc: &mut Value,
    issues: &[Issue],
) -> Result<Vec<String>, String> {
    if issues.is_empty() {
        return Ok(Vec::new());
    }
    let prompt = format!(
        "{PATCH_PROMPT}{}\n\nDOCUMENT (for shape reference):\n{}",
        issues.iter().map(|i| i.message.as_str()).collect::<Vec<_>>().join("\n\n"),
        serde_json::to_string(doc).map_err(|e| e.to_string())?,
    );
    let text = ex.call_text(&prompt).await?;
    let patch: Value = crate::extract::parse_json(&text)?;
    let obj = patch.as_object().ok_or("coverage patch is not an object")?;

    let baseline_errors: std::collections::HashSet<String> = crate::validate::validate(doc)
        .iter().map(|i| i.path.clone()).collect();
    // Default evidence source per path — the document where the keyword hit.
    let source_for = |path: &str| -> Option<String> {
        issues.iter().find(|i| i.path == path).and_then(|i| {
            i.message.lines().find(|l| l.contains(": ") && l.contains("("))
                .map(|l| l.split(" (").next().unwrap_or("").trim().to_string())
                .filter(|n| !n.is_empty())
        })
    };
    let mut merged = Vec::new();
    let mut rejected = Vec::new();
    for (path, value) in obj {
        let snapshot = doc.clone();
        // details.* values are WithSource — a missing source would break
        // provenance, so default it to the evidence document.
        let mut v = value.clone();
        if path.starts_with("details.") {
            if let Some(o) = v.as_object_mut() {
                let src_empty = o.get("source").and_then(|s| s.as_str()).map_or(true, |s| s.trim().is_empty());
                if src_empty {
                    if let Some(name) = source_for(path) {
                        o.insert("source".to_string(), Value::String(name));
                    }
                }
            }
        }
        set_path(doc, path, v);
        let new_issues = crate::validate::validate(doc);
        let bad = new_issues.iter().any(|i| i.path.starts_with(path.as_str()) && !baseline_errors.contains(&i.path));
        let worse = new_issues.iter().filter(|i| !baseline_errors.contains(&i.path)).count() > 6;
        if bad || worse {
            *doc = snapshot;
            rejected.push(path.clone());
        } else {
            merged.push(path.clone());
        }
    }
    if !rejected.is_empty() {
        eprintln!("[coverage] rejected patch paths (validation): {}", rejected.join(", "));
    }
    Ok(merged)
}

/// Set `a.b.c = value`, creating intermediate objects as needed.
fn set_path(doc: &mut Value, path: &str, value: Value) {
    let mut cur = doc;
    let segs: Vec<&str> = path.split('.').collect();
    for seg in &segs[..segs.len() - 1] {
        let entry = cur.as_object_mut().map(|o| o.entry(seg.to_string()).or_insert_with(|| Value::Object(serde_json::Map::new())));
        match entry {
            Some(v) => cur = v,
            None => return,
        }
    }
    if let Some(o) = cur.as_object_mut() {
        o.insert(segs[segs.len() - 1].to_string(), value);
    }
}
