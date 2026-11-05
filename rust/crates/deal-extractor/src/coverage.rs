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
/// trigger a targeted fill.
///
/// Presence classes, from observing what UK property-finance packs actually
/// carry (broker teasers, IMs, term sheets, valuation reports, appraisals):
///
/// - **Core** (`min_keywords: 1`, no gating): near-universal — site address,
///   tenure, areas, the valuation basis. One keyword hit is worth chasing.
/// - **Conditional** (`min_keywords: 2`, optional deal-type/asset gating):
///   legitimately absent in large classes of packs — principals are rarely
///   named (the sponsor *company* is), exits are implicit, equity structure
///   is confidential, planning history and guarantees exist only when the
///   pack documents them. A single weak keyword ("exit", "equity", "lease")
///   matches almost any corpus; two distinct keywords mean the pack really
///   discusses the topic.
///
/// `deal_kinds` restricts a spec to `classification.project_type.category`
/// values (empty = all); `unless_asset_classes` skips the spec when
/// `classification.asset_class_primary` contains a listed substring (e.g. a
/// residential unit schedule on an industrial open-storage pack).
struct FieldSpec {
    path: &'static str,
    keywords: &'static [&'static str],
    hint: &'static str,
    min_keywords: usize,
    deal_kinds: &'static [&'static str],
    unless_asset_classes: &'static [&'static str],
}

const fn core(
    path: &'static str,
    keywords: &'static [&'static str],
    hint: &'static str,
) -> FieldSpec {
    FieldSpec {
        path,
        keywords,
        hint,
        min_keywords: 1,
        deal_kinds: &[],
        unless_asset_classes: &[],
    }
}

const fn conditional(
    path: &'static str,
    keywords: &'static [&'static str],
    hint: &'static str,
    deal_kinds: &'static [&'static str],
    unless_asset_classes: &'static [&'static str],
) -> FieldSpec {
    FieldSpec {
        path,
        keywords,
        hint,
        min_keywords: 2,
        deal_kinds,
        unless_asset_classes,
    }
}

const DEV: &[&str] = &["development"];
const NON_RESI_CLASSES: &[&str] = &[
    "industrial",
    "logistics",
    "warehouse",
    "office",
    "retail",
    "self-storage",
    "open storage",
    "storage",
    "data centre",
    "data center",
    "leisure",
    "hotel",
];
const RESI_CLASSES: &[&str] = &["residential", "student", "pbsa", "btr", "co-living", "care"];

const FIELD_SPECS: &[FieldSpec] = &[
    conditional("details.sponsor_principals",
        &["director", "shareholder", "principal", "shareholding", "owned by"],
        "every named principal/director/shareholder with a per-person track_record list",
        &[], &[]),
    conditional("details.sponsor_track_record",
        &["track record", "previous projects", "delivered", "completed schemes", "experience delivering"],
        "the sponsor's delivered-scheme track record",
        &[], &[]),
    // A residential unit schedule exists only on resi/mixed schemes — on
    // industrial/storage deals "units" meant lettable storage space and the
    // field was chased on every run (International House IOS).
    FieldSpec {
        path: "details.residential_units",
        keywords: &["unit mix", "bedroom", "apartments", "dwellings", "flat", "studio"],
        hint: "the per-type unit schedule (bedrooms, quantity, area, GDV)",
        min_keywords: 1,
        deal_kinds: &[],
        unless_asset_classes: NON_RESI_CLASSES,
    },
    core("details.gross_internal_area", &["gia", "sq ft", "sqft", "sq m"], "gross internal area as { value, unit }"),
    core("details.net_internal_area", &["nia", "net internal"], "net internal area as { value, unit }"),
    // Development packs quote site/purchase/residual values — "market value"
    // is a standing-investment concept and needs real evidence to chase.
    conditional("details.market_value",
        &["market value", "day-one value", "existing value", "current value", "existing use value", "open market value"],
        "the current market value (day-1 / existing value), distinct from GDV",
        &[], &[]),
    FieldSpec {
        path: "details.gross_development_value",
        keywords: &["gdv", "gross development value"],
        hint: "the scheme GDV",
        min_keywords: 1,
        deal_kinds: DEV,
        unless_asset_classes: &[],
    },
    FieldSpec {
        path: "details.total_development_cost",
        keywords: &["total cost", "development cost", "build cost", "construction cost", "tdc"],
        hint: "total development cost",
        min_keywords: 1,
        deal_kinds: DEV,
        unless_asset_classes: &[],
    },
    core("details.address", &["street", "postcode"], "the site address incl. line1, city, postcode"),
    // Most packs never state an exit strategy — it is implied (unit sales,
    // refinance). Generic "exit"/"sale" matched every corpus.
    conditional("details.exit_plan",
        &["exit strategy", "exit route", "exit via", "disposal strategy", "forward sale", "sell-down", "sale programme", "unit sales"],
        "the asset-level exit (unit sales / disposal), not facility repayment",
        &[], &[]),
    core("details.tenure", &["freehold", "leasehold", "tenure"], "tenure type and length"),
    // Acquisitions state a price; refinances and already-owned sites do not.
    conditional("details.purchase_price",
        &["purchase price", "asking price", "guide price", "offers in excess", "acquisition price"],
        "purchase price / asking price",
        &[], &[]),
    FieldSpec {
        path: "details.planning_status",
        keywords: &["planning", "consent", "prior approval"],
        hint: "planning status description",
        min_keywords: 1,
        deal_kinds: DEV,
        unless_asset_classes: &[],
    },
    // Pure resi packs mention tenants/leases only incidentally (ASTs) — an
    // occupational tenancy schedule needs two distinct tenancy signals.
    conditional("details.commercial_units",
        &["tenant", "lease", "occupier", "tenancy", "letting", "rent passing", "rental income"],
        "the commercial tenancy schedule",
        &[], RESI_CLASSES),
    // Equity structure is confidential or implied in most packs — the
    // resolver derives implied equity downstream; only chase real quotes.
    conditional("details.developer_equity",
        &["developer equity", "sponsor equity", "equity contribution", "equity injection", "cash equity"],
        "sponsor/developer equity contribution",
        &[], &[]),
    conditional("canonical_deal.planning_applications",
        &["planning application", "application ref", "planning ref", "lpa ref", "application number", "planning consent"],
        "every planning application referenced (ref, status, local authority)",
        DEV, &[]),
    conditional("canonical_deal.phases",
        &["phase 1", "phase one", "phase 2", "phase two", "phases", "phased"],
        "scheme phases with unit_count and status",
        &[], &[]),
    conditional("canonical_deal.facility_parties",
        &["lender", "borrower", "facility agent", "security trustee"],
        "borrower and lender parties per facility",
        &[], &[]),
    core("canonical_deal.valuations", &["valuation", "valuer", "valued at"], "valuation entries (value, basis, valuer, date)"),
    conditional("canonical_deal.guarantees",
        &["guarantor", "personal guarantee", "corporate guarantee", "performance bond", "letter of credit", "indemnity in favour"],
        "stated guarantees (guarantor, beneficiary, cap, trigger)",
        &[], &[]),
    conditional("canonical_deal.security",
        &["first legal charge", "legal charge", "debenture", "security package", "fixed charge", "charge over"],
        "the security package (type, ranking, secured asset)",
        &[], &[]),
    // Finance-cost totals live on appraisals/term sheets only.
    conditional("details.finance_costs",
        &["finance cost", "rolled interest", "interest over the term", "total interest"],
        "stated finance/interest total over the loan term (e.g. \"Finance costs £749,659\")",
        &[], &[]),
    conditional("details.loan_including_interest",
        &["incl. interest", "including interest", "gross loan", "all-in loan"],
        "the all-in loan figure incl. rolled interest (e.g. \"loan including interest £5,699,659\")",
        &[], &[]),
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
        cur.get("source").is_some()
            || cur.get("source_type").is_some()
            || cur.get("evidence_text").is_some()
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
    let project_type = doc
        .pointer("/classification/project_type/category")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    let asset_class = doc
        .pointer("/classification/asset_class_primary")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    for spec in FIELD_SPECS {
        if issues.len() >= MAX_ISSUES {
            break;
        }
        // Deal-type gating: GDV/TDC/planning fields only exist on
        // development deals; asset-class gating: a resi unit schedule does
        // not exist on an industrial pack. An unknown/absent classification
        // never suppresses a spec.
        if !spec.deal_kinds.is_empty()
            && !project_type.is_empty()
            && !spec.deal_kinds.iter().any(|k| project_type.contains(k))
        {
            continue;
        }
        if spec
            .unless_asset_classes
            .iter()
            .any(|c| asset_class.contains(c))
        {
            continue;
        }
        if !field_is_empty(doc, spec.path) {
            continue;
        }
        // Evidence = keyword hits; keep the strongest snippets inline so the
        // fill call can extract without needing document tools. Conditional
        // fields need `min_keywords` DISTINCT keywords to match — a single
        // generic term ("exit", "lease") appears in nearly every pack and
        // was firing a gap on fields the source never carried.
        let mut hits = Vec::new();
        let mut doc_names: Vec<String> = Vec::new();
        let mut keywords_matched = 0usize;
        for kw in spec.keywords {
            let kw_hits = corpus.search(kw, 3);
            if !kw_hits.is_empty() {
                keywords_matched += 1;
            }
            for hit in kw_hits {
                if !doc_names.contains(&hit.doc_name) {
                    doc_names.push(hit.doc_name.clone());
                }
                hits.push(format!(
                    "{} ({}): {}",
                    hit.doc_name,
                    hit.page_label,
                    hit.snippet.trim()
                ));
            }
            if keywords_matched >= spec.min_keywords && hits.len() >= 4 {
                break;
            }
        }
        // The bar: `min_keywords` distinct keywords, OR (for conditional
        // specs) a distinctive phrase repeated — "exit strategy" appearing
        // three times is evidence even if no other keyword matches.
        if doc_names.is_empty() || (keywords_matched < spec.min_keywords && hits.len() < 3) {
            continue;
        }
        issues.push(Issue {
            code: "missing_with_evidence",
            path: spec.path.to_string(),
            message: format!(
                "{} — extract {}. Evidence:\n{}",
                spec.path,
                spec.hint,
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

    #[test]
    fn conditional_field_needs_two_keywords_or_a_repeated_phrase() {
        // Generic single-keyword mention — "equity" language alone must not
        // chase developer_equity on a pack that never states it.
        let doc = json!({ "details": {}, "classification": { "project_type": { "category": "development" } } });
        let weak = corpus_with("the investor equity tranche is priced to yield");
        assert!(!coverage_issues(&doc, &weak)
            .iter()
            .any(|i| i.path == "details.developer_equity"));
        // Two distinct keywords = the pack really discusses it.
        let strong = corpus_with("developer equity contribution funds the equity injection");
        assert!(coverage_issues(&doc, &strong)
            .iter()
            .any(|i| i.path == "details.developer_equity"));
    }

    #[test]
    fn asset_class_gate_skips_residential_units_on_industrial_deals() {
        let doc = json!({
            "details": {},
            "classification": { "asset_class_primary": "industrial open storage" },
        });
        let c = corpus_with("the unit mix includes studio apartments and one-bed flats");
        assert!(!coverage_issues(&doc, &c)
            .iter()
            .any(|i| i.path == "details.residential_units"));
    }

    #[test]
    fn deal_kind_gate_skips_gdv_on_non_development_deals() {
        let doc = json!({
            "details": {},
            "classification": { "project_type": { "category": "hold" } },
        });
        let c = corpus_with("GDV commentary and gross development value estimates");
        assert!(!coverage_issues(&doc, &c)
            .iter()
            .any(|i| i.path == "details.gross_development_value"));
    }
}

const PATCH_PROMPT: &str = "You are filling specific gaps in an extracted deal document. For each field below, the extraction left it empty but the quoted evidence shows the value. Return a JSON object mapping each path to its extracted value.\n\nSHAPE RULES:\n- paths under `details.` → a WithSource object   `{\"value\": <value>, \"source\": \"<document name>\", \"source_type\": \"extracted\", \"evidence_text\": \"<short quote>\", \"notes\": null, \"scope\": null}`\n- paths under `canonical_deal.` → the raw value only (array or object), matching   the shapes used elsewhere in the document\n- extract only what the evidence supports; do not guess or round to invented precision\n- omit a path entirely if the evidence does not answer it\n\nFIELDS TO FILL:\n";

/// Ask the coverage model for a `{path: value}` patch and merge the paths
/// that don't introduce validation errors. Returns the merged paths.
pub async fn fill_missing(
    ex: &crate::extract::Extractor,
    doc: &mut Value,
    issues: &[Issue],
) -> Result<(Vec<String>, Vec<String>, Vec<String>), String> {
    if issues.is_empty() {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }
    let prompt = format!(
        "{PATCH_PROMPT}{}\n\nDOCUMENT (for shape reference):\n{}",
        issues
            .iter()
            .map(|i| i.message.as_str())
            .collect::<Vec<_>>()
            .join("\n\n"),
        serde_json::to_string(doc).map_err(|e| e.to_string())?,
    );
    let text = ex.call_text(&prompt).await?;
    let patch: Value = crate::extract::parse_json(&text)?;
    let obj = patch.as_object().ok_or("coverage patch is not an object")?;

    // Default evidence source per path — the document where the keyword hit.
    let source_for = |path: &str| -> Option<String> {
        issues.iter().find(|i| i.path == path).and_then(|i| {
            i.message
                .lines()
                .find(|l| l.contains(": ") && l.contains("("))
                .map(|l| l.split(" (").next().unwrap_or("").trim().to_string())
                .filter(|n| !n.is_empty())
        })
    };
    Ok(merge_patch_guarded(doc, obj, source_for, false))
}

/// Merge a `{path: value}` patch one path at a time, rolling back any path
/// that introduces validation errors. With `only_empty`, paths whose current
/// value is populated are rejected — fills may add, never overwrite.
/// Returns (merged, rejected, rejection_reasons) where each reason is
/// "path: code message" for the first new issue the insert raised — the
/// model routinely omits nullable required keys, so the value is conformed
/// BEFORE validation or every fill dies on missing-null keys
/// (facility_parties / planning_applications rejected on every run).
pub(crate) fn merge_patch_guarded(
    doc: &mut Value,
    obj: &serde_json::Map<String, Value>,
    source_for: impl Fn(&str) -> Option<String>,
    only_empty: bool,
) -> (Vec<String>, Vec<String>, Vec<String>) {
    let baseline_errors: std::collections::HashSet<String> = crate::validate::validate(doc)
        .iter()
        .map(|i| i.path.clone())
        .collect();
    let mut merged = Vec::new();
    let mut rejected = Vec::new();
    let mut reasons = Vec::new();
    for (path, value) in obj {
        if value.is_null() || (only_empty && !path_is_empty(doc, path)) {
            rejected.push(path.clone());
            continue;
        }
        let snapshot = doc.clone();
        // details.* values are WithSource — a missing source would break
        // provenance, so default it to the evidence document.
        let mut v = value.clone();
        if path.starts_with("details.") {
            if let Some(o) = v.as_object_mut() {
                let src_empty = o
                    .get("source")
                    .and_then(|s| s.as_str())
                    .map_or(true, |s| s.trim().is_empty());
                if src_empty {
                    if let Some(name) = source_for(path) {
                        o.insert("source".to_string(), Value::String(name));
                    }
                }
            }
        }
        set_path(doc, path, v);
        // Conform the inserted value BEFORE judging it — schema `required`
        // lists nullable keys the model omits, and judging the raw insert
        // rejects every facility_parties / planning_applications fill.
        crate::conform::conform_document(doc);
        let new_issues = crate::validate::validate(doc);
        let bad = new_issues
            .iter()
            .any(|i| i.path.starts_with(path.as_str()) && !baseline_errors.contains(&i.path));
        let worse = new_issues
            .iter()
            .filter(|i| !baseline_errors.contains(&i.path))
            .count()
            > 6;
        if bad || worse {
            *doc = snapshot;
            rejected.push(path.clone());
            if let Some(issue) = new_issues
                .iter()
                .find(|i| i.path.starts_with(path.as_str()) && !baseline_errors.contains(&i.path))
                .or_else(|| {
                    new_issues
                        .iter()
                        .find(|i| !baseline_errors.contains(&i.path))
                })
            {
                reasons.push(format!("{path}: {} {}", issue.code, issue.message));
            }
        } else {
            merged.push(path.clone());
        }
    }
    if !rejected.is_empty() {
        eprintln!("[coverage] rejected patch paths: {}", rejected.join(", "));
    }
    (merged, rejected, reasons)
}

/// Like `field_is_empty`, but follows `[i]` array indices; an index past the
/// end of an array counts as empty (it is an append).
pub(crate) fn path_is_empty(doc: &Value, path: &str) -> bool {
    let mut cur = doc;
    for seg in path.split('.') {
        let (key, idx) = match seg.split_once('[') {
            Some((k, i)) => (k, i.trim_end_matches(']').parse::<usize>().ok()),
            None => (seg, None),
        };
        if !key.is_empty() {
            match cur.get(key) {
                Some(v) => cur = v,
                None => return true,
            }
        }
        if let Some(i) = idx {
            match cur.get(i) {
                Some(v) => cur = v,
                None => return true,
            }
        }
    }
    if let Some(inner) = cur.get("value").filter(|_| {
        cur.get("source").is_some()
            || cur.get("source_type").is_some()
            || cur.get("evidence_text").is_some()
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

/// Set `a.b.c = value`, creating intermediate objects as needed.
/// Supports array indices (`facilities[2].commitment.amount`); setting an
/// array element to null splices it out (the model's way to delete entries).
pub(crate) fn set_path(doc: &mut Value, path: &str, value: Value) {
    let segs: Vec<&str> = path.split('.').collect();
    set_path_at(doc, &segs, value);
}

fn set_path_at(node: &mut Value, segs: &[&str], value: Value) {
    if segs.is_empty() {
        *node = value;
        return;
    }
    let (key, idx) = match segs[0].split_once('[') {
        Some((k, i)) => (k, i.trim_end_matches(']').parse::<usize>().ok()),
        None => (segs[0], None),
    };
    let child = if key.is_empty() {
        Some(node)
    } else {
        if !node.is_object() {
            *node = Value::Object(serde_json::Map::new());
        }
        node.as_object_mut().map(|o| {
            o.entry(key.to_string()).or_insert_with(|| {
                if idx.is_some() {
                    Value::Array(Vec::new())
                } else {
                    Value::Object(serde_json::Map::new())
                }
            })
        })
    };
    let Some(child) = child else { return };
    match idx {
        Some(i) => {
            if !child.is_array() {
                *child = Value::Array(Vec::new());
            }
            let arr = child.as_array_mut().unwrap();
            while arr.len() <= i {
                arr.push(Value::Null);
            }
            if segs.len() == 1 && value.is_null() {
                arr.remove(i); // delete element
            } else if segs.len() == 1 {
                arr[i] = value;
            } else {
                if arr[i].is_null() {
                    arr[i] = Value::Object(serde_json::Map::new());
                }
                let slot = &mut arr[i];
                set_path_at(slot, &segs[1..], value);
            }
        }
        None => {
            if segs.len() == 1 {
                *child = value;
            } else {
                set_path_at(child, &segs[1..], value);
            }
        }
    }
}

#[cfg(test)]
mod extra_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn planning_application_missing_nullable_keys_merges() {
        // Prod order: conform runs before merge, so baseline issues are
        // pre-existing and only the inserted path is judged.
        let mut doc = json!({
            "details": {},
            "canonical_deal": { "planning_applications": [], "facilities": [] },
        });
        crate::conform::conform_document(&mut doc);
        let obj = serde_json::Map::from_iter([(
            "canonical_deal.planning_applications[0]".to_string(),
            json!({"id":"pa_1","reference":"LUT/2024/1234","description":"183 dwellings","status":"granted"}),
        )]);
        let (merged, rejected, reasons) = merge_patch_guarded(&mut doc, &obj, |_| None, false);
        assert!(
            rejected.is_empty(),
            "rejected={rejected:?} reasons={reasons:?}"
        );
        assert!(merged.contains(&"canonical_deal.planning_applications[0]".to_string()));
    }
}
