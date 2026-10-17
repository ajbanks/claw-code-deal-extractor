//! Cheap post-extraction verification: ground-check extracted numbers
//! against the corpus text, then one cheap LLM call per mismatch to fix.
//!
//! Deterministic pass — no model calls on the happy path. Every money /
//! percentage value extracted is checked that its figure appears (in any
//! formatting) in the source text. Mismatches are the extraction-misread
//! class: right field, wrong cell (e.g. cap rate read as 0.04 vs the
//! sheet's stated 0.045).

use serde_json::Value;
use crate::corpus::Corpus;
use crate::extract::Extractor;

/// Fields worth ground-checking: money amounts and percentages on details.
/// Path prefixes to skip — computed/derived fields that won't appear
/// verbatim in the text.
const SKIP_PREFIXES: &[&str] = &["computed.", "loan_amount", "gross_debt", "senior_debt", "mezzanine_debt", "developer_equity", "loan_to_value", "loan_to_gdv"];

/// A number string that appears in the corpus under *any* formatting:
/// "4500000", "4,500,000", "4.5m", "£4.5m", "4.5" (for percentages).
fn figure_variants(amount: f64) -> Vec<String> {
    let mut v = vec![];
    // raw value
    v.push(format!("{}", amount));
    // comma-separated absolute
    if amount.abs() >= 1_000.0 {
        let s = format!("{}", amount.abs() as i64);
        let mut with_commas = String::new();
        for (i, c) in s.chars().rev().enumerate() {
            if i > 0 && i % 3 == 0 { with_commas.push(','); }
            with_commas.push(c);
        }
        v.push(with_commas.chars().rev().collect());
        // millions shorthand: "4.5m", "4.5"
        v.push(format!("{}m", amount / 1_000_000.0));
    }
    // percentage forms: 4.5, 4.5%, 0.045
    if amount.abs() > 0.0 && amount.abs() < 1.0 {
        v.push(format!("{:.1}%", amount * 100.0));
        v.push(format!("{}", amount * 100.0));
    }
    v
}

fn contains_figure(corpus_text: &str, amount: f64) -> bool {
    let lc = corpus_text.to_lowercase();
    figure_variants(amount).iter().any(|v| !v.is_empty() && lc.contains(&v.to_lowercase()))
}

/// Collect {path, amount, kind} for every ground-checkable field.
fn collect_checkable(document: &Value) -> Vec<(String, f64, &'static str)> {
    let mut out = vec![];
    let details = match document.get("details").and_then(Value::as_object) {
        Some(d) => d,
        None => return out,
    };
    for (key, field) in details {
        if SKIP_PREFIXES.iter().any(|p| key.starts_with(p)) { continue; }
        let Some(obj) = field.as_object() else { continue };
        let Some(val) = obj.get("value") else { continue; };
        // money { amount, currency } or bare percentage
        if let Some(amount) = val.get("amount").and_then(Value::as_f64) {
            if amount.abs() >= 1.0 { out.push((key.clone(), amount, "money")); }
        } else if let Some(pct) = val.as_f64() {
            if pct > 0.0 && pct < 15.0 { out.push((key.clone(), pct, "percent")); }
        }
    }
    out
}

/// Run the verification pass. Returns (document, issues_checked, fixes_applied).
pub async fn verify_extracted_numbers(
    document: &mut Value,
    corpus: &Corpus,
    cheap_model: &Extractor,
) -> (usize, usize) {
    let corpus_text = corpus.render();
    let checkable = collect_checkable(document);
    if checkable.is_empty() { return (0, 0); }

    // Cheap pass 1: which figures are missing from the corpus text?
    let mut mismatches: Vec<String> = vec![];
    for (path, amount, kind) in &checkable {
        if !contains_figure(&corpus_text, *amount) {
            mismatches.push(format!("details.{} = {:?} ({})", path, amount, kind));
        }
    }
    if mismatches.is_empty() {
        return (checkable.len(), 0);
    }

    // Cheap pass 2: one LLM call to re-check the flagged fields.
    let prompt = format!(
        "Verify these extracted values against the source documents. For each, report whether the extracted value is CORRECT or the source shows a DIFFERENT figure. Only fix genuine misreads (wrong cell / rounding / scope), not formatting.\n\n\
         Fields flagged as not found in source text:\n{}\n\n\
         Return JSON patch of corrections only: {{\"fixes\": {{\"<json.path>\": <corrected value or null>}}}}\n\
         Use paths like details.stabilised_yield.value. Only include fields you are CONFIDENT are wrong.",
        mismatches.join("\n"),
    );

    let text = match cheap_model.call(&prompt, &corpus.render()).await {
        Ok(t) => t,
        Err(_) => return (checkable.len(), 0),
    };
    let fixes = match crate::extract::parse_json(&text) {
        Ok(v) => v,
        Err(_) => return (checkable.len(), 0),
    };

    let mut applied = 0usize;
    if let Some(fix_obj) = fixes.get("fixes").and_then(Value::as_object) {
        for (path, new_val) in fix_obj {
            // path form: details.<key>.value
            let Some(key) = path.strip_prefix("details.").and_then(|p| p.strip_suffix(".value")).map(str::to_string)
                .or_else(|| path.strip_prefix("details.").map(str::to_string))
            else { continue };
            if let Some(field) = document.get_mut("details").and_then(|d| d.get_mut(&key)) {
                if let Some(obj) = field.as_object_mut() {
                    obj.insert("value".to_string(), new_val.clone());
                    obj.insert("notes".to_string(), Value::String(format!("verified — corrected by verify pass (was extracted differently in source)")));
                    applied += 1;
                }
            }
        }
    }
    (checkable.len(), applied)
}
