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
/// verbatim in the text. Stated headline debt figures (loan_amount,
/// senior_debt, mezzanine_debt, developer_equity) ARE checked — they are
/// stated in the pack and are the most common misquote site.
const SKIP_PREFIXES: &[&str] = &["computed.", "gross_debt", "net_debt", "loan_to_value", "loan_to_gdv", "loan_to_cost", "loan_including_interest"];

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
        let with_commas: String = with_commas.chars().rev().collect();
        v.push(with_commas);
        // millions shorthand with sane precision: "35.81m", "35.8m", "35m"
        let m = amount / 1_000_000.0;
        for n in [format!("{:.2}", m), format!("{:.1}", m), format!("{:.0}", m)] {
            for cand in [n.clone(), format!("{}m", n)] {
                if !v.contains(&cand) {
                    v.push(cand);
                }
            }
        }
    }
    // percentage forms: 4.5, 4.5%, 0.045 — both 1dp and 2dp renderings.
    if amount.abs() > 0.0 && amount.abs() < 1.0 {
        let pct = amount * 100.0;
        v.push(format!("{:.2}%", pct));
        v.push(format!("{:.1}%", pct));
        v.push(format!("{:.2}", pct));
        v.push(format!("{:.1}", pct));
    }
    v
}

/// Boundary-aware substring check: "4500000" must not match inside
/// "14500000", and "4.5" must not match inside "14.5" or "4.5m" handled by
/// the variant list itself.
fn contains_figure(corpus_text: &str, amount: f64) -> bool {
    let lc = corpus_text.to_lowercase();
    figure_variants(amount).iter().any(|v| {
        if v.is_empty() {
            return false;
        }
        let needle = v.to_lowercase();
        let mut search_from = 0usize;
        while let Some(pos) = lc[search_from..].find(&needle).map(|p| p + search_from) {
            let before_ok = pos == 0
                || !lc.as_bytes()[pos - 1].is_ascii_digit() && lc.as_bytes()[pos - 1] != b'.';
            let end = pos + needle.len();
            let after_ok = end >= lc.len()
                || (!lc.as_bytes()[end].is_ascii_digit() && lc.as_bytes()[end] != b'.')
                // Allow a % or m suffix directly after the figure.
                || matches!(lc.as_bytes()[end], b'%' | b'm');
            if before_ok && after_ok {
                return true;
            }
            search_from = end;
        }
        false
    })
}

/// Collect {path, amount, kind} for every ground-checkable field.
/// Checks details.* money/percent fields AND canonical facility commitments
/// (decimal-string amounts — the most consequential figures in the output).
fn collect_checkable(document: &Value) -> Vec<(String, f64, &'static str)> {
    let mut out = vec![];
    if let Some(details) = document.get("details").and_then(Value::as_object) {
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
    }
    // Canonical facility commitments — amounts are decimal strings.
    if let Some(facilities) = document
        .get("canonical_deal")
        .and_then(|c| c.get("facilities"))
        .and_then(Value::as_array)
    {
        for (i, fac) in facilities.iter().enumerate() {
            let name = fac.get("name").and_then(Value::as_str).unwrap_or("facility");
            if let Some(amount) = fac
                .get("commitment")
                .and_then(|c| c.get("amount"))
                .and_then(|a| a.as_f64().or_else(|| a.as_str().and_then(|s| s.parse::<f64>().ok())))
            {
                if amount.abs() >= 1.0 {
                    out.push((format!("canonical_deal.facilities[{i}].commitment ({name})"), amount, "money"));
                }
            }
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
            mismatches.push(format!("{} = {:?} ({})", path, amount, kind));
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
         Use paths like details.stabilised_yield.value for detail fields, or canonical_deal.facilities[0].commitment for facility commitments. Only include fields you are CONFIDENT are wrong.",
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
            // Facility commitment fix: canonical_deal.facilities[N].commitment
            if let Some(rest) = path.strip_prefix("canonical_deal.facilities[") {
                let Some(idx_str) = rest.split(']').next() else { continue };
                let Ok(idx) = idx_str.parse::<usize>() else { continue };
                let Some(fac) = document
                    .get_mut("canonical_deal")
                    .and_then(|c| c.get_mut("facilities"))
                    .and_then(Value::as_array_mut)
                    .and_then(|f| f.get_mut(idx))
                else { continue };
                // Commitment amount is a decimal string — coerce the patch.
                let as_str = match new_val {
                    Value::String(s) => s.clone(),
                    other => match other.as_f64() {
                        Some(n) => {
                            if n.fract() == 0.0 { format!("{}", n as i64) } else { format!("{}", n) }
                        }
                        None => continue,
                    },
                };
                if let Some(commitment) = fac.get_mut("commitment").and_then(Value::as_object_mut) {
                    commitment.insert("amount".to_string(), Value::String(as_str));
                    applied += 1;
                }
                continue;
            }
            // Detail field fix: details.<key>.value
            let Some(key) = path.strip_prefix("details.").and_then(|p| p.strip_suffix(".value")).map(str::to_string)
                .or_else(|| path.strip_prefix("details.").map(str::to_string))
            else { continue };
            if let Some(field) = document.get_mut("details").and_then(|d| d.get_mut(&key)) {
                if let Some(obj) = field.as_object_mut() {
                    // Coerce the patch to the field's existing shape — keep
                    // money objects' {amount,currency} and scalar numbers as
                    // numbers rather than swapping types.
                    if let Some(existing) = obj.get("value") {
                        let coerced = match existing {
                            Value::Object(o) if o.contains_key("amount") => {
                                let amt = match new_val {
                                    Value::Number(_) => new_val.clone(),
                                    Value::String(s) => s.parse::<f64>().ok().map(|n| Value::from(n)).unwrap_or(new_val.clone()),
                                    _ => new_val.clone(),
                                };
                                let mut money = o.clone();
                                money.insert("amount".to_string(), amt);
                                Value::Object(money)
                            }
                            Value::Number(_) => match new_val {
                                Value::String(s) => s.parse::<f64>().ok().map(Value::from).unwrap_or(new_val.clone()),
                                _ => new_val.clone(),
                            },
                            _ => new_val.clone(),
                        };
                        obj.insert("value".to_string(), coerced);
                    } else {
                        obj.insert("value".to_string(), new_val.clone());
                    }
                    obj.insert("notes".to_string(), Value::String(format!("verified — corrected by verify pass (was extracted differently in source)")));
                    applied += 1;
                }
            }
        }
    }
    (checkable.len(), applied)
}
