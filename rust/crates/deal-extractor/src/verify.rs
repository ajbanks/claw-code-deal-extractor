//! Cheap post-extraction verification: ground-check extracted numbers
//! against the corpus text, then one cheap LLM call per mismatch to fix.
//!
//! Deterministic pass — no model calls on the happy path. Every money /
//! percentage value extracted is checked that its figure appears (in any
//! formatting) in the source text. Mismatches are the extraction-misread
//! class: right field, wrong cell (e.g. cap rate read as 0.04 vs the
//! sheet's stated 0.045).

use crate::corpus::Corpus;
use crate::extract::Extractor;
use crate::recall;
use serde_json::Value;

/// Fields worth ground-checking: money amounts and percentages on details.
/// Path prefixes to skip — computed/derived fields that won't appear
/// verbatim in the text. Stated headline debt figures (loan_amount,
/// senior_debt, mezzanine_debt, developer_equity) ARE checked — they are
/// stated in the pack and are the most common misquote site.
const SKIP_PREFIXES: &[&str] = &[
    "computed.",
    "gross_debt",
    "net_debt",
    "loan_to_value",
    "loan_to_gdv",
    "loan_to_cost",
    "loan_including_interest",
];

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
            if i > 0 && i % 3 == 0 {
                with_commas.push(',');
            }
            with_commas.push(c);
        }
        let with_commas: String = with_commas.chars().rev().collect();
        v.push(with_commas);
        // millions shorthand with sane precision: "35.81m", "35.8m", "35m"
        let m = amount / 1_000_000.0;
        for n in [
            format!("{:.2}", m),
            format!("{:.1}", m),
            format!("{:.0}", m),
        ] {
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
            if SKIP_PREFIXES.iter().any(|p| key.starts_with(p)) {
                continue;
            }
            let Some(obj) = field.as_object() else {
                continue;
            };
            let Some(val) = obj.get("value") else {
                continue;
            };
            // money { amount, currency } or bare percentage
            if let Some(amount) = val.get("amount").and_then(Value::as_f64) {
                if amount.abs() >= 1.0 {
                    out.push((key.clone(), amount, "money"));
                }
            } else if let Some(pct) = val.as_f64() {
                if pct > 0.0 && pct < 15.0 {
                    out.push((key.clone(), pct, "percent"));
                }
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
            let name = fac
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("facility");
            if let Some(amount) = fac
                .get("commitment")
                .and_then(|c| c.get("amount"))
                .and_then(|a| {
                    a.as_f64()
                        .or_else(|| a.as_str().and_then(|s| s.parse::<f64>().ok()))
                })
            {
                if amount.abs() >= 1.0 {
                    out.push((
                        format!("canonical_deal.facilities[{i}].commitment ({name})"),
                        amount,
                        "money",
                    ));
                }
            }
        }
    }
    out
}

/// Check 2: a field whose `value` disagrees with a figure quoted inside its
/// own `evidence_text`. This catches "right evidence, wrong value" — e.g.
/// stabilised_yield extracted as the implied 0.0416 while the evidence text
/// quotes the sheet's stated `Cap Rate 0.045`.
///
/// Only same-kind comparisons, and only when the evidence quotes at least
/// one comparable figure: money {amount} fields vs explicit money figures;
/// percent fields vs percent figures and sub-1 fractions (×100, matching
/// both representations since values may be stored as 4.5 or 0.045).
fn evidence_mismatches(document: &Value) -> Vec<String> {
    let mut out = vec![];
    let Some(details) = document.get("details").and_then(Value::as_object) else {
        return out;
    };
    for (key, field) in details {
        if SKIP_PREFIXES.iter().any(|p| key.starts_with(p)) {
            continue;
        }
        let Some(obj) = field.as_object() else {
            continue;
        };
        let Some(val) = obj.get("value") else {
            continue;
        };
        let Some(ev) = obj.get("evidence_text").and_then(Value::as_str) else {
            continue;
        };
        let figures = recall::parse_figures(ev);
        if let Some(amount) = val.get("amount").and_then(Value::as_f64) {
            if amount.abs() < 1.0 {
                continue;
            }
            let cands: Vec<_> = figures
                .iter()
                .filter(|p| p.kind == recall::FigureKind::Money && p.explicit_money)
                .collect();
            if !cands.is_empty()
                && !cands
                    .iter()
                    .any(|p| (p.value - amount).abs() <= p.tol.max(amount.abs() * 0.002))
            {
                let raws: Vec<_> = cands.iter().map(|p| &ev[p.start..p.end]).collect();
                out.push(format!(
                    "details.{key}.value = {amount:?} but its evidence text states {raws:?}"
                ));
            }
        } else if let Some(pct) = val.as_f64() {
            if pct <= 0.0 || pct >= 15.0 {
                continue;
            }
            // Candidates in every representation: 4.5 (pct) and 0.045 (fraction).
            let cands: Vec<f64> = figures
                .iter()
                .filter(|p| {
                    p.kind == recall::FigureKind::Percent
                        || (p.kind == recall::FigureKind::Money && p.value > 0.0 && p.value < 1.0)
                })
                .flat_map(|p| [p.value, p.value * 100.0, p.value / 100.0])
                .collect();
            if !cands.is_empty()
                && !cands
                    .iter()
                    .any(|c| (c - pct).abs() <= (pct * 0.002).max(0.001))
            {
                let raws: Vec<_> = figures
                    .iter()
                    .filter(|p| {
                        p.kind == recall::FigureKind::Percent
                            || (p.kind == recall::FigureKind::Money
                                && p.value > 0.0
                                && p.value < 1.0)
                    })
                    .map(|p| &ev[p.start..p.end])
                    .collect();
                out.push(format!(
                    "details.{key}.value = {pct:?} but its evidence text states {raws:?}"
                ));
            }
        }
    }
    out
}

/// Run the verification pass. Returns (issues_checked, fixes_applied, flagged).
pub async fn verify_extracted_numbers(
    document: &mut Value,
    corpus: &Corpus,
    cheap_model: &Extractor,
) -> (usize, usize, Vec<String>) {
    let corpus_text = corpus.render();
    let checkable = collect_checkable(document);

    // Cheap pass 1: which figures are missing from the corpus text? Plus:
    // does any field's value contradict a figure in its own evidence text?
    let mut mismatches = evidence_mismatches(document);
    for (path, amount, kind) in &checkable {
        if !contains_figure(&corpus_text, *amount) {
            mismatches.push(format!("{} = {:?} ({})", path, amount, kind));
        }
    }
    if mismatches.is_empty() {
        return (checkable.len(), 0, vec![]);
    }
    let flagged = mismatches.clone();

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
        Err(_) => return (checkable.len(), 0, flagged),
    };
    let fixes = match crate::extract::parse_json(&text) {
        Ok(v) => v,
        Err(_) => return (checkable.len(), 0, flagged),
    };

    let mut applied = 0usize;
    if let Some(fix_obj) = fixes.get("fixes").and_then(Value::as_object) {
        for (path, new_val) in fix_obj {
            if apply_fix(document, path, new_val) {
                applied += 1;
            }
        }
    }
    (checkable.len(), applied, flagged)
}

/// Apply one correction patch at `path`, coercing the value to the target
/// field's existing representation. Returns true if the patch was applied.
fn apply_fix(document: &mut Value, path: &str, new_val: &Value) -> bool {
    // Facility commitment fix: canonical_deal.facilities[N].commitment
    if let Some(rest) = path.strip_prefix("canonical_deal.facilities[") {
        let Some(idx_str) = rest.split(']').next() else {
            return false;
        };
        let Ok(idx) = idx_str.parse::<usize>() else {
            return false;
        };
        let Some(fac) = document
            .get_mut("canonical_deal")
            .and_then(|c| c.get_mut("facilities"))
            .and_then(Value::as_array_mut)
            .and_then(|f| f.get_mut(idx))
        else {
            return false;
        };
        // Commitment amount is a decimal string — coerce the patch.
        let as_str = match new_val {
            Value::String(s) => s.clone(),
            other => match other.as_f64() {
                Some(n) => {
                    if n.fract() == 0.0 {
                        format!("{}", n as i64)
                    } else {
                        format!("{}", n)
                    }
                }
                None => return false,
            },
        };
        if let Some(commitment) = fac.get_mut("commitment").and_then(Value::as_object_mut) {
            commitment.insert("amount".to_string(), Value::String(as_str));
            return true;
        }
        return false;
    }
    // Detail field fix: details.<key>.value
    let Some(key) = path
        .strip_prefix("details.")
        .and_then(|p| p.strip_suffix(".value"))
        .map(str::to_string)
        .or_else(|| path.strip_prefix("details.").map(str::to_string))
    else {
        return false;
    };
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
                            Value::String(s) => s
                                .parse::<f64>()
                                .ok()
                                .map(Value::from)
                                .unwrap_or(new_val.clone()),
                            _ => new_val.clone(),
                        };
                        let mut money = o.clone();
                        money.insert("amount".to_string(), amt);
                        Value::Object(money)
                    }
                    Value::Number(_) => match new_val {
                        Value::String(s) => s
                            .parse::<f64>()
                            .ok()
                            .map(Value::from)
                            .unwrap_or(new_val.clone()),
                        _ => new_val.clone(),
                    },
                    _ => new_val.clone(),
                };
                obj.insert("value".to_string(), coerced);
            } else {
                obj.insert("value".to_string(), new_val.clone());
            }
            obj.insert(
                "notes".to_string(),
                Value::String(
                    "verified — corrected by verify pass (was extracted differently in source)"
                        .to_string(),
                ),
            );
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── figure_variants ──────────────────────────────────────────────

    #[test]
    fn millions_variants_have_sane_precision() {
        let v = figure_variants(35_810_099.96);
        for want in ["35.81m", "35.8m", "36m", "35,810,099"] {
            assert!(
                v.contains(&want.to_string()),
                "missing variant {want}: {v:?}"
            );
        }
        // Regression: earlier code produced "35.8099999m" style garbage.
        assert!(
            !v.iter().any(|s| s.contains("8099999")),
            "malformed m-variant: {v:?}"
        );

        let v = figure_variants(4_500_000.0);
        for want in ["4.5m", "4,500,000"] {
            assert!(
                v.contains(&want.to_string()),
                "missing variant {want}: {v:?}"
            );
        }
    }

    #[test]
    fn percent_variants_keep_two_decimals() {
        let v = figure_variants(0.0485);
        for want in ["4.85%", "4.85", "4.9%", "4.9"] {
            assert!(
                v.contains(&want.to_string()),
                "missing variant {want}: {v:?}"
            );
        }
    }

    // ── evidence_mismatches: value vs own evidence text ───────────────

    #[test]
    fn flags_percent_value_contradicting_evidence() {
        let doc = json!({
            "details": {
                "stabilised_yield": {
                    "value": 0.0416,
                    "evidence_text": "Cap Rate 0.045; Commercial Cap Rate 0.07"
                }
            }
        });
        let m = evidence_mismatches(&doc);
        assert_eq!(m.len(), 1, "{m:?}");
        assert!(m[0].contains("stabilised_yield"));
    }

    #[test]
    fn no_flag_when_percent_value_matches_evidence() {
        let doc = json!({
            "details": {
                "stabilised_yield": {
                    "value": 0.045,
                    "evidence_text": "Cap Rate 0.045; Commercial Cap Rate 0.07"
                },
                "margin": {
                    "value": 6.5,
                    "evidence_text": "margin of 6.5%"
                }
            }
        });
        assert!(evidence_mismatches(&doc).is_empty());
    }

    #[test]
    fn flags_money_value_contradicting_evidence() {
        let doc = json!({
            "details": {
                "gdv": {
                    "value": { "amount": 115_000_000.0 },
                    "evidence_text": "GDV of £125,000,000"
                }
            }
        });
        let m = evidence_mismatches(&doc);
        assert_eq!(m.len(), 1, "{m:?}");
        let doc = json!({
            "details": {
                "gdv": {
                    "value": { "amount": 115_000_000.0 },
                    "evidence_text": "GDV of £115.2m across the site"
                }
            }
        });
        let m = evidence_mismatches(&doc);
        // £115.2m vs 115,000,000 is within the 0.2% tolerance → not flagged.
        assert!(m.is_empty(), "{m:?}");
    }

    #[test]
    fn no_flag_when_evidence_has_no_comparable_figure() {
        let doc = json!({
            "details": {
                "gdv": {
                    "value": { "amount": 115_000_000.0 },
                    "evidence_text": "59903985.80 stated on the model"
                }
            }
        });
        assert!(evidence_mismatches(&doc).is_empty());
    }

    // ── contains_figure: boundary-aware matching ─────────────────────

    #[test]
    fn does_not_match_inside_larger_number() {
        assert!(!contains_figure(
            "senior facility of £14,500,000",
            4_500_000.0
        ));
        assert!(!contains_figure("total cost 14500000", 4_500_000.0));
    }

    #[test]
    fn does_not_match_percent_inside_larger_decimal() {
        // 4.5 must not match the "4.5" inside "14.5".
        assert!(!contains_figure("margin of 14.5% over base", 0.045));
    }

    #[test]
    fn matches_formatting_variants() {
        for corpus in [
            "£4,500,000",
            "loan amount: 4500000",
            "facility of £4.5m",
            "facility of 4.5m",
        ] {
            assert!(
                contains_figure(corpus, 4_500_000.0),
                "no match in: {corpus}"
            );
        }
    }

    #[test]
    fn matches_percent_forms() {
        assert!(contains_figure("stabilised yield of 4.85%", 0.0485));
        assert!(!contains_figure("stabilised yield of 5.85%", 0.0485));
    }

    #[test]
    fn allows_m_and_pct_suffixes_but_not_digits() {
        assert!(contains_figure("the 35.8m senior facility", 35_800_000.0));
        assert!(!contains_figure("the 35.81m senior facility", 35_800_000.0));
        assert!(!contains_figure("the 4.85m senior facility", 4_500_000.0));
    }

    // ── collect_checkable ────────────────────────────────────────────

    #[test]
    fn collects_stated_debt_fields_and_skips_computed() {
        let doc = json!({
            "details": {
                "loan_amount": {"value": {"amount": 4500000, "currency": "GBP"}},
                "senior_debt": {"value": {"amount": 10654740, "currency": "GBP"}},
                "developer_equity": {"value": {"amount": 1058660, "currency": "GBP"}},
                "gross_debt": {"value": {"amount": 999999999, "currency": "GBP"}},
                "loan_to_gdv": {"value": 0.65},
                "stabilised_yield": {"value": 4.85},
            }
        });
        let got = collect_checkable(&doc);
        let paths: Vec<&str> = got.iter().map(|(p, _, _)| p.as_str()).collect();
        for want in [
            "loan_amount",
            "senior_debt",
            "developer_equity",
            "stabilised_yield",
        ] {
            assert!(paths.contains(&want), "missing {want}: {paths:?}");
        }
        assert!(!paths.contains(&"gross_debt"), "computed field checked");
        assert!(!paths.contains(&"loan_to_gdv"), "ratio field checked");
    }

    #[test]
    fn collects_facility_commitments_as_decimal_strings() {
        let doc = json!({
            "canonical_deal": {"facilities": [
                {"name": "Senior Debt", "commitment": {"amount": "10654740", "currency": "GBP"}},
                {"name": "Bond", "commitment": {"amount": 1766161, "currency": "GBP"}},
                {"name": "NoAmount", "commitment": {}},
            ]}
        });
        let got = collect_checkable(&doc);
        assert_eq!(got.len(), 2);
        assert!(got[0].0.contains("facilities[0].commitment"));
        assert!(got[0].0.contains("Senior Debt"));
        assert_eq!(got[0].1, 10_654_740.0);
        assert_eq!(got[1].1, 1_766_161.0);
    }

    #[test]
    fn skips_tiny_and_subunit_amounts() {
        let doc = json!({
            "details": {
                "fee_rate": {"value": {"amount": 0.5, "currency": "GBP"}},
                "loan_amount": {"value": {"amount": 4500000, "currency": "GBP"}},
            }
        });
        let got = collect_checkable(&doc);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, "loan_amount");
    }

    // ── apply_fix: type-preserving patches ───────────────────────────

    #[test]
    fn facility_commitment_fix_coerces_number_to_decimal_string() {
        let mut doc = json!({
            "canonical_deal": {"facilities": [
                {"name": "Senior", "commitment": {"amount": "1000000", "currency": "GBP"}},
            ]}
        });
        assert!(apply_fix(
            &mut doc,
            "canonical_deal.facilities[0].commitment",
            &json!(10654740)
        ));
        let amt = &doc["canonical_deal"]["facilities"][0]["commitment"]["amount"];
        assert_eq!(
            amt,
            &json!("10654740"),
            "commitment must stay a decimal string"
        );
    }

    #[test]
    fn facility_commitment_fix_accepts_string_and_rejects_nonnumeric() {
        let mut doc = json!({
            "canonical_deal": {"facilities": [
                {"name": "Senior", "commitment": {"amount": "1", "currency": "GBP"}},
            ]}
        });
        assert!(apply_fix(
            &mut doc,
            "canonical_deal.facilities[0].commitment",
            &json!("4500000.50")
        ));
        assert_eq!(
            doc["canonical_deal"]["facilities"][0]["commitment"]["amount"],
            json!("4500000.50")
        );
        assert!(!apply_fix(
            &mut doc,
            "canonical_deal.facilities[0].commitment",
            &json!({"x": 1})
        ));
        assert!(!apply_fix(
            &mut doc,
            "canonical_deal.facilities[9].commitment",
            &json!(1)
        ));
        assert!(!apply_fix(
            &mut doc,
            "canonical_deal.facilities[x].commitment",
            &json!(1)
        ));
    }

    #[test]
    fn detail_money_fix_preserves_money_object_shape() {
        let mut doc = json!({
            "details": {
                "loan_amount": {"value": {"amount": 4500000, "currency": "GBP"}},
            }
        });
        // String patch coerced to number inside the money object.
        assert!(apply_fix(
            &mut doc,
            "details.loan_amount.value",
            &json!("5250000")
        ));
        let v = &doc["details"]["loan_amount"]["value"];
        assert_eq!(v["amount"], json!(5250000.0));
        assert_eq!(v["currency"], json!("GBP"), "currency must survive");
        // notes stamped
        assert!(doc["details"]["loan_amount"]["notes"]
            .as_str()
            .unwrap()
            .contains("verify pass"));
    }

    #[test]
    fn detail_numeric_fix_coerces_string_to_number() {
        let mut doc = json!({
            "details": {
                "stabilised_yield": {"value": 4.0},
            }
        });
        assert!(apply_fix(
            &mut doc,
            "details.stabilised_yield",
            &json!("4.85")
        ));
        assert_eq!(doc["details"]["stabilised_yield"]["value"], json!(4.85));
        assert!(
            doc["details"]["stabilised_yield"]["value"].is_f64(),
            "must remain numeric, not string"
        );
    }

    #[test]
    fn detail_fix_rejects_unknown_paths() {
        let mut doc = json!({"details": {"loan_amount": {"value": 1}}});
        assert!(!apply_fix(&mut doc, "details.nonexistent", &json!(5)));
        assert!(!apply_fix(&mut doc, "unrelated.path", &json!(5)));
        assert_eq!(doc["details"]["loan_amount"]["value"], json!(1));
    }
}
