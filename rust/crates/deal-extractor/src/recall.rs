//! Recall check — the reverse of `verify`.
//!
//! `verify` asks "does every extracted number appear in the corpus?"
//! (precision). This asks "does every labelled deal figure in the corpus
//! appear somewhere in the extraction?" (recall). It is deterministic: scan
//! the corpus for money amounts and percentages sitting next to a financial
//! label, collect every number the output mentions (structured values and
//! figures quoted in evidence text), and report corpus figures nothing in
//! the output accounts for. Those become targeted fill issues.

use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;

use crate::corpus::Corpus;
use crate::validate::{Issue, Severity};

/// Cap on figures reported per extraction — keeps the fill prompt focused.
const MAX_FIGURES: usize = 15;
/// Money figures below this are not headline deal facts.
const MIN_MONEY: f64 = 100_000.0;
/// Relative tolerance when matching a corpus figure to an output number.
const REL_TOL: f64 = 0.002;

static MONEY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(?P<cur>[£$€])?\s?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?:\s?(?P<suf>bn|billion|million|mn|m|k)\b)?(?P<pct>\s?%)?",
    )
    .expect("money regex")
});

const MONEY_KEYWORDS: &[(&str, u8)] = &[
    ("loan", 2),
    ("facility", 2),
    ("gdv", 2),
    ("development value", 2),
    ("market value", 2),
    ("valuation", 2),
    ("value", 1),
    ("purchase", 2),
    ("price", 2),
    ("acquisition", 2),
    ("total cost", 2),
    ("development cost", 2),
    ("tdc", 2),
    ("construction", 2),
    ("build cost", 2),
    ("cost", 1),
    ("equity", 2),
    ("rent", 2),
    ("income", 1),
    ("noi", 2),
    ("profit", 2),
    ("debt", 2),
    ("senior", 1),
    ("mezz", 2),
    ("bridge", 2),
    ("interest", 1),
    ("fee", 1),
    ("cil", 1),
    ("levy", 1),
    ("contingency", 1),
    ("redemption", 2),
    ("finance", 1),
    ("revenue", 1),
    ("capital", 1),
];

const PCT_KEYWORDS: &[(&str, u8)] = &[
    ("yield", 2),
    ("ltv", 2),
    ("ltc", 2),
    ("ltgdv", 2),
    ("loan to", 2),
    ("loan-to", 2),
    ("margin", 2),
    ("irr", 2),
    ("profit", 2),
    ("coupon", 2),
    ("interest", 2),
    ("cap rate", 2),
    ("return", 1),
    ("rate", 1),
    ("occupancy", 1),
    ("vacancy", 1),
    ("fee", 1),
];

/// Context words that mark a figure as market/comparable data, not a deal fact.
const EXCLUDE: &[&str] = &[
    "comparable",
    "comps",
    "sold for",
    "vat",
    "inflation",
    "growth",
    "hpi",
    "house price index",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FigureKind {
    Money,
    Percent,
}

#[derive(Debug, Clone)]
pub struct UnaccountedFigure {
    pub doc_name: String,
    pub page_label: String,
    pub raw: String,
    pub value: f64,
    pub kind: FigureKind,
    /// Label text immediately before the figure.
    pub label: String,
    /// The source line, trimmed for display.
    pub line: String,
    pub occurrences: usize,
    score: u8,
    tol: f64,
}

impl UnaccountedFigure {
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{} p{}: {} {}",
            self.doc_name, self.page_label, self.label, self.raw
        )
    }

    #[must_use]
    pub fn issue(&self) -> Issue {
        Issue {
            code: "figure_unaccounted",
            path: format!("{}#{}", self.doc_name, self.page_label),
            message: format!(
                "{} p{} states \"{}\" — {} appears nowhere in the extraction{}",
                self.doc_name,
                self.page_label,
                self.line,
                self.raw,
                if self.occurrences > 1 {
                    format!(" (stated {} times in the pack)", self.occurrences)
                } else {
                    String::new()
                }
            ),
            severity: Severity::Error,
        }
    }
}

/// A figure parsed from text: value, kind, the tolerance its stated
/// precision implies, and the byte span of the match.
pub(crate) struct Parsed {
    pub(crate) value: f64,
    pub(crate) kind: FigureKind,
    pub(crate) tol: f64,
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) explicit_money: bool,
}

pub(crate) fn parse_figures(text: &str) -> Vec<Parsed> {
    let mut out = Vec::new();
    for c in MONEY_RE.captures_iter(text) {
        let whole = c.get(0).expect("match");
        let num_m = c.name("num").expect("num");
        let num_s = num_m.as_str().replace(',', "");
        let Ok(n) = num_s.parse::<f64>() else {
            continue;
        };
        let decimals = num_s.split_once('.').map_or(0, |(_, d)| d.len()) as i32;
        let half_step = 0.5 * 10f64.powi(-decimals);
        if c.name("pct").is_some() {
            out.push(Parsed {
                value: n,
                kind: FigureKind::Percent,
                tol: half_step.max(n * REL_TOL),
                start: whole.start(),
                end: whole.end(),
                explicit_money: false,
            });
            continue;
        }
        let mult = match c.name("suf").map(|s| s.as_str().to_lowercase()).as_deref() {
            Some("bn" | "billion") => 1e9,
            Some("m" | "mn" | "million") => 1e6,
            Some("k") => 1e3,
            _ => 1.0,
        };
        let value = n * mult;
        out.push(Parsed {
            value,
            kind: FigureKind::Money,
            tol: (half_step * mult).max(value * REL_TOL),
            start: num_m.start().min(whole.start()),
            end: whole.end(),
            explicit_money: c.name("cur").is_some()
                || c.name("suf").is_some()
                || num_m.as_str().contains(','),
        });
    }
    out
}

/// Every number the output mentions, in each representation a percentage
/// may take (4.5 / 0.045 / 450). Sorted for range lookup.
fn output_numbers(doc: &Value) -> Vec<f64> {
    fn walk(v: &Value, out: &mut Vec<f64>) {
        match v {
            Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    out.push(f);
                }
            }
            Value::String(s) => {
                if let Ok(f) = s.trim().parse::<f64>() {
                    out.push(f);
                } else {
                    out.extend(parse_figures(s).into_iter().map(|p| p.value));
                }
            }
            Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            Value::Object(o) => o.values().for_each(|x| walk(x, out)),
            Value::Bool(_) | Value::Null => {}
        }
    }
    let mut raw = Vec::new();
    walk(doc, &mut raw);
    let mut out: Vec<f64> = raw
        .iter()
        .flat_map(|&f| [f, f * 100.0, f / 100.0])
        .filter(|f| f.is_finite())
        .collect();
    out.sort_by(f64::total_cmp);
    out
}

fn accounted(sorted: &[f64], value: f64, tol: f64) -> bool {
    let lo = sorted.partition_point(|&x| x < value - tol);
    sorted.get(lo).is_some_and(|&x| x <= value + tol)
}

fn floor_boundary(s: &str, mut i: usize) -> usize {
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn ceil_boundary(s: &str, mut i: usize) -> usize {
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// Corpus figures sitting next to a financial label: (doc, page, figure,
/// label, line, score).
fn labelled_figures(corpus: &Corpus) -> Vec<UnaccountedFigure> {
    let mut out = Vec::new();
    for page in &corpus.pages {
        if page.page_label == "error" {
            continue;
        }
        for line in page.text.lines() {
            for p in parse_figures(line) {
                let prefix = &line[..p.start];
                // Only the first figure after a label counts: a figure with
                // other digits between it and the label text is a later
                // column (time series, sensitivity grid), not the stated value.
                let last_alpha = prefix
                    .char_indices()
                    .rfind(|(_, ch)| ch.is_alphabetic())
                    .map(|(i, ch)| i + ch.len_utf8());
                let Some(label_end) = last_alpha else {
                    continue;
                };
                let between = &prefix[label_end..];
                if between.chars().any(|ch| ch.is_ascii_digit()) {
                    continue;
                }
                // Label = nearest text cell (tab-separated sheets) or the
                // 80 chars before the figure (prose / layout PDFs).
                let label_src = &prefix[..label_end];
                let label = if label_src.contains('\t') {
                    label_src
                        .rsplit('\t')
                        .find(|s| s.chars().any(char::is_alphabetic))
                        .unwrap_or("")
                        .to_string()
                } else {
                    label_src[floor_boundary(label_src, label_src.len().saturating_sub(80))..]
                        .to_string()
                };
                let label = label.split_whitespace().collect::<Vec<_>>().join(" ");
                let after = &line[p.end..ceil_boundary(line, (p.end + 40).min(line.len()))];
                // Sheet rows put the row label several cells from the value —
                // "Cap Rate" may sit before intervening alpha cells. Match
                // keywords against every alpha cell on the row, not just the
                // nearest one.
                let context = if line.contains('\t') {
                    let cells = line
                        .split('\t')
                        .filter(|s| s.chars().any(char::is_alphabetic))
                        .collect::<Vec<_>>()
                        .join(" ");
                    format!("{cells} {after}").to_lowercase()
                } else {
                    format!("{label} {after}").to_lowercase()
                };
                if EXCLUDE.iter().any(|x| context.contains(x)) {
                    continue;
                }
                // A bare decimal in (0,1) on a sheet row next to a yield/rate
                // label is a percent stored as a fraction (0.045 = 4.5%).
                let sheet_fraction = p.kind == FigureKind::Money
                    && !p.explicit_money
                    && p.value > 0.0
                    && p.value < 1.0
                    && line.contains('\t')
                    && PCT_KEYWORDS.iter().any(|(k, _)| context.contains(k));
                let (value, tol, kind) = if sheet_fraction {
                    (p.value * 100.0, p.tol * 100.0, FigureKind::Percent)
                } else {
                    (p.value, p.tol, p.kind)
                };
                let (keywords, ok) = match kind {
                    FigureKind::Money => (
                        MONEY_KEYWORDS,
                        // Bare numbers (no £/m/commas) only count on sheet
                        // rows — in prose they are refs, phone numbers etc.
                        value >= MIN_MONEY
                            && (p.explicit_money
                                || line.contains('\t')
                                || page.page_label.parse::<u32>().is_err()),
                    ),
                    FigureKind::Percent => (PCT_KEYWORDS, value > 0.0 && value <= 100.0),
                };
                if !ok {
                    continue;
                }
                let Some(kw_score) = keywords
                    .iter()
                    .filter(|(k, _)| context.contains(k))
                    .map(|(_, s)| *s)
                    .max()
                else {
                    continue;
                };
                let raw = line[p.start..p.end].trim().to_string();
                let line_disp: String = line.split_whitespace().collect::<Vec<_>>().join(" ");
                let line_disp: String = line_disp.chars().take(200).collect();
                out.push(UnaccountedFigure {
                    doc_name: page.doc_name.clone(),
                    page_label: page.page_label.clone(),
                    raw,
                    value,
                    kind,
                    label: label.chars().take(80).collect(),
                    line: line_disp,
                    occurrences: 1,
                    score: kw_score + u8::from(p.explicit_money),
                    tol,
                });
            }
        }
    }
    out
}

/// Does the corpus literally state this money amount (in any of its written
/// forms — "35,810,099.96", "£35.81m")? Used to decide whether a disputed
/// extraction element is evidence-backed or a vote hallucination.
#[must_use]
pub(crate) fn corpus_has_amount(corpus: &Corpus, value: f64) -> bool {
    for page in &corpus.pages {
        for line in page.text.lines() {
            for p in parse_figures(line) {
                if p.kind == FigureKind::Money
                    && (p.value - value).abs() <= p.tol.max(value * REL_TOL)
                {
                    return true;
                }
            }
        }
    }
    false
}

/// Labelled corpus figures that nothing in the output accounts for, deduped
/// by value, ranked (keyword strength, explicit money, repetition across the
/// pack) and capped at `MAX_FIGURES`.
#[must_use]
pub fn unaccounted_figures(doc: &Value, corpus: &Corpus) -> Vec<UnaccountedFigure> {
    let have = output_numbers(doc);
    let mut found: Vec<UnaccountedFigure> = Vec::new();
    for f in labelled_figures(corpus) {
        if accounted(&have, f.value, f.tol) {
            continue;
        }
        if let Some(prev) = found
            .iter_mut()
            .find(|g| g.kind == f.kind && (g.value - f.value).abs() <= g.tol.max(f.tol))
        {
            prev.occurrences += 1;
            if f.score > prev.score {
                let occ = prev.occurrences;
                *prev = f;
                prev.occurrences = occ;
            }
            continue;
        }
        found.push(f);
    }
    found.sort_by_key(|f| std::cmp::Reverse(u32::from(f.score) * 4 + f.occurrences.min(3) as u32));
    found.truncate(MAX_FIGURES);
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::documents::DocPage;
    use serde_json::json;

    fn corpus(pages: &[(&str, &str, &str)]) -> Corpus {
        Corpus::from_pages(
            pages
                .iter()
                .map(|(d, p, t)| DocPage {
                    doc_name: (*d).into(),
                    page_label: (*p).into(),
                    text: (*t).into(),
                })
                .collect(),
        )
    }

    fn values(v: &[UnaccountedFigure]) -> Vec<f64> {
        v.iter().map(|f| f.value).collect()
    }

    #[test]
    fn parses_money_and_percent_forms() {
        let p = parse_figures("GDV £45,250,000, loan £35.8m, yield 4.50%, 12k fee");
        let got: Vec<(f64, FigureKind)> = p.iter().map(|x| (x.value, x.kind)).collect();
        assert!(got.contains(&(45_250_000.0, FigureKind::Money)));
        assert!(got.contains(&(35_800_000.0, FigureKind::Money)));
        assert!(got.contains(&(4.5, FigureKind::Percent)));
        assert!(got.contains(&(12_000.0, FigureKind::Money)));
    }

    #[test]
    fn m_suffix_does_not_eat_units() {
        let p = parse_figures("term 18 months, 450 m2");
        assert!(p.iter().all(|x| x.value < 1000.0), "{:?}", values_p(&p));
    }

    fn values_p(p: &[Parsed]) -> Vec<f64> {
        p.iter().map(|x| x.value).collect()
    }

    #[test]
    fn flags_labelled_figure_missing_from_output() {
        let c = corpus(&[(
            "im.pdf",
            "3",
            "Construction costs £12,400,000 including contingency",
        )]);
        let doc = json!({ "details": { "gross_development_value": { "value": { "amount": 20000000 } } } });
        let got = unaccounted_figures(&doc, &c);
        assert_eq!(values(&got), vec![12_400_000.0]);
        assert!(got[0].issue().message.contains("im.pdf p3"));
    }

    #[test]
    fn decimal_string_and_rounding_count_as_accounted() {
        let c = corpus(&[(
            "im.pdf",
            "1",
            "Senior loan of £35.8m. Total development cost £35,810,100.",
        )]);
        let doc = json!({ "canonical_deal": { "facilities": [
            { "commitment": { "amount": "35810100", "currency": "gbp" } }
        ] } });
        assert!(unaccounted_figures(&doc, &c).is_empty());
    }

    #[test]
    fn evidence_quotes_count_as_accounted() {
        let c = corpus(&[("im.pdf", "1", "Purchase price £8,250,000")]);
        let doc = json!({ "details": { "x": { "evidence_text": "purchase price of £8.25m" } } });
        assert!(unaccounted_figures(&doc, &c).is_empty());
    }

    #[test]
    fn percent_matches_fraction_or_whole() {
        let c = corpus(&[("v.pdf", "2", "Net initial yield 4.50%\nLTV 65%")]);
        let doc = json!({ "kpis": { "a": { "value": 0.045 }, "b": { "value": 65.0 } } });
        assert!(unaccounted_figures(&doc, &c).is_empty());
        let got = unaccounted_figures(&json!({}), &c);
        assert_eq!(values(&got).len(), 2);
    }

    #[test]
    fn ignores_unlabelled_small_and_excluded_figures() {
        let c = corpus(&[(
            "im.pdf",
            "1",
            "Built in 1998 with 120 units.\nComparable sold for £4,000,000\nVAT 20%\nArrangement fee £50,000",
        )]);
        assert!(unaccounted_figures(&json!({}), &c).is_empty());
    }

    #[test]
    fn only_first_figure_after_label_in_a_row() {
        // Cash-flow row: monthly spend columns after the label are skipped.
        let c = corpus(&[(
            "cf.xlsx",
            "Cash Flow",
            "Construction cost\t12400000\t250000\t300000\t410000",
        )]);
        assert_eq!(
            values(&unaccounted_figures(&json!({}), &c)),
            vec![12_400_000.0]
        );
    }

    #[test]
    fn plain_numbers_need_a_sheet_page_or_explicit_money() {
        // A bare 7-digit number in a PDF (no £, no commas) is too ambiguous.
        let c = corpus(&[("im.pdf", "1", "Loan reference 4500000")]);
        assert!(unaccounted_figures(&json!({}), &c).is_empty());
    }

    #[test]
    fn sheet_fraction_counts_as_percent() {
        // Yield stored as a decimal in a spreadsheet cell ("0.045" = 4.5%).
        let c = corpus(&[("m.xlsx", "Returns", "Stabilised yield\t0.045")]);
        let got = unaccounted_figures(&json!({}), &c);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].kind, FigureKind::Percent);
        assert_eq!(got[0].value, 4.5);
        // Accounted whether the output stores 0.045 or 4.5.
        let doc = json!({ "details": { "stabilised_yield": { "value": 0.045 } } });
        assert!(unaccounted_figures(&doc, &c).is_empty());
    }

    #[test]
    fn sheet_fraction_matches_row_label_not_nearest_cell() {
        // Luton CF sheet shape: "Cap Rate" is the row label but intervening
        // alpha cells ("Education Contribution", "x") sit before the value.
        let c = corpus(&[(
            "cf.xlsx",
            "Luton P2 Block C",
            "Cap Rate\t\tEducation Contribution\tx\t\t\t\t0.045",
        )]);
        let got = unaccounted_figures(&json!({}), &c);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].value, 4.5);
    }

    #[test]
    fn sheet_fraction_needs_a_pct_label() {
        let c = corpus(&[("m.xlsx", "Sheet1", "Rainfall factor\t0.045")]);
        assert!(unaccounted_figures(&json!({}), &c).is_empty());
    }

    #[test]
    fn dedupes_repeated_figures_and_ranks_them_first() {
        let c = corpus(&[
            (
                "a.pdf",
                "1",
                "Loan facility £20,000,000\nOther income £150,000",
            ),
            ("b.pdf", "4", "Requested loan £20m"),
        ]);
        let got = unaccounted_figures(&json!({}), &c);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].value, 20_000_000.0);
        assert_eq!(got[0].occurrences, 2);
    }
}
