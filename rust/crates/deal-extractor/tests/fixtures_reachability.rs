//! Offline integration test: verifies that document extraction surfaces the
//! gold-standard values textually before any LLM call is made. If a value
//! is NOT found here, the LLM cannot be blamed for missing it.
//!
//! These tests require the fixtures to exist at the copilot repo paths.
//! They are skipped (with a clear message) if the dirs are absent, so the
//! crate still builds/tests in CI without the fixtures.

use deal_extractor::documents;
use std::path::Path;

fn corpus(dir: &str) -> Option<String> {
    let path = Path::new(dir);
    if !path.is_dir() {
        return None;
    }
    let pages = documents::load_documents(path);
    Some(documents::render_corpus(&pages))
}

fn assert_contains(corpus: &str, needle: &str, label: &str) {
    let lower = corpus.to_lowercase();
    if !lower.contains(&needle.to_lowercase()) {
        panic!("gold-standard value '{label}' ('{needle}') not found in extracted corpus — LLM cannot be blamed for missing it");
    }
}

#[test]
fn ih_corpus_reachability() {
    let Some(corpus) = corpus("/home/alex94/copilot/test_docs") else {
        eprintln!("[skip] IH fixtures not present");
        return;
    };
    let pages = documents::load_documents(Path::new("/home/alex94/copilot/test_docs"));
    assert!(
        pages.len() > 50,
        "IH should yield many pages, got {}",
        pages.len()
    );
    assert_contains(&corpus, "W5 5DB", "postcode");
    assert_contains(&corpus, "25.75", "25.75M bridge ask");
    assert_contains(&corpus, "73393", "73.39M development facility");
    assert_contains(&corpus, "115200", "115.2M GDV");
    assert_contains(&corpus, "148", "148 units phase 1");
    assert_contains(&corpus, "bridge", "bridge keyword");
    assert_contains(&corpus, "development", "development keyword");
    assert_contains(&corpus, "mezzanine", "mezzanine keyword");
    eprintln!(
        "[ok] IH corpus: {} pages, {} chars",
        pages.len(),
        corpus.len()
    );
}

#[test]
fn edenbridge_corpus_reachability() {
    let Some(corpus) = corpus("/home/alex94/copilot/test_docs_multi/edenbridge") else {
        eprintln!("[skip] Edenbridge fixtures not present");
        return;
    };
    assert_contains(&corpus, "mezzanine", "mezzanine keyword");
    assert_contains(&corpus, "development", "development keyword");
    eprintln!("[ok] Edenbridge corpus: {} chars", corpus.len());
}

#[test]
fn oakley_square_corpus_reachability() {
    let Some(corpus) = corpus("/home/alex94/copilot/test_docs_multi/oakley_square") else {
        eprintln!("[skip] Oakley Square fixtures not present");
        return;
    };
    assert_contains(&corpus, "whole", "whole loan keyword");
    eprintln!("[ok] Oakley Square corpus: {} chars", corpus.len());
}

#[test]
fn octagon_retail_corpus_reachability() {
    let Some(corpus) = corpus("/home/alex94/copilot/test_docs_multi/octagon_retail") else {
        eprintln!("[skip] Octagon Retail fixtures not present");
        return;
    };
    assert_contains(&corpus, "bridge", "bridge keyword");
    assert_contains(&corpus, "retail", "retail keyword");
    eprintln!("[ok] Octagon Retail corpus: {} chars", corpus.len());
}

#[test]
fn scoring_sanity_known_good_ih() {
    // A hand-crafted IH-shaped document should score 100% on the IH scorer.
    use deal_extractor::schema::*;
    use deal_extractor::scoring;
    let doc = DealDocument {
        canonical_deal: Some(CanonicalDeal {
            facilities: vec![
                Facility {
                    id: "bridge".into(),
                    name: Some("Bridge Facility".into()),
                    role: Some("senior_debt".into()),
                    loan_type: Some("bridge".into()),
                    commitment: Some(Money {
                        amount: 25_750_000.0,
                        currency: Some("GBP".into()),
                        ..Default::default()
                    }),
                    term_months: Some(6),
                    status: Some("candidate".into()),
                    disposition: Some("requested".into()),
                    ..Default::default()
                },
                Facility {
                    id: "dev".into(),
                    name: Some("Development Facility".into()),
                    role: Some("senior_debt".into()),
                    loan_type: Some("development".into()),
                    commitment: Some(Money {
                        amount: 73_393_202.0,
                        currency: Some("GBP".into()),
                        ..Default::default()
                    }),
                    term_months: Some(24),
                    status: Some("candidate".into()),
                    disposition: Some("proposed".into()),
                    ..Default::default()
                },
                Facility {
                    id: "shojin_senior".into(),
                    name: Some("Shojin Senior Debt".into()),
                    role: Some("senior_debt".into()),
                    commitment: Some(Money {
                        amount: 20_361_250.0,
                        currency: Some("GBP".into()),
                        ..Default::default()
                    }),
                    status: Some("repaid".into()),
                    disposition: Some("repaid".into()),
                    ..Default::default()
                },
                Facility {
                    id: "shojin_mezz".into(),
                    name: Some("Shojin Mezzanine Debt".into()),
                    role: Some("mezzanine_debt".into()),
                    commitment: Some(Money {
                        amount: 3_228_000.0,
                        currency: Some("GBP".into()),
                        ..Default::default()
                    }),
                    status: Some("repaid".into()),
                    disposition: Some("repaid".into()),
                    ..Default::default()
                },
                Facility {
                    id: "equity".into(),
                    name: Some("Developer Equity".into()),
                    role: Some("developer_equity".into()),
                    commitment: Some(Money {
                        amount: 5_500_000.0,
                        currency: Some("GBP".into()),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ],
            focus_facility_id: Some("bridge".into()),
            focus_resolution_status: Some("resolved".into()),
            valuations: vec![
                Valuation {
                    id: "gdv".into(),
                    value_type: Some("gross_development_value".into()),
                    amount: Money {
                        amount: 115_200_000.0,
                        currency: Some("GBP".into()),
                        ..Default::default()
                    },
                    ..Default::default()
                },
                Valuation {
                    id: "gdv_p1".into(),
                    value_type: Some("gross_development_value".into()),
                    amount: Money {
                        amount: 80_545_000.0,
                        currency: Some("GBP".into()),
                        ..Default::default()
                    },
                    ..Default::default()
                },
                Valuation {
                    id: "mv".into(),
                    value_type: Some("market_value".into()),
                    amount: Money {
                        amount: 32_500_000.0,
                        currency: Some("GBP".into()),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ],
            phases: vec![
                Phase {
                    id: "p1".into(),
                    name: Some("Phase 1".into()),
                    sequence: Some(1),
                    unit_count: Some(148),
                    ..Default::default()
                },
                Phase {
                    id: "p2".into(),
                    name: Some("Phase 2".into()),
                    sequence: Some(2),
                    unit_count: Some(41),
                    ..Default::default()
                },
            ],
            capital_events: vec![
                CapitalEvent {
                    id: "e1".into(),
                    event_type: Some("fully_repaid".into()),
                    facility_id: Some("shojin_senior".into()),
                    ..Default::default()
                },
                CapitalEvent {
                    id: "e2".into(),
                    event_type: Some("fully_repaid".into()),
                    facility_id: Some("shojin_mezz".into()),
                    ..Default::default()
                },
            ],
            ..Default::default()
        }),
        classification: Some(Classification {
            capital_position: Some("debt".into()),
            asset_class_primary: Some("residential".into()),
            project_type: Some(CategoryValue {
                category: Some("development".into()),
                ..Default::default()
            }),
            ..Default::default()
        }),
        details: Some(DealDetails {
            loan_amount: Some(SourcedValue {
                value: Some(serde_json::json!({"total":{"amount":"25750000"}})),
                source: Some("focus_facility".into()),
                ..Default::default()
            }),
            senior_debt: Some(SourcedValue {
                value: Some(serde_json::json!({"amount":"25750000"})),
                source: Some("facilities".into()),
                ..Default::default()
            }),
            gross_debt: Some(SourcedValue {
                value: Some(serde_json::json!({"amount":"25750000"})),
                source: Some("facilities".into()),
                ..Default::default()
            }),
            developer_equity: Some(SourcedValue {
                value: Some(serde_json::json!({"amount":"5500000"})),
                source: Some("facilities".into()),
                ..Default::default()
            }),
            market_value: Some(SourcedValue {
                value: Some(serde_json::json!({"amount":"32500000"})),
                source: Some("valuations".into()),
                ..Default::default()
            }),
            gross_development_value: Some(SourcedValue {
                value: Some(serde_json::json!({"amount":"115200000"})),
                source: Some("valuations".into()),
                ..Default::default()
            }),
            address: Some(SourcedValue {
                value: Some(serde_json::json!({"postcode":"W5 5DB"})),
                source: Some("doc".into()),
                ..Default::default()
            }),
            gross_internal_area: Some(SourcedValue {
                value: Some(serde_json::json!(194_871.0)),
                source: Some("doc".into()),
                ..Default::default()
            }),
            ..Default::default()
        }),
        kpis: Some(Kpis {
            loan_to_value: Some(SourcedValue {
                value: Some(serde_json::json!({"value":79.0})),
                source: Some("computed".into()),
                ..Default::default()
            }),
            loan_to_gdv: Some(SourcedValue {
                value: Some(serde_json::json!({"value":22.0})),
                source: Some("computed".into()),
                ..Default::default()
            }),
            build_cost_per_area_unit: Some(SourcedValue {
                value: Some(serde_json::json!({"amount":"180"})),
                source: Some("doc".into()),
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..Default::default()
    };
    let results = scoring::score_ih(&doc);
    let failed: Vec<_> = results.iter().filter(|r| !r.passed).collect();
    if !failed.is_empty() {
        for f in &failed {
            eprintln!("FAIL: {} — {}", f.name, f.reason);
        }
        panic!(
            "known-good IH fixture should score 100%, got {} failures",
            failed.len()
        );
    }
    eprintln!(
        "[ok] known-good IH fixture scored 100% ({} checks)",
        results.len()
    );
}
