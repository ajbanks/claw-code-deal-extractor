use std::path::Path;

use deal_extractor::schema::DealDocument;
use deal_extractor::scoring;

const CACHED_DIR: &str = "/home/alex94/claw-code/rust/crates/deal-extractor/test_output";

fn load_cached(name: &str) -> Option<DealDocument> {
    let path = Path::new(CACHED_DIR).join(format!("{name}.json"));
    if !path.exists() {
        return None;
    }
    let text = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&text).ok()
}

fn run_live(deal: &str) -> Option<serde_json::Value> {
    if std::env::var("RUN_LLM").as_deref() != Ok("1") {
        return None;
    }

    let model =
        std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-5.4".to_string());
    let max_rounds: usize = std::env::var("MAX_REPAIR_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    let docs_dir = match deal {
        "international_house" => "/home/alex94/copilot/test_docs",
        "edenbridge" => "/home/alex94/copilot/test_docs_multi/edenbridge",
        "oakley_square" => "/home/alex94/copilot/test_docs_multi/oakley_square",
        "octagon_retail" => "/home/alex94/copilot/test_docs_multi/octagon_retail",
        _ => return None,
    };

    let corpus = deal_extractor::corpus::Corpus::from_pages(
        deal_extractor::documents::load_documents(Path::new(docs_dir)),
    );
    if corpus.pages.is_empty() {
        eprintln!("[{deal}] no documents found at {docs_dir}");
        return None;
    }

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let extractor = deal_extractor::extract::Extractor::new(&model).ok()?;
        let candidate = extractor.extract(&corpus).await.ok()?;
        let (doc_json, _issues, rounds) =
            deal_extractor::repair::repair_loop(&extractor, &corpus, candidate, max_rounds).await;
        eprintln!("[{deal}] extraction complete, {rounds} repair rounds");

        std::fs::create_dir_all(CACHED_DIR).ok();
        let out_path = Path::new(CACHED_DIR).join(format!("{deal}.json"));
        let json = serde_json::to_string_pretty(&doc_json).ok()?;
        std::fs::write(&out_path, &json).ok();
        eprintln!("[{deal}] cached to {}", out_path.display());

        Some(doc_json)
    })
}

fn get_doc(deal: &str) -> Option<DealDocument> {
    if let Some(cached) = load_cached(deal) {
        eprintln!("[{deal}] using cached output");
        return Some(cached);
    }
    let json = run_live(deal)?;
    serde_json::from_value(json).ok()
}

fn assert_focus_is_bridge(doc: &DealDocument) {
    let cd = doc.canonical_deal.as_ref().expect("canonical_deal missing");
    let focus_id = cd
        .focus_facility_id
        .as_ref()
        .expect("focus_facility_id missing");
    let focus = cd
        .facilities
        .iter()
        .find(|f| &f.id == focus_id)
        .expect("focus facility not in facilities[]");

    assert_eq!(
        focus.loan_type.as_deref(),
        Some("bridge"),
        "CRITICAL: focus facility loan_type must be 'bridge', not 'development'. Got: {:?} (name: {:?})",
        focus.loan_type,
        focus.name,
    );

    let amt = focus
        .commitment
        .as_ref()
        .map(|m| {
            let a = m.amount;
            if a < 1000.0 { a * 1_000_000.0 } else { a }
        })
        .unwrap_or(0.0);
    assert!(
        amt > 20_000_000.0 && amt < 28_000_000.0,
        "CRITICAL: focus commitment must be ~£25.75M (bridge), not £73M (dev). Got: {amt:.0}",
    );

    let disp = focus.disposition.as_deref().unwrap_or("");
    assert_eq!(
        disp, "requested",
        "focus facility disposition must be 'requested', got '{disp}'",
    );
}

#[test]
fn ih_focus_is_bridge_not_dev() {
    let Some(doc) = get_doc("international_house") else {
        eprintln!("[skip] no cached output and RUN_LLM!=1");
        return;
    };
    assert_focus_is_bridge(&doc);
}

#[test]
fn ih_loan_amount_is_25m_not_73m() {
    let Some(doc) = get_doc("international_house") else {
        eprintln!("[skip] no cached output and RUN_LLM!=1");
        return;
    };
    let results = scoring::score_ih(&doc);
    let la_check = results
        .iter()
        .find(|r| r.name == "loan_amount_25.75M")
        .expect("loan_amount_25.75M check missing");
    assert!(
        la_check.passed,
        "CRITICAL: loan_amount must be ~£25.75M (bridge ask), not a sum of all facilities. {}",
        la_check.reason,
    );
}

#[test]
fn ih_full_gold_standard() {
    let Some(doc) = get_doc("international_house") else {
        eprintln!("[skip] no cached output and RUN_LLM!=1");
        return;
    };
    let results = scoring::score_ih(&doc);
    scoring::print_results("IH Gold Standard", &results);
    let failed: Vec<_> = results.iter().filter(|r| !r.passed).collect();
    if !failed.is_empty() {
        for f in &failed {
            eprintln!("  FAIL: {} — {}", f.name, f.reason);
        }
    }
    let total = results.len();
    let pass_count = results.iter().filter(|r| r.passed).count();
    eprintln!("[ih] {pass_count}/{total} checks passed");
    assert!(
        failed.len() <= 3,
        "IH must pass at least {}/{total} checks, failed {} ({})",
        total - 3,
        failed.len(),
        failed.iter().map(|f| f.name.as_str()).collect::<Vec<_>>().join(", "),
    );
}

#[test]
fn edenbridge_scoring() {
    let Some(doc) = get_doc("edenbridge") else {
        eprintln!("[skip] no cached output and RUN_LLM!=1");
        return;
    };
    let results = scoring::score_edenbridge(&doc);
    scoring::print_results("Edenbridge", &results);
    let failed: Vec<_> = results.iter().filter(|r| !r.passed).collect();
    assert!(
        failed.len() <= 1,
        "Edenbridge must pass at least {}/{}, failed: {}",
        results.len() - 1,
        results.len(),
        failed.iter().map(|f| f.name.as_str()).collect::<Vec<_>>().join(", "),
    );
}

#[test]
fn oakley_square_scoring() {
    let Some(doc) = get_doc("oakley_square") else {
        eprintln!("[skip] no cached output and RUN_LLM!=1");
        return;
    };
    let results = scoring::score_oakley_square(&doc);
    scoring::print_results("Oakley Square", &results);
    let failed: Vec<_> = results.iter().filter(|r| !r.passed).collect();
    assert!(
        failed.is_empty(),
        "Oakley Square must pass all checks, failed: {}",
        failed.iter().map(|f| f.name.as_str()).collect::<Vec<_>>().join(", "),
    );
}

#[test]
fn octagon_retail_scoring() {
    let Some(doc) = get_doc("octagon_retail") else {
        eprintln!("[skip] no cached output and RUN_LLM!=1");
        return;
    };
    let results = scoring::score_octagon_retail(&doc);
    scoring::print_results("Octagon Retail", &results);
    let failed: Vec<_> = results.iter().filter(|r| !r.passed).collect();
    assert!(
        failed.len() <= 1,
        "Octagon Retail must pass at least {}/{}, failed: {}",
        results.len() - 1,
        results.len(),
        failed.iter().map(|f| f.name.as_str()).collect::<Vec<_>>().join(", "),
    );
}

#[test]
fn ih_senior_debt_excludes_dev_facility() {
    let Some(doc) = get_doc("international_house") else {
        eprintln!("[skip] no cached output and RUN_LLM!=1");
        return;
    };
    let results = scoring::score_ih(&doc);
    let check = results
        .iter()
        .find(|r| r.name == "senior_debt_excludes_73M")
        .expect("senior_debt_excludes_73M check missing");
    assert!(
        check.passed,
        "senior_debt must NOT include the £73M development facility. {}",
        check.reason,
    );
}

#[test]
fn ih_gross_debt_is_deal_scope_only() {
    let Some(doc) = get_doc("international_house") else {
        eprintln!("[skip] no cached output and RUN_LLM!=1");
        return;
    };
    let results = scoring::score_ih(&doc);
    let check = results
        .iter()
        .find(|r| r.name == "gross_debt_deal_scope")
        .expect("gross_debt_deal_scope check missing");
    assert!(
        check.passed,
        "gross_debt must be deal-scope only (<£60M), not £100M+. {}",
        check.reason,
    );
}

#[test]
fn ih_focus_no_drawn_event() {
    let Some(doc) = get_doc("international_house") else {
        eprintln!("[skip] no cached output and RUN_LLM!=1");
        return;
    };
    let results = scoring::score_ih(&doc);
    let check = results
        .iter()
        .find(|r| r.name == "focus_no_drawn_event")
        .expect("focus_no_drawn_event check missing");
    assert!(
        check.passed,
        "focus facility (the bridge ask) must NOT have a drawn capital event. {}",
        check.reason,
    );
}

#[test]
fn ih_no_profit_role_in_cap_stack() {
    let Some(doc) = get_doc("international_house") else {
        eprintln!("[skip] no cached output and RUN_LLM!=1");
        return;
    };
    let results = scoring::score_ih(&doc);
    let check = results
        .iter()
        .find(|r| r.name == "no_profit_role")
        .expect("no_profit_role check missing");
    assert!(
        check.passed,
        "capital stack must not include profit/return as a funding role. {}",
        check.reason,
    );
}
