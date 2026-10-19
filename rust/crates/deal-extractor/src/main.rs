use std::path::PathBuf;
use std::time::Instant;

use deal_extractor::corpus::Corpus;
use deal_extractor::documents;
use deal_extractor::extract::Extractor;
use deal_extractor::pipeline;
use deal_extractor::repair;
use deal_extractor::schema::DealDocument;
use deal_extractor::scoring;
use deal_extractor::validate;

#[derive(Debug)]
struct DealSpec {
    name: &'static str,
    docs_dir: &'static str,
    scorer: fn(&DealDocument) -> Vec<scoring::CheckResult>,
}

const DEALS: &[DealSpec] = &[
    DealSpec {
        name: "International House",
        docs_dir: "/home/alex94/copilot/test_docs",
        scorer: scoring::score_ih,
    },
    DealSpec {
        name: "Edenbridge",
        docs_dir: "/home/alex94/copilot/test_docs_multi/edenbridge",
        scorer: scoring::score_edenbridge,
    },
    DealSpec {
        name: "Oakley Square",
        docs_dir: "/home/alex94/copilot/test_docs_multi/oakley_square",
        scorer: scoring::score_oakley_square,
    },
    DealSpec {
        name: "Octagon Retail",
        docs_dir: "/home/alex94/copilot/test_docs_multi/octagon_retail",
        scorer: scoring::score_octagon_retail,
    },
];

fn print_usage() {
    eprintln!(
        "Usage: deal-extractor [--model MODEL] [--out-dir DIR] [--deal NAME] [--docs-dir DIR] [--score FILE] [--max-repair-rounds N]\n\
         \n\
         --model MODEL            OpenAI model (default: gpt-5.4)\n\
         --out-dir DIR            Output directory for JSON (default: ./out)\n\
         --deal NAME              Process only one deal (ih|edenbridge|oakley_square|octagon_retail)\n\
         --docs-dir DIR           Override docs dir (used with --deal)\n\
         --score FILE             Score a previously-extracted JSON file (no LLM calls)\n\
         --max-repair-rounds N    Max validate→repair rounds (default: 3)\n\
         --single-pass            Legacy one-call extraction instead of the production multi-pass pipeline\n\
         --list                   List available deals and exit"
    );
}

fn deal_by_name(name: &str) -> Option<&'static DealSpec> {
    let key = name.to_lowercase();
    let key = match key.as_str() {
        "ih" => "international_house",
        "edenbridge" => "edenbridge",
        "oakley" | "oakley_square" => "oakley_square",
        "octagon" | "octagon_retail" => "octagon_retail",
        other => other,
    };
    DEALS.iter().find(|d| {
        let dkey = d.name.to_lowercase().replace(' ', "_");
        dkey == key || dkey.contains(key) || key.contains(&dkey)
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-5.4".to_string());
    let mut out_dir = PathBuf::from("./out");
    let mut single_deal: Option<String> = None;
    let mut docs_override: Option<String> = None;
    let mut score_file: Option<String> = None;
    let mut max_repair_rounds: usize = 3;
    let mut list_only = false;
    let mut single_pass = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model = args.get(i + 1).cloned().unwrap_or(model);
                i += 2;
            }
            "--out-dir" => {
                out_dir = PathBuf::from(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--deal" => {
                single_deal = args.get(i + 1).cloned();
                i += 2;
            }
            "--docs-dir" => {
                docs_override = args.get(i + 1).cloned();
                i += 2;
            }
            "--score" => {
                score_file = args.get(i + 1).cloned();
                i += 2;
            }
            "--max-repair-rounds" => {
                max_repair_rounds = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(3);
                i += 2;
            }
            "--list" => {
                list_only = true;
                i += 1;
            }
            "--single-pass" => {
                single_pass = true;
                i += 1;
            }
            "--help" | "-h" => {
                print_usage();
                return;
            }
            other => {
                eprintln!("unknown arg: {other}");
                print_usage();
                std::process::exit(2);
            }
        }
        if i == 0 {
            i = 1;
        }
    }

    if list_only {
        println!("Available deals:");
        for d in DEALS {
            println!("  {} — {}", d.name, d.docs_dir);
        }
        return;
    }

    if let Some(file) = score_file {
        let text = std::fs::read_to_string(&file).unwrap_or_else(|e| {
            eprintln!("cannot read {file}: {e}");
            std::process::exit(1);
        });
        let doc: DealDocument = serde_json::from_str(&text).unwrap_or_else(|e| {
            eprintln!("cannot parse {file}: {e}");
            std::process::exit(1);
        });
        let name = single_deal.unwrap_or_else(|| "Unknown".to_string());
        let spec = deal_by_name(&name);
        let results = match spec {
            Some(s) => (s.scorer)(&doc),
            None => scoring::score_ih(&doc),
        };
        scoring::print_results(&name, &results);
        let failed = results.iter().filter(|r| !r.passed).count();
        std::process::exit(if failed == 0 { 0 } else { 1 });
    }

    std::fs::create_dir_all(&out_dir).ok();

    let deals_to_run: Vec<DealSpec> = if let Some(name) = &single_deal {
        match deal_by_name(name) {
            Some(spec) => {
                let mut s = DealSpec {
                    name: spec.name,
                    docs_dir: spec.docs_dir,
                    scorer: spec.scorer,
                };
                if let Some(ref d) = docs_override {
                    s.docs_dir = Box::leak(d.clone().into_boxed_str());
                }
                vec![s]
            }
            None => {
                eprintln!("unknown deal: {name}. Use --list to see options.");
                std::process::exit(2);
            }
        }
    } else {
        DEALS
            .iter()
            .map(|d| DealSpec {
                name: d.name,
                docs_dir: d.docs_dir,
                scorer: d.scorer,
            })
            .collect()
    };

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let extractor = match Extractor::new(&model) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("failed to create extractor for model {model}: {e}");
                eprintln!("ensure OPENAI_API_KEY is set");
                std::process::exit(1);
            }
        };

        let mut total_passed = 0usize;
        let mut total_failed = 0usize;

        for spec in &deals_to_run {
            println!("\n>>> Processing {} ({})", spec.name, spec.docs_dir);
            let start = Instant::now();
            let corpus = Corpus::from_pages(documents::load_documents(std::path::Path::new(
                spec.docs_dir,
            )));
            println!(
                "    loaded {} pages ({} chars)",
                corpus.pages.len(),
                corpus.char_count()
            );
            if corpus.pages.is_empty() {
                eprintln!("    no documents found, skipping");
                continue;
            }

            // Default: the production multi-pass pipeline (same routing as the
            // service). --single-pass keeps the legacy one-call extraction.
            let run = if single_pass {
                match extractor.extract(&corpus).await {
                    Ok(candidate) => {
                        Ok(
                            repair::repair_loop(&extractor, &corpus, candidate, max_repair_rounds)
                                .await,
                        )
                    }
                    Err(e) => Err(e),
                }
            } else {
                pipeline::run_routed(&extractor, &corpus, max_repair_rounds, None, None, None)
                    .await
                    .map(|r| {
                        let rc = &r.diagnostics.recall;
                        println!(
                            "    recall: retries={:?} gaps {}→{} unaccounted {}→{} filled={} tool_merged={}",
                            rc.pass_retries,
                            rc.coverage_gaps_initial.len(),
                            rc.coverage_gaps_remaining.len(),
                            rc.unaccounted_figures.len(),
                            rc.unaccounted_remaining.len(),
                            rc.coverage_filled.len(),
                            rc.tool_fill_merged.len(),
                        );
                        for f in &rc.unaccounted_remaining {
                            println!("      still unaccounted: {f}");
                        }
                        for g in &rc.coverage_gaps_remaining {
                            println!("      still empty: {g}");
                        }
                        (
                            r.document,
                            r.diagnostics.issues_remaining,
                            r.diagnostics.repair_rounds,
                        )
                    })
            };
            match run {
                Ok((doc_json, issues, rounds)) => {

                    let out_file = out_dir.join(format!(
                        "{}.json",
                        spec.name.to_lowercase().replace(' ', "_")
                    ));
                    let json = serde_json::to_string_pretty(&doc_json)
                        .unwrap_or_else(|e| format!("serialize error: {e}"));
                    std::fs::write(&out_file, &json).ok();
                    println!("    wrote {} ({} bytes)", out_file.display(), json.len());
                    println!(
                        "    elapsed {:.1}s, repair rounds: {}",
                        start.elapsed().as_secs_f64(),
                        rounds
                    );
                    let errors: Vec<_> = issues
                        .iter()
                        .filter(|i| i.severity == validate::Severity::Error)
                        .collect();
                    if !errors.is_empty() {
                        println!("    {} remaining issue(s):", errors.len());
                        for i in errors {
                            println!("      - [{}] {}", i.code, i.message);
                        }
                    }

                    // Score via the tolerant scorer view.
                    match serde_json::from_value::<DealDocument>(doc_json.clone()) {
                        Ok(doc) => {
                            let results = (spec.scorer)(&doc);
                            scoring::print_results(spec.name, &results);
                            total_passed += results.iter().filter(|r| r.passed).count();
                            total_failed += results.iter().filter(|r| !r.passed).count();
                        }
                        Err(e) => {
                            eprintln!("    scorer could not parse output: {e}");
                            total_failed += 1;
                        }
                    }
                }
                Err(e) => {
                    eprintln!("    extraction failed: {e}");
                    total_failed += 1;
                }
            }
        }

        println!("\n=== TOTALS: {total_passed} passed, {total_failed} failed ===");
        std::process::exit(if total_failed == 0 { 0 } else { 1 });
    });
}
