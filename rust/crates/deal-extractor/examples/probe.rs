use deal_extractor::documents;
use std::path::Path;
fn main() {
    let pages = documents::load_documents(Path::new("/home/alex94/copilot/test_docs"));
    let corpus = documents::render_corpus(&pages);
    println!("IH: {} pages, {} chars", pages.len(), corpus.len());
    for needle in [
        "73393", "73,393", "73.39", "115200", "115,200", "115.2", "23589", "23,589", "23.58",
        "189", "148",
    ] {
        let n = corpus
            .to_lowercase()
            .matches(&needle.to_lowercase())
            .count();
        println!("  '{needle}' x{n}");
    }
    println!();
    let pages2 = documents::load_documents(Path::new(
        "/home/alex94/copilot/test_docs_multi/oakley_square",
    ));
    let corpus2 = documents::render_corpus(&pages2);
    println!("Oakley: {} pages, {} chars", pages2.len(), corpus2.len());
    for needle in ["whole", "loan", "camden"] {
        let n = corpus2
            .to_lowercase()
            .matches(&needle.to_lowercase())
            .count();
        println!("  '{needle}' x{n}");
    }
}
