use phrasegen::adapt::{adapt_digraph_model, AdaptConfig};
use phrasegen::data::load_rows;
use phrasegen::model::{fit_digraph_model, DigraphModel, FitConfig};
use phrasegen::score::score_phrase;

#[test]
fn fit_and_score_smoke() {
    let rows = load_rows(std::path::Path::new("data/example.csv")).unwrap();
    let (m, stats) = fit_digraph_model(
        &rows,
        FitConfig {
            min_count: 1,
            clamp_dt_ms: None,
            allow_corrections: false,
        },
    );
    assert!(stats.rows >= 1);
    assert!(m.global_mean_ms() >= 0.0);
    let s = score_phrase(&m, "hello world");
    assert!(s.predicted_ms >= 0.0);
}

#[test]
fn model_roundtrip_json() {
    let rows = load_rows(std::path::Path::new("data/example.csv")).unwrap();
    let (m, _stats) = fit_digraph_model(
        &rows,
        FitConfig {
            min_count: 1,
            clamp_dt_ms: None,
            allow_corrections: false,
        },
    );

    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("model.json");
    m.save_json(&p).unwrap();
    let m2 = DigraphModel::load_json(&p).unwrap();
    let s1 = score_phrase(&m, "hello world").predicted_ms;
    let s2 = score_phrase(&m2, "hello world").predicted_ms;
    assert!((s1 - s2).abs() <= 1e-6);
}

#[test]
fn adapt_model_smoke() {
    let rows = load_rows(std::path::Path::new("data/example.csv")).unwrap();
    let (base, _stats) = fit_digraph_model(
        &rows,
        FitConfig {
            min_count: 1,
            clamp_dt_ms: None,
            allow_corrections: false,
        },
    );
    let (tuned, stats) = adapt_digraph_model(
        &base,
        &rows,
        AdaptConfig {
            prior_count: 1.0,
            min_new_count: 1,
        },
    );
    assert!(stats.user_rows >= 1);
    // Should still be able to score.
    let s = score_phrase(&tuned, "hello world");
    assert!(s.predicted_ms >= 0.0);
}
