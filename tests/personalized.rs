use fastphrase::data::Row;
use fastphrase::model::DigraphModel;
use fastphrase::timing::{build_personalized_model, AnyTimingModel, TimingModel};

#[test]
fn personalized_backoff_imputes_unseen_digraphs() {
    // Base: no digraphs, global mean 300ms.
    let base = DigraphModel::from_parts(std::collections::HashMap::new(), 300.0);

    // User only types digraphs starting with 'a' quickly.
    let user_rows = vec![
        Row {
            phrase: "ab".to_string(),
            digraph_dt_ms: vec![100.0],
            source: Some("user".to_string()),
            note: None,
        },
        Row {
            phrase: "ac".to_string(),
            digraph_dt_ms: vec![100.0],
            source: Some("user".to_string()),
            note: None,
        },
    ];

    let (pm, _adapt, _bstats) = build_personalized_model(
        &base,
        &user_rows,
        fastphrase::adapt::AdaptConfig {
            prior_count: 0.0,
            min_new_count: 1,
        },
        1,
    );

    // Seen digraph should be close to user value (100ms).
    let seen = pm.mean_ms_for("a", "b");
    assert!((seen - 100.0).abs() < 1e-3, "seen={seen}");

    // Unseen digraph starting with 'a' should be imputed lower than base global.
    let imp = pm.mean_ms_for("a", "z");
    assert!(imp < 300.0, "imp={imp}");
    assert!(imp > 0.0, "imp={imp}");
}

#[test]
fn any_timing_model_loads_personalized_and_scores() {
    let base = DigraphModel::from_parts(std::collections::HashMap::new(), 300.0);
    let user_rows = vec![Row {
        phrase: "ab".to_string(),
        digraph_dt_ms: vec![100.0],
        source: Some("user".to_string()),
        note: None,
    }];
    let (pm, _adapt, _bstats) = build_personalized_model(
        &base,
        &user_rows,
        fastphrase::adapt::AdaptConfig {
            prior_count: 0.0,
            min_new_count: 1,
        },
        1,
    );

    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("pm.json");
    pm.save_json(&p).unwrap();

    let any = AnyTimingModel::load_json(&p).unwrap();
    let ms = any.mean_ms_for("a", "b");
    assert!((ms - 100.0).abs() < 1e-3, "ms={ms}");
}
