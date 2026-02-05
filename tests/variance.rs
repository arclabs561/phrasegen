use phrasegen::data::Row;
use phrasegen::model::{fit_digraph_model, FitConfig};
use phrasegen::score::score_phrase;
use proptest::prelude::*;

fn mean_var_ms2(xs: &[f32]) -> (f32, f32) {
    if xs.is_empty() {
        return (0.0, 0.0);
    }
    let n = xs.len() as f64;
    let sum = xs.iter().map(|&x| x as f64).sum::<f64>();
    let sum_sq = xs.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();
    // Match `fit_digraph_model`'s computation: do math in f64, then cast.
    let mean_f = sum / n;
    let ex2_f = sum_sq / n;
    let var_f = (ex2_f - mean_f * mean_f).max(0.0);
    (mean_f as f32, var_f as f32)
}

proptest! {
    #[test]
    fn fit_computes_mean_and_variance_for_single_digraph(dts in prop::collection::vec(0.0f32..2000.0f32, 1..200)) {
        let rows: Vec<Row> = dts.iter().map(|&dt| Row {
            phrase: "ab".to_string(),
            digraph_dt_ms: vec![dt],
            total_ms: None,
            backspaces: None,
            source: None,
            note: None,
        }).collect();

        let (m, stats) = fit_digraph_model(&rows, FitConfig { min_count: 1, clamp_dt_ms: None, allow_corrections: true });
        let (mu, var) = mean_var_ms2(&dts);
        let tol_mu = 1e-3f32.max(1e-5f32 * mu.abs());
        let tol_var = 1e-3f32.max(1e-5f32 * var.abs());

        prop_assert!((m.mean_ms_for("a", "b") - mu).abs() <= tol_mu);
        prop_assert!((m.var_ms2_for("a", "b") - var).abs() <= tol_var);

        prop_assert!((stats.global_mean_ms - mu).abs() <= tol_mu);
        prop_assert!((stats.global_var_ms2 - var).abs() <= tol_var);
    }

    #[test]
    fn score_std_matches_sqrt_sum_var_for_two_digraphs(
        dts_ab in prop::collection::vec(0.0f32..2000.0f32, 1..100),
        dts_ba in prop::collection::vec(0.0f32..2000.0f32, 1..100),
    ) {
        let mut rows: Vec<Row> = Vec::new();
        for &dt in &dts_ab {
            rows.push(Row {
                phrase: "ab".to_string(),
                digraph_dt_ms: vec![dt],
                total_ms: None,
                backspaces: None,
                source: None,
                note: None,
            });
        }
        for &dt in &dts_ba {
            rows.push(Row {
                phrase: "ba".to_string(),
                digraph_dt_ms: vec![dt],
                total_ms: None,
                backspaces: None,
                source: None,
                note: None,
            });
        }

        let (m, _stats) = fit_digraph_model(&rows, FitConfig { min_count: 1, clamp_dt_ms: None, allow_corrections: true });
        let (mu_ab, var_ab) = mean_var_ms2(&dts_ab);
        let (mu_ba, var_ba) = mean_var_ms2(&dts_ba);

        // Phrase "aba" has digraphs ("a","b") then ("b","a").
        let sc = score_phrase(&m, "aba");
        prop_assert!((sc.predicted_ms - (mu_ab + mu_ba)).abs() <= 1e-2);

        let expected_sd = (var_ab + var_ba).max(0.0).sqrt();
        let got_sd = sc.predicted_std_ms.unwrap_or(0.0);
        prop_assert!((got_sd - expected_sd).abs() <= 1e-2);
    }
}
