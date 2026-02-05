use phrasegen::corrections::fit_total_ms_linear;

#[test]
fn linear_fit_recovers_coefficients_on_exact_data() {
    let (a, b, c) = (50.0f64, 1.1f64, 120.0f64);
    let mut samples = Vec::new();
    for (pred, bs) in [
        (100.0, 0.0),
        (200.0, 1.0),
        (400.0, 0.0),
        (800.0, 2.0),
        (1500.0, 1.0),
    ] {
        let total = a + b * pred + c * bs;
        samples.push((pred, bs, total));
    }

    let fit = fit_total_ms_linear(&samples).expect("fit");
    assert!((fit.a - a).abs() < 1e-9, "a fit={}", fit.a);
    assert!((fit.b - b).abs() < 1e-12, "b fit={}", fit.b);
    assert!((fit.c - c).abs() < 1e-9, "c fit={}", fit.c);
    assert!(fit.rmse_ms < 1e-9, "rmse={}", fit.rmse_ms);
    assert_eq!(fit.n, samples.len());
}

#[test]
fn linear_fit_returns_none_on_singular_design() {
    // All samples identical => singular normal equations.
    let samples = vec![
        (100.0, 0.0, 200.0),
        (100.0, 0.0, 200.0),
        (100.0, 0.0, 200.0),
    ];
    assert!(fit_total_ms_linear(&samples).is_none());
}
