use std::path::PathBuf;
use std::process::Command;

use tempfile::tempdir;

fn bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_phrasegen"))
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn run_ok(args: &[&str]) -> String {
    let bin = bin_path();
    let root = repo_root();
    let out = Command::new(&bin)
        .args(args)
        .current_dir(&root)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {bin:?} {args:?}: {e}"));

    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    if !out.status.success() {
        panic!(
            "command failed: {bin:?} {args:?}\nexit={}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n",
            out.status
        );
    }
    // For debugging test flakiness, keep stderr attached on success too.
    // (CLI uses stdout for results and stderr only for exceptional logs.)
    stdout
}

fn find_line<'a>(text: &'a str, prefix: &str) -> &'a str {
    text.lines()
        .find(|l| l.trim_start().starts_with(prefix))
        .unwrap_or_else(|| panic!("missing line with prefix {prefix:?} in output:\n{text}"))
}

fn parse_f64_after_prefix(line: &str, prefix: &str) -> f64 {
    let l = line.trim_start();
    let rest = l
        .strip_prefix(prefix)
        .unwrap_or_else(|| panic!("line did not start with {prefix:?}: {line:?}"))
        .trim();
    rest.parse::<f64>()
        .unwrap_or_else(|e| panic!("failed to parse f64 after {prefix:?} from {line:?}: {e}"))
}

fn parse_delta(line: &str, key: &str) -> f64 {
    let idx = line
        .find(key)
        .unwrap_or_else(|| panic!("missing token {key:?} in line: {line:?}"));
    let rest = &line[(idx + key.len())..];
    let end = rest
        .find(|c: char| c == ' ' || c == '\t')
        .unwrap_or(rest.len());
    rest[..end]
        .parse::<f64>()
        .unwrap_or_else(|e| panic!("failed to parse {key} value in {line:?}: {e}"))
}

fn style_max_chars(style: &str) -> u32 {
    // Use a *slightly* tight limit to force some rejection sampling while keeping the test fast.
    //
    // With `data/wordlist.txt` (lengths mostly 4–7) and 4 words:
    // - styles with 3 separators have min length 22; use 23.
    // - login-title-2digits adds 2 digits: min 21; use 22.
    // - login-title-endpunct adds 1: min 20; use 21.
    match style {
        "spaces" | "hyphens" | "numbers" | "numbers-symbols" => 23,
        "login-title-2digits" => 22,
        "login-title-endpunct" => 21,
        other => panic!("unexpected style in test suite: {other}"),
    }
}

#[test]
fn experiments_all_styles_have_expected_tradeoffs() {
    // Build a tiny, deterministic model from committed example data (no network).
    let root = repo_root();
    let tmp = tempdir().expect("tempdir");
    let model_path = tmp.path().join("model_example.json");
    let example_csv = root.join("data/example.csv");

    run_ok(&[
        "fit",
        "--input",
        example_csv
            .to_str()
            .expect("example_csv path should be utf-8"),
        "--min-count",
        "1",
        "--output-model",
        model_path.to_str().expect("model_path should be utf-8"),
    ]);

    let wordlist = root.join("data/wordlist.txt");
    let wordlist_s = wordlist.to_str().expect("wordlist path should be utf-8");
    let model_s = model_path.to_str().expect("model_path should be utf-8");

    // Cover all built-in style presets.
    let styles = [
        "spaces",
        "hyphens",
        "numbers",
        "numbers-symbols",
        "login-title-2digits",
        "login-title-endpunct",
    ];

    for style in styles {
        let max_chars = style_max_chars(style).to_string();
        let out = run_ok(&[
            "analyze-generator",
            "--model",
            model_s,
            "--wordlist",
            wordlist_s,
            "--corpus",
            wordlist_s,
            "--style",
            style,
            "--words",
            "4",
            "--samples",
            "800",
            "--pick-best-of",
            "10",
            "--max-chars",
            &max_chars,
            "--seed",
            "1",
            "--show-top",
            "0",
        ]);

        // Make sure we exercised the expected code paths.
        assert!(
            out.contains("baseline (pick_best_of=1 semantics):"),
            "missing baseline section for style={style}\n{out}"
        );
        assert!(
            out.contains("picked_fastest_of_10 (models manual choice):"),
            "missing picked section for style={style}\n{out}"
        );

        // Rejection sampling should be active (accept_rate < 1).
        let accept_rate = parse_f64_after_prefix(find_line(&out, "accept_rate:"), "accept_rate:");
        assert!(
            accept_rate > 0.0 && accept_rate < 1.0,
            "unexpected accept_rate={accept_rate} for style={style}\n{out}"
        );

        // Choosing the fastest-of-M must improve typing time (negative deltas).
        let tg = find_line(&out, "typing_gain_ms (picked - baseline):");
        let d_mean = parse_delta(tg, "Δmean=");
        let d_p95 = parse_delta(tg, "Δp95=");
        assert!(
            d_mean < 0.0 && d_p95 < 0.0,
            "expected typing_gain_ms negative for style={style} (got Δmean={d_mean}, Δp95={d_p95})\n{out}"
        );

        // Best-of-M should concentrate the distribution; the marginal-entropy penalty should be non-positive.
        let ep = find_line(
            &out,
            "entropy_penalty_bits_word_marginals (Δsum_positions upper bound):",
        );
        let dh1 = parse_delta(ep, "ΔH1=");
        let dh2 = parse_delta(ep, "ΔH2=");
        let dhinf = parse_delta(ep, "ΔHinf=");
        assert!(
            dh1 <= 1e-9 && dh2 <= 1e-9 && dhinf <= 1e-9,
            "expected marginal entropy deltas <= 0 for style={style} (ΔH1={dh1}, ΔH2={dh2}, ΔHinf={dhinf})\n{out}"
        );
    }
}
