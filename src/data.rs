use std::path::Path;

use serde::{Deserialize, Serialize};

/// A single training example: the phrase and observed digraph timings.
///
/// Invariants:
/// - `digraph_dt_ms.len() == graphemes_in_phrase.saturating_sub(1)` (best-effort checked)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Row {
    pub phrase: String,
    pub digraph_dt_ms: Vec<f32>,
    /// Optional provenance tag (e.g. "cmu_dsl", "bksd_phrase_en").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Optional note (e.g. filename/user/session identifiers).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum DataError {
    #[error("unsupported file extension: {0}")]
    UnsupportedExtension(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Csv(#[from] csv::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub fn load_rows(path: &Path) -> Result<Vec<Row>, DataError> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "csv" => load_rows_csv(path),
        "jsonl" | "ndjson" => load_rows_jsonl(path),
        _ => Err(DataError::UnsupportedExtension(ext)),
    }
}

/// CSV format:
/// - `phrase`: string
/// - `digraph_dt_ms_json`: JSON array of numbers (ms), length should be chars-1
#[derive(Debug, Deserialize)]
struct CsvRow {
    phrase: String,
    digraph_dt_ms_json: String,
}

fn load_rows_csv(path: &Path) -> Result<Vec<Row>, DataError> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut out = Vec::new();
    for rec in rdr.deserialize::<CsvRow>() {
        let rec = rec?;
        let digraph_dt_ms: Vec<f32> = serde_json::from_str(&rec.digraph_dt_ms_json)?;
        out.push(Row {
            phrase: rec.phrase,
            digraph_dt_ms,
            source: None,
            note: None,
        });
    }
    Ok(out)
}

/// JSONL format: each line is a `Row` (with fields `phrase`, `digraph_dt_ms`).
fn load_rows_jsonl(path: &Path) -> Result<Vec<Row>, DataError> {
    let f = std::fs::File::open(path)?;
    let r = std::io::BufReader::new(f);
    let mut out = Vec::new();
    for line in std::io::BufRead::lines(r) {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Row = serde_json::from_str(&line)?;
        out.push(row);
    }
    Ok(out)
}
