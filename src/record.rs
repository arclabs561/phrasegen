use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal;

use crate::data::Row;

#[derive(Debug, Clone)]
pub struct RecordConfig {
    /// Optional target string the user must type exactly (no backspaces).
    pub target: Option<String>,
    /// Abort the current sample if backspace is pressed.
    pub abort_on_backspace: bool,
    /// Maximum characters allowed in a sample (safety).
    pub max_len: usize,
}

impl Default for RecordConfig {
    fn default() -> Self {
        Self {
            target: None,
            abort_on_backspace: true,
            max_len: 200,
        }
    }
}

#[derive(Debug, Clone)]
pub enum RecordOutcome {
    Recorded(Row),
    Aborted,
}

/// Record a single typing sample from terminal key-press events.
///
/// Captures key-press → key-press deltas (ms), i.e. a “DD” style timing trace.
/// This does *not* capture key-up events (so no hold times).
pub fn record_once(cfg: &RecordConfig) -> anyhow::Result<RecordOutcome> {
    terminal::enable_raw_mode()?;
    let res = record_once_inner(cfg);
    terminal::disable_raw_mode()?;
    res
}

fn record_once_inner(cfg: &RecordConfig) -> anyhow::Result<RecordOutcome> {
    eprintln!();
    if let Some(t) = &cfg.target {
        eprintln!("Type target exactly (Enter to submit, Esc to cancel):");
        eprintln!("{t}");
    } else {
        eprintln!("Type anything (Enter to submit, Esc to cancel):");
    }
    eprintln!("(Backspace aborts this sample.)");
    eprintln!();

    let mut phrase = String::new();
    let mut dts_ms: Vec<f32> = Vec::new();
    let mut last: Option<Instant> = None;

    loop {
        // Block waiting for next event.
        let ev = event::read()?;
        match ev {
            Event::Key(k) => {
                if k.kind != KeyEventKind::Press && k.kind != KeyEventKind::Repeat {
                    continue;
                }
                match k.code {
                    KeyCode::Esc => return Ok(RecordOutcome::Aborted),
                    KeyCode::Enter => break,
                    KeyCode::Backspace => {
                        if cfg.abort_on_backspace && !phrase.is_empty() {
                            eprintln!("\n(backspace) sample aborted; please retype\n");
                            return Ok(RecordOutcome::Aborted);
                        }
                    }
                    KeyCode::Char(c) => {
                        if phrase.chars().count() >= cfg.max_len {
                            eprintln!("\n(max_len reached) sample aborted\n");
                            return Ok(RecordOutcome::Aborted);
                        }
                        let now = Instant::now();
                        if let Some(prev) = last {
                            let dt = now.duration_since(prev);
                            dts_ms.push(dt.as_secs_f32() * 1000.0);
                        }
                        last = Some(now);
                        phrase.push(c);
                        // best-effort echo (raw mode means terminal may not echo)
                        eprint!("{c}");
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    eprintln!();
    if phrase.is_empty() {
        return Ok(RecordOutcome::Aborted);
    }
    // Validate timing length.
    let n = phrase.chars().count();
    if dts_ms.len() != n.saturating_sub(1) {
        // This can happen if non-char keys were pressed; treat as abort.
        eprintln!("(warning) timing length mismatch; sample discarded");
        return Ok(RecordOutcome::Aborted);
    }

    if let Some(t) = &cfg.target {
        if &phrase != t {
            eprintln!("(mismatch) expected target; sample discarded");
            return Ok(RecordOutcome::Aborted);
        }
    }

    let ts_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::from_secs(0))
        .as_millis();

    Ok(RecordOutcome::Recorded(Row {
        phrase,
        digraph_dt_ms: dts_ms,
        source: Some("user_terminal".to_string()),
        note: Some(format!("ts_ms={ts_ms}")),
    }))
}

pub fn append_row_jsonl(path: &std::path::Path, row: &Row) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?,
    );
    serde_json::to_writer(&mut f, row)?;
    use std::io::Write as _;
    f.write_all(b"\n")?;
    f.flush()?;
    Ok(())
}
