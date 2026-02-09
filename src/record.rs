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
    if cfg.abort_on_backspace {
        eprintln!("(Backspace aborts this sample.)");
    } else {
        eprintln!("(Backspace is allowed; timings include correction overhead.)");
    }
    eprintln!();

    // Store the current (editable) character buffer and the timestamp each character was typed.
    // This lets us support backspace while still producing a final phrase and a digraph dt vector.
    let mut buf: Vec<(char, Instant)> = Vec::new();
    let mut backspaces: u32 = 0;
    let mut start: Option<Instant> = None;

    loop {
        // Block waiting for next event.
        let ev = event::read()?;
        if let Event::Key(k) = ev {
            if k.kind != KeyEventKind::Press && k.kind != KeyEventKind::Repeat {
                continue;
            }
            match k.code {
                KeyCode::Esc => return Ok(RecordOutcome::Aborted),
                KeyCode::Enter => break,
                KeyCode::Backspace => {
                    if cfg.abort_on_backspace && !buf.is_empty() {
                        eprintln!("\n(backspace) sample aborted; please retype\n");
                        return Ok(RecordOutcome::Aborted);
                    }
                    backspaces = backspaces.saturating_add(1);
                    if !buf.is_empty() {
                        buf.pop();
                        // best-effort echo: backspace + space + backspace to erase
                        eprint!("\u{8} \u{8}");
                    }
                }
                KeyCode::Char(c) => {
                    if buf.len() >= cfg.max_len {
                        eprintln!("\n(max_len reached) sample aborted\n");
                        return Ok(RecordOutcome::Aborted);
                    }
                    let now = Instant::now();
                    start.get_or_insert(now);
                    buf.push((c, now));
                    // best-effort echo (raw mode means terminal may not echo)
                    eprint!("{c}");
                }
                _ => {}
            }
        }
    }

    eprintln!();
    if buf.is_empty() {
        return Ok(RecordOutcome::Aborted);
    }
    let phrase: String = buf.iter().map(|(c, _)| *c).collect();
    let end = Instant::now();
    let total_ms = start.map(|t0| end.duration_since(t0).as_secs_f32() * 1000.0);

    // Digraph dt(ms) derived from the retained character timestamps.
    // Note: if backspace is allowed, these timings can include correction overhead.
    let mut dts_ms: Vec<f32> = Vec::new();
    if buf.len() >= 2 {
        for i in 0..(buf.len() - 1) {
            let dt = buf[i + 1].1.duration_since(buf[i].1).as_secs_f32() * 1000.0;
            dts_ms.push(dt);
        }
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
        total_ms,
        backspaces: if backspaces == 0 {
            None
        } else {
            Some(backspaces)
        },
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
