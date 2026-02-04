use std::collections::HashMap;
use std::io::Write as _;

use crate::data::Row;

pub fn download_to_file(url: &str, out_path: &std::path::Path) -> anyhow::Result<()> {
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let resp = ureq::get(url).call()?;
    let mut r = resp.into_reader();
    let mut f = std::fs::File::create(out_path)?;
    std::io::copy(&mut r, &mut f)?;
    Ok(())
}

/// GREYC Web-based keystroke dynamics dataset (archived).
///
/// The archive contains per-session directories with:
/// - `password.txt` or `passphrase.txt` (string typed)
/// - `p_pp.txt` (keypress→keypress deltas for adjacent characters, in ms; length = len(text)-1)
pub mod greyc_web {
    use super::*;

    const DEFAULT_URL: &str =
        "https://web.archive.org/web/20181101124951/http://www.ecole.ensicaen.fr/~rosenber/pub/webkeystroke.tar.gz";

    pub fn default_url() -> &'static str {
        DEFAULT_URL
    }

    #[derive(Debug, Clone)]
    pub struct ImportConfig {
        pub include_passwords: bool,
        pub include_passphrases: bool,
        pub include_impostor: bool,
        pub max_rows: Option<usize>,
    }

    impl Default for ImportConfig {
        fn default() -> Self {
            Self {
                include_passwords: true,
                include_passphrases: true,
                include_impostor: true,
                max_rows: None,
            }
        }
    }

    pub fn write_jsonl_from_tar_gz_path(
        tar_gz_path: &std::path::Path,
        out_path: &std::path::Path,
        cfg: ImportConfig,
    ) -> anyhow::Result<usize> {
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let f = std::fs::File::open(tar_gz_path)?;
        let gz = flate2::read::GzDecoder::new(f);
        let mut ar = tar::Archive::new(gz);

        let mut out = std::io::BufWriter::new(std::fs::File::create(out_path)?);

        // We stream entries, assembling per-session bundles by directory prefix.
        #[derive(Default)]
        struct Bundle {
            text: Option<String>,
            p_pp: Option<Vec<f32>>,
            source: Option<String>,
            note: Option<String>,
        }
        let mut bundles: std::collections::HashMap<String, Bundle> = std::collections::HashMap::new();

        let mut rows_written = 0usize;

        for entry in ar.entries()? {
            let mut entry = entry?;
            let path = entry.path()?.to_string_lossy().to_string();
            if !path.starts_with("output_numpy/") {
                continue;
            }

            let is_password_tree = path.starts_with("output_numpy/passwords/");
            let is_passphrase_tree = path.starts_with("output_numpy/passphrases/");
            if is_password_tree && !cfg.include_passwords {
                continue;
            }
            if is_passphrase_tree && !cfg.include_passphrases {
                continue;
            }
            if !(is_password_tree || is_passphrase_tree) {
                continue;
            }
            if !cfg.include_impostor && path.contains("/impostor/") {
                continue;
            }

            // We key by session directory (everything up to the filename).
            let Some((dir, file)) = path.rsplit_once('/') else { continue };
            let key = dir.to_string();

            let b = bundles.entry(key.clone()).or_default();

            if file == "password.txt" {
                let mut s = String::new();
                std::io::Read::read_to_string(&mut entry, &mut s)?;
                b.text = Some(s.trim_end_matches(['\n', '\r']).to_string());
                b.source = Some(if is_password_tree {
                    "greyc_web_password".to_string()
                } else {
                    "greyc_web_passphrase".to_string()
                });
                b.note = Some(path.clone());
            } else if file == "p_pp.txt" {
                let mut s = String::new();
                std::io::Read::read_to_string(&mut entry, &mut s)?;
                let mut v = Vec::new();
                for line in s.lines() {
                    let t = line.trim();
                    if t.is_empty() {
                        continue;
                    }
                    let x: f32 = t.parse()?;
                    v.push(x);
                }
                b.p_pp = Some(v);
                b.source = Some(if is_password_tree {
                    "greyc_web_password".to_string()
                } else {
                    "greyc_web_passphrase".to_string()
                });
                b.note = Some(path.clone());
            } else {
                continue;
            }

            // If we now have both fields, emit.
            let ready = b.text.is_some() && b.p_pp.is_some();
            if !ready {
                continue;
            }
            let text = b.text.take().unwrap();
            let p_pp = b.p_pp.take().unwrap();

            // Validate.
            let grams = crate::score::graphemes_normalized(&text);
            if grams.len() < 2 {
                continue;
            }
            if p_pp.len() != grams.len() - 1 {
                continue;
            }
            if !p_pp.iter().all(|x| x.is_finite() && *x >= 0.0) {
                continue;
            }

            let row = Row {
                phrase: text,
                digraph_dt_ms: p_pp,
                source: b.source.clone(),
                note: b.note.clone(),
            };
            serde_json::to_writer(&mut out, &row)?;
            out.write_all(b"\n")?;
            rows_written += 1;

            if let Some(max) = cfg.max_rows {
                if rows_written >= max {
                    break;
                }
            }
        }

        out.flush()?;
        Ok(rows_written)
    }
}

/// Import a public keystroke-dynamics dataset as `phrasegen` JSONL rows.
///
/// Currently supported:
/// - CMU DSL-StrongPasswordData.csv (Killourhy & Maxion)
pub mod cmu_dsl {
    use super::*;

    /// The fixed password in the CMU DSL strong-password dataset.
    pub const PHRASE: &str = ".tie5Roanl";

    /// Adjacent digraph `DD.*.*` columns we extract, in order, corresponding to `PHRASE`.
    ///
    /// Notes:
    /// - Data is stored in seconds in the CMU table; we convert to milliseconds.
    /// - We intentionally drop the final `DD.l.Return` column, since `PHRASE` does not include
    ///   Return. This keeps the invariant `digraph_dt_ms.len() == phrase_len-1`.
    const DD_COLUMNS: [&str; 9] = [
        "DD.period.t",
        "DD.t.i",
        "DD.i.e",
        "DD.e.five",
        "DD.five.Shift.r",
        "DD.Shift.r.o",
        "DD.o.a",
        "DD.a.n",
        "DD.n.l",
    ];

    /// Download the CMU `DSL-StrongPasswordData.csv` file.
    pub fn download_csv(url: &str) -> anyhow::Result<Vec<u8>> {
        let resp = ureq::get(url).call()?;
        let mut r = resp.into_reader();
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut r, &mut buf)?;
        Ok(buf)
    }

    /// Parse the CMU CSV bytes into `Row`s.
    pub fn parse_csv_bytes(bytes: &[u8]) -> anyhow::Result<Vec<Row>> {
        let mut rdr = csv::Reader::from_reader(bytes);
        let headers = rdr.headers()?.clone();
        let idx = build_header_index(&headers);

        let mut missing = Vec::new();
        for &col in DD_COLUMNS.iter() {
            if !idx.contains_key(col) {
                missing.push(col);
            }
        }
        if !missing.is_empty() {
            anyhow::bail!("CMU DSL CSV is missing required columns: {missing:?}");
        }

        let mut out = Vec::new();
        for rec in rdr.records() {
            let rec = rec?;
            let mut digraph_dt_ms = Vec::with_capacity(DD_COLUMNS.len());
            for &col in DD_COLUMNS.iter() {
                let i = idx[col];
                let s = rec
                    .get(i)
                    .ok_or_else(|| anyhow::anyhow!("missing field for column {col}"))?;
                let v_s: f32 = s.parse()?;
                digraph_dt_ms.push(v_s * 1000.0);
            }
            out.push(Row {
                phrase: PHRASE.to_string(),
                digraph_dt_ms,
                source: Some("cmu_dsl".to_string()),
                note: None,
            });
        }
        Ok(out)
    }

    pub fn write_jsonl(rows: &[Row], out_path: &std::path::Path) -> anyhow::Result<()> {
        let mut f = std::fs::File::create(out_path)?;
        for row in rows {
            serde_json::to_writer(&mut f, row)?;
            f.write_all(b"\n")?;
        }
        Ok(())
    }

    fn build_header_index(headers: &csv::StringRecord) -> HashMap<String, usize> {
        headers
            .iter()
            .enumerate()
            .map(|(i, h)| (h.to_string(), i))
            .collect()
    }
}

/// CMU LASER-2012: “Free vs. Transcribed Text for Keystroke-Dynamics Evaluations”
///
/// This dataset provides *character-labeled* keydown-keydown (DD) features for many digraphs.
/// We import only digraphs where we can map both keys to a single-character grapheme
/// (e.g., letters, space, some punctuation).
pub mod cmu_laser2012 {
    use super::*;

    const DEFAULT_URL: &str = "https://www.cs.cmu.edu/~keystroke/laser-2012/DSL-Free-vs-Transcribed.zip";

    pub fn default_url() -> &'static str {
        DEFAULT_URL
    }

    pub fn parse_zip_bytes(zip_bytes: &[u8]) -> anyhow::Result<Vec<Row>> {
        let r = std::io::Cursor::new(zip_bytes);
        let mut z = zip::ZipArchive::new(r)?;

        let mut dd_text: Option<String> = None;
        let mut session_map_text: Option<String> = None;

        for i in 0..z.len() {
            let mut f = z.by_index(i)?;
            let name = f.name().to_string();
            if name.ends_with("data/TimingFeatures-DD.txt") {
                let mut s = String::new();
                std::io::Read::read_to_string(&mut f, &mut s)?;
                dd_text = Some(s);
            } else if name.ends_with("data/SessionMap.txt") {
                let mut s = String::new();
                std::io::Read::read_to_string(&mut f, &mut s)?;
                session_map_text = Some(s);
            }
        }

        let dd_text = dd_text.ok_or_else(|| anyhow::anyhow!("missing data/TimingFeatures-DD.txt in LASER zip"))?;
        let session_map_text =
            session_map_text.ok_or_else(|| anyhow::anyhow!("missing data/SessionMap.txt in LASER zip"))?;
        let sess_kind = parse_session_map(&session_map_text);

        let mut out = Vec::new();
        for (lineno, line) in dd_text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            // header
            if lineno == 0 && line.to_ascii_lowercase().contains("subject") && line.contains("key1") {
                continue;
            }
            let cols: Vec<&str> = line.split_whitespace().collect();
            if cols.len() < 7 {
                continue;
            }
            let subject = cols[0];
            let session_index: u32 = match cols[1].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let screen_index: u32 = match cols[2].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            // cols[3] is an index we don't need
            let key1 = cols[4];
            let key2 = cols[5];
            let time_s: f32 = match cols[6].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            if !time_s.is_finite() || time_s < 0.0 {
                continue;
            }
            let a = map_key_to_grapheme(key1);
            let b = map_key_to_grapheme(key2);
            let (Some(a), Some(b)) = (a, b) else {
                continue;
            };
            let phrase = format!("{a}{b}");
            let dt_ms = time_s * 1000.0;

            let kind = sess_kind
                .get(&(subject.to_string(), session_index))
                .map(|s| s.as_str())
                .unwrap_or("unknown");
            let source = format!("cmu_laser2012_{kind}");
            let note = format!("subject={subject} session={session_index} screen={screen_index}");
            out.push(Row {
                phrase,
                digraph_dt_ms: vec![dt_ms],
                source: Some(source),
                note: Some(note),
            });
        }

        Ok(out)
    }

    fn parse_session_map(text: &str) -> std::collections::HashMap<(String, u32), String> {
        // Lines look like:
        // s019  1 session.id "Free vs Transcribed - Runaway - Trans"
        let mut out = std::collections::HashMap::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with("user") {
                continue;
            }
            let mut it = line.split_whitespace();
            let Some(user) = it.next() else { continue };
            let Some(sess) = it.next().and_then(|s| s.parse::<u32>().ok()) else { continue };
            let rest = line.to_ascii_lowercase();
            let kind = if rest.contains(" - free") {
                "free"
            } else if rest.contains(" - trans") {
                "trans"
            } else {
                "unknown"
            };
            out.insert((user.to_string(), sess), kind.to_string());
        }
        out
    }

    fn map_key_to_grapheme(key: &str) -> Option<String> {
        let mut k = key.trim().to_ascii_lowercase();
        if let Some((_pfx, tail)) = k.rsplit_once('.') {
            // e.g. "shift.t" => "t"
            k = tail.to_string();
        }
        match k.as_str() {
            "space" => Some(" ".to_string()),
            "comma" => Some(",".to_string()),
            "period" => Some(".".to_string()),
            "apostrophe" => Some("'".to_string()),
            "semicolon" => Some(";".to_string()),
            "minus" => Some("-".to_string()),
            "equal" => Some("=".to_string()),
            "slash" => Some("/".to_string()),
            "backslash" => Some("\\".to_string()),
            _ => {
                // Single visible ASCII character.
                if k.len() == 1 {
                    Some(k)
                } else {
                    None
                }
            }
        }
    }
}

/// KeyRecs (Zenodo, CC-BY 4.0) keystroke dynamics dataset.
///
/// KeyRecs provides a free-text file with character-labeled digraph features, including
/// a `DD.key1.key2` column.
pub mod keyrecs {
    use super::*;

    const FREE_TEXT_URL: &str = "https://zenodo.org/records/7886743/files/free-text.csv?download=1";
    const FIXED_TEXT_URL: &str = "https://zenodo.org/records/7886743/files/fixed-text.csv?download=1";

    pub fn free_text_url() -> &'static str {
        FREE_TEXT_URL
    }

    pub fn fixed_text_url() -> &'static str {
        FIXED_TEXT_URL
    }

    pub fn parse_free_text_csv_bytes(bytes: &[u8]) -> anyhow::Result<Vec<Row>> {
        let mut rdr = csv::Reader::from_reader(bytes);
        let headers = rdr.headers()?.clone();
        let mut idx = std::collections::HashMap::<String, usize>::new();
        for (i, h) in headers.iter().enumerate() {
            idx.insert(h.trim().to_string(), i);
        }

        fn get<'a>(
            rec: &'a csv::StringRecord,
            idx: &std::collections::HashMap<String, usize>,
            col: &str,
        ) -> Option<&'a str> {
            idx.get(col).and_then(|&i| rec.get(i)).map(|s| s.trim())
        }

        let mut out = Vec::new();
        for rec in rdr.records() {
            let rec = rec?;
            let participant = match get(&rec, &idx, "participant") {
                Some(s) if !s.is_empty() => s,
                _ => continue,
            };
            let session: u32 = match get(&rec, &idx, "session").and_then(|s| s.parse::<u32>().ok()) {
                Some(v) => v,
                None => continue,
            };
            let key1 = match get(&rec, &idx, "key1") {
                Some(s) if !s.is_empty() => s,
                _ => continue,
            };
            let key2 = match get(&rec, &idx, "key2") {
                Some(s) if !s.is_empty() => s,
                _ => continue,
            };
            let dd_s: f32 = match get(&rec, &idx, "DD.key1.key2").and_then(|s| s.parse::<f32>().ok()) {
                Some(v) => v,
                None => continue,
            };
            if !dd_s.is_finite() || dd_s < 0.0 {
                // Some rows appear to encode non-physical ordering (e.g., Shift interactions) and can be negative.
                continue;
            }
            let a = map_key_to_grapheme(key1);
            let b = map_key_to_grapheme(key2);
            let (Some(a), Some(b)) = (a, b) else {
                continue;
            };

            // KeyRecs appears to store times in seconds; convert to ms.
            let dt_ms = dd_s * 1000.0;
            out.push(Row {
                phrase: format!("{a}{b}"),
                digraph_dt_ms: vec![dt_ms],
                source: Some("keyrecs_free_text".to_string()),
                note: Some(format!("participant={participant} session={session}")),
            });
        }
        Ok(out)
    }

    fn map_key_to_grapheme(key: &str) -> Option<String> {
        let k = key.trim();
        if k.is_empty() {
            return None;
        }
        // KeyRecs uses "Space" and "Shift" capitalized.
        let kl = k.to_ascii_lowercase();
        match kl.as_str() {
            "space" => Some(" ".to_string()),
            "comma" => Some(",".to_string()),
            "period" => Some(".".to_string()),
            "apostrophe" => Some("'".to_string()),
            "semicolon" => Some(";".to_string()),
            "minus" => Some("-".to_string()),
            "equal" => Some("=".to_string()),
            "slash" => Some("/".to_string()),
            "backslash" => Some("\\".to_string()),
            _ => {
                // Single character keys.
                if kl.len() == 1 {
                    Some(kl)
                } else {
                    None
                }
            }
        }
    }
}

/// BKSD (Bilingual Keystroke Dynamics Dataset) import helpers.
///
/// BKSD provides positional timing features (DD.1_2, DD.2_3, ...), not character-labeled digraphs.
/// We still import the DD timings into `Row` by using a synthetic phrase made of private-use chars,
/// so the digraph count matches `phrase_len-1`. This contributes to a better global mean time and
/// preserves position-specific digraph means (though those won’t match real words).
pub mod bksd {
    use super::*;

    pub fn parse_csv_bytes(bytes: &[u8], source: &str, note: Option<String>) -> anyhow::Result<Vec<Row>> {
        let mut rdr = csv::Reader::from_reader(bytes);
        let headers = rdr.headers()?.clone();
        let dd_cols = dd_columns_in_order(&headers)?;

        if dd_cols.is_empty() {
            anyhow::bail!("BKSD CSV: no DD.* columns found");
        }

        let mut out = Vec::new();
        for rec in rdr.records() {
            let rec = rec?;
            let mut digraph_dt_ms = Vec::with_capacity(dd_cols.len());
            for &idx in dd_cols.iter() {
                let s = rec
                    .get(idx)
                    .ok_or_else(|| anyhow::anyhow!("missing field at index {idx}"))?;
                let v: f32 = s.parse()?;
                digraph_dt_ms.push(v);
            }

            let phrase = synthetic_phrase(dd_cols.len() + 1)?;
            out.push(Row {
                phrase,
                digraph_dt_ms,
                source: Some(source.to_string()),
                note: note.clone(),
            });
        }
        Ok(out)
    }

    pub fn parse_zip_bytes(zip_bytes: &[u8], source_prefix: &str) -> anyhow::Result<Vec<Row>> {
        let mut all = Vec::new();
        let r = std::io::Cursor::new(zip_bytes);
        let mut z = zip::ZipArchive::new(r)?;
        for i in 0..z.len() {
            let mut f = z.by_index(i)?;
            if !f.name().to_ascii_lowercase().ends_with(".csv") {
                continue;
            }
            let mut buf = Vec::new();
            std::io::Read::read_to_end(&mut f, &mut buf)?;
            let note = Some(f.name().to_string());
            let rows = parse_csv_bytes(&buf, source_prefix, note)?;
            all.extend(rows);
        }
        Ok(all)
    }

    fn dd_columns_in_order(headers: &csv::StringRecord) -> anyhow::Result<Vec<usize>> {
        // Find header fields like "DD.1_2", "DD.10_11", etc. and sort by the left index.
        let mut cols: Vec<(usize, usize)> = Vec::new(); // (i_left, col_idx)
        for (idx, h) in headers.iter().enumerate() {
            let Some(rest) = h.strip_prefix("DD.") else { continue };
            let Some((a, _b)) = rest.split_once('_') else { continue };
            let Ok(i_left) = a.parse::<usize>() else { continue };
            cols.push((i_left, idx));
        }
        cols.sort_by_key(|(i_left, _)| *i_left);
        Ok(cols.into_iter().map(|(_, idx)| idx).collect())
    }

    fn synthetic_phrase(len: usize) -> anyhow::Result<String> {
        // Use Private Use Area chars to create len distinct “characters”.
        // U+E000..U+F8FF gives 6400 code points.
        if len == 0 {
            return Ok(String::new());
        }
        if len > 6400 {
            anyhow::bail!("cannot synthesize phrase of length {len}: exceeds private-use range");
        }
        let mut s = String::new();
        for i in 0..len {
            let cp = 0xE000u32 + (i as u32);
            let ch = char::from_u32(cp).ok_or_else(|| anyhow::anyhow!("invalid char at {cp:x}"))?;
            s.push(ch);
        }
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::cmu_dsl;
    use super::bksd;

    #[test]
    fn cmu_parse_smoke() {
        // Minimal synthetic CSV with only the required columns (and a couple leading ones).
        let csv = [
            "subject,sessionIndex,rep,DD.period.t,DD.t.i,DD.i.e,DD.e.five,DD.five.Shift.r,DD.Shift.r.o,DD.o.a,DD.a.n,DD.n.l",
            "s001,1,1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        ]
        .join("\n");
        let rows = cmu_dsl::parse_csv_bytes(csv.as_bytes()).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].phrase, cmu_dsl::PHRASE);
        assert_eq!(rows[0].digraph_dt_ms.len(), 9);
        assert!((rows[0].digraph_dt_ms[0] - 100.0).abs() < 1e-6);
    }

    #[test]
    fn bksd_parse_smoke() {
        let csv = [
            "H.1,DD.1_2,UD.1_2,H.2,DD.2_3,UD.2_3,total,user",
            "10,100,90,20,200,180,999,foo",
        ]
        .join("\n");
        let rows = bksd::parse_csv_bytes(csv.as_bytes(), "bksd_phrase_en", Some("S65e-ksd.csv".to_string())).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].digraph_dt_ms.len(), 2);
        assert_eq!(rows[0].digraph_dt_ms[0], 100.0);
        assert_eq!(rows[0].digraph_dt_ms[1], 200.0);
        assert_eq!(rows[0].source.as_deref(), Some("bksd_phrase_en"));
        assert_eq!(rows[0].note.as_deref(), Some("S65e-ksd.csv"));
        // synthetic phrase length must be digraphs+1
        assert_eq!(rows[0].phrase.chars().count(), 3);
    }
}

