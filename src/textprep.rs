//! Small text preprocessing helpers.
//!
//! We intentionally keep this crate self-contained (CI-friendly) rather than depending on a
//! sibling `textprep/` repo in your super-workspace.

use unicode_normalization::UnicodeNormalization as _;

/// Scrub a token into a typing-friendly word.
///
/// Current policy:
/// - NFKD normalize
/// - drop combining marks (diacritics)
/// - lowercase
/// - keep ASCII letters + a small set of separators (space, hyphen, underscore, apostrophe)
/// - collapse internal whitespace and trim
pub fn scrub(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = false;
    for ch in s.nfkd() {
        // Strip diacritics/combining marks.
        if is_combining_mark(ch) {
            continue;
        }
        for lo in ch.to_lowercase() {
            let keep = match lo {
                'a'..='z' => Some(lo),
                ' ' | '-' | '_' | '\'' => Some(lo),
                // common “smart apostrophe” -> ASCII apostrophe
                '’' => Some('\''),
                _ => None,
            };
            let Some(k) = keep else { continue };
            if k == ' ' {
                if out.is_empty() || prev_space {
                    continue;
                }
                prev_space = true;
                out.push(' ');
                continue;
            }
            prev_space = false;
            out.push(k);
        }
    }
    out.trim().to_string()
}

fn is_combining_mark(ch: char) -> bool {
    // Broadly covers Mn/Mc/Me (good enough for scrub; avoids pulling in a large unicode table crate).
    matches!(ch as u32, 0x0300..=0x036F | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x20D0..=0x20FF | 0xFE20..=0xFE2F)
}
