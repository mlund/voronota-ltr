// Copyright (c) 2026 Kliment Olechnovic and Mikael Lund
// Part of the voronota-ltr project, licensed under the MIT License.
// SPDX-License-Identifier: MIT

//! PDB file format parser.
//!
//! Parses ATOM/HETATM records from PDB files with filtering for hydrogens, heteroatoms, and altLoc.

use std::io::BufRead;

use super::{AtomRecord, ParseOptions};

/// Extract substring from fixed-width PDB columns (1-indexed, inclusive).
fn extract_column(line: &str, start: usize, end: usize) -> &str {
    let bytes = line.as_bytes();
    let len = bytes.len();
    let start_idx = start.saturating_sub(1);
    let end_idx = end.min(len);

    if start_idx >= len {
        return "";
    }

    line.get(start_idx..end_idx).unwrap_or("").trim()
}

/// Parse a PDB line float value, returning None if invalid.
fn parse_column_f64(line: &str, start: usize, end: usize) -> Option<f64> {
    let s = extract_column(line, start, end);
    if s.is_empty() {
        return None;
    }
    s.parse().ok()
}

/// Parse a PDB line integer value, returning None if invalid.
fn parse_column_i32(line: &str, start: usize, end: usize) -> Option<i32> {
    let s = extract_column(line, start, end);
    if s.is_empty() {
        return None;
    }
    s.parse().ok()
}

/// Move leading digits to end of atom name (e.g., "1CA" → "CA1").
fn normalize_atom_name(name: &str) -> String {
    if name.is_empty() {
        return String::new();
    }

    let first_letter = name.find(|c: char| !c.is_ascii_digit());
    match first_letter {
        Some(0) | None => name.to_string(),
        Some(pos) => format!("{}{}", &name[pos..], &name[..pos]),
    }
}

/// Fix undefined string values ("." or "?" become empty).
fn fix_undefined(s: &str) -> String {
    if s == "." || s == "?" {
        String::new()
    } else {
        s.to_string()
    }
}

/// Parse a single ATOM/HETATM line into an `AtomRecord`.
fn parse_atom_line(line: &str) -> Option<AtomRecord> {
    let record_name = extract_column(line, 1, 6);
    if record_name != "ATOM" && record_name != "HETATM" {
        return None;
    }

    // Coordinates are required
    let x = parse_column_f64(line, 31, 38)?;
    let y = parse_column_f64(line, 39, 46)?;
    let z = parse_column_f64(line, 47, 54)?;

    let serial = parse_column_i32(line, 7, 11)?;
    let res_seq = parse_column_i32(line, 23, 26)?;

    let raw_name = extract_column(line, 13, 16);
    if raw_name.is_empty() {
        return None;
    }

    Some(AtomRecord {
        record_name: record_name.to_string(),
        serial,
        name: normalize_atom_name(raw_name),
        alt_loc: fix_undefined(extract_column(line, 17, 17)),
        res_name: fix_undefined(extract_column(line, 18, 20)),
        chain_id: fix_undefined(extract_column(line, 22, 22)),
        res_seq,
        i_code: fix_undefined(extract_column(line, 27, 27)),
        x,
        y,
        z,
        element: fix_undefined(extract_column(line, 77, 78)),
    })
}

/// Check if atom is a hydrogen based on name or element.
fn is_hydrogen(record: &AtomRecord) -> bool {
    record.name.starts_with('H') || record.element == "H" || record.element == "D"
}

/// Check if atom should be accepted based on filter options.
fn is_acceptable(record: &AtomRecord, options: &ParseOptions) -> bool {
    // HETATM filtering
    if record.record_name == "HETATM" && options.exclude_heteroatoms {
        return false;
    }

    // AltLoc filtering: only accept empty, A, 1, or .
    if !record.alt_loc.is_empty()
        && record.alt_loc != "A"
        && record.alt_loc != "1"
        && record.alt_loc != "."
    {
        return false;
    }

    // Hydrogen filtering
    if !options.include_hydrogens && is_hydrogen(record) {
        return false;
    }

    // Always exclude water
    if record.res_name == "HOH" {
        return false;
    }

    true
}

/// Parse PDB format from a buffered reader.
pub fn parse_pdb<R: BufRead>(reader: R, options: &ParseOptions) -> Vec<AtomRecord> {
    let mut records = Vec::new();
    let mut model_num = 1;

    for line in reader.lines().map_while(Result::ok) {
        let record_name = extract_column(&line, 1, 6);

        match record_name {
            "ATOM" | "HETATM" => {
                if let Some(mut record) = parse_atom_line(&line)
                    && is_acceptable(&record, options)
                {
                    // Append model number to chain ID for multi-model assemblies
                    if options.as_assembly && model_num > 1 {
                        record.chain_id = format!("{}{}", record.chain_id, model_num);
                    }
                    record.alt_loc.clear();
                    records.push(record);
                }
            }
            "ENDMDL" => {
                model_num += 1;
                if !options.as_assembly {
                    break;
                }
            }
            "END" => break,
            _ => {}
        }
    }

    records
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_PDB: &str = "\
ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  MET A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   MET A   1       2.009   1.420   0.000  1.00  0.00           C
HETATM   99  ZN  ZN  A 100       5.000   5.000   5.000  1.00  0.00          ZN
END
";

    #[test]
    fn parse_atom_records() {
        let options = ParseOptions::default();
        let records = parse_pdb(SAMPLE_PDB.as_bytes(), &options);
        assert_eq!(records.len(), 4); // HETATM included by default (matching C++)
        assert_eq!(records[0].name, "N");
        assert_eq!(records[0].res_name, "MET");
        assert_eq!(records[1].name, "CA");
        assert_eq!(records[3].name, "ZN");
    }

    #[test]
    fn exclude_heteroatoms() {
        let options = ParseOptions {
            exclude_heteroatoms: true,
            ..Default::default()
        };
        let records = parse_pdb(SAMPLE_PDB.as_bytes(), &options);
        assert_eq!(records.len(), 3);
        assert_eq!(records[2].name, "C");
    }

    #[test]
    fn normalize_numbered_atom() {
        assert_eq!(normalize_atom_name("1HG2"), "HG21");
        assert_eq!(normalize_atom_name("CA"), "CA");
        assert_eq!(normalize_atom_name("2HD1"), "HD12");
    }

    #[test]
    fn column_extraction() {
        let line = "ATOM      1  CA  ALA A   1       1.000   2.000   3.000";
        assert_eq!(extract_column(line, 1, 6), "ATOM");
        assert_eq!(extract_column(line, 13, 16), "CA");
        assert_eq!(extract_column(line, 18, 20), "ALA");
    }
}
