// Copyright (c) 2026 Kliment Olechnovic and Mikael Lund
// Part of the voronota-ltr project, licensed under the MIT License.
// SPDX-License-Identifier: MIT

//! mmCIF file format parser.
//!
//! Parses `_atom_site` loop blocks from mmCIF files with filtering for hydrogens, heteroatoms, and altLoc.

use std::collections::HashMap;
use std::io::BufRead;

use super::{AtomRecord, ParseOptions};

/// Field names we extract from `_atom_site` loop.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Field {
    ModelNum,
    GroupPdb,
    Id,
    AuthAtomId,
    LabelAtomId,
    LabelAltId,
    AuthCompId,
    LabelCompId,
    AuthAsymId,
    LabelAsymId,
    AuthSeqId,
    LabelSeqId,
    InsCode,
    CartnX,
    CartnY,
    CartnZ,
    TypeSymbol,
}

/// Map mmCIF field names to our Field enum.
fn field_from_name(name: &str) -> Option<Field> {
    match name {
        "pdbx_PDB_model_num" => Some(Field::ModelNum),
        "group_PDB" => Some(Field::GroupPdb),
        "id" => Some(Field::Id),
        "auth_atom_id" => Some(Field::AuthAtomId),
        "label_atom_id" => Some(Field::LabelAtomId),
        "label_alt_id" => Some(Field::LabelAltId),
        "auth_comp_id" => Some(Field::AuthCompId),
        "label_comp_id" => Some(Field::LabelCompId),
        "auth_asym_id" => Some(Field::AuthAsymId),
        "label_asym_id" => Some(Field::LabelAsymId),
        "auth_seq_id" => Some(Field::AuthSeqId),
        "label_seq_id" => Some(Field::LabelSeqId),
        "pdbx_PDB_ins_code" => Some(Field::InsCode),
        "Cartn_x" => Some(Field::CartnX),
        "Cartn_y" => Some(Field::CartnY),
        "Cartn_z" => Some(Field::CartnZ),
        "type_symbol" => Some(Field::TypeSymbol),
        _ => None,
    }
}

/// Token reader for mmCIF format (handles quotes and comments).
struct Tokenizer<R> {
    reader: R,
    buffer: String,
    position: usize,
    eof: bool,
}

impl<R: BufRead> Tokenizer<R> {
    const fn new(reader: R) -> Self {
        Self {
            reader,
            buffer: String::new(),
            position: 0,
            eof: false,
        }
    }

    /// Refill buffer if needed.
    fn ensure_data(&mut self) -> bool {
        while self.position >= self.buffer.len() && !self.eof {
            self.buffer.clear();
            self.position = 0;
            match self.reader.read_line(&mut self.buffer) {
                Ok(0) | Err(_) => {
                    self.eof = true;
                    return false;
                }
                Ok(_) => {}
            }
        }
        !self.eof || self.position < self.buffer.len()
    }

    /// Skip whitespace and comments.
    fn skip_whitespace_and_comments(&mut self) {
        loop {
            if !self.ensure_data() {
                return;
            }

            let bytes = self.buffer.as_bytes();
            while self.position < bytes.len() && bytes[self.position].is_ascii_whitespace() {
                self.position += 1;
            }

            if self.position >= bytes.len() {
                continue;
            }

            // Skip comment lines
            if bytes[self.position] == b'#' {
                self.position = bytes.len();
                continue;
            }

            break;
        }
    }

    /// Read next token, returning None at EOF.
    fn next_token(&mut self) -> Option<String> {
        self.skip_whitespace_and_comments();

        if !self.ensure_data() {
            return None;
        }

        let bytes = self.buffer.as_bytes();
        if self.position >= bytes.len() {
            return None;
        }

        let ch = bytes[self.position];

        // Handle quoted strings
        if ch == b'\'' || ch == b'"' {
            let quote = ch;
            self.position += 1;
            let start = self.position;

            while self.position < bytes.len() && bytes[self.position] != quote {
                self.position += 1;
            }

            let token = String::from_utf8_lossy(&bytes[start..self.position]).to_string();

            if self.position < bytes.len() {
                self.position += 1; // Skip closing quote
            }

            return Some(token);
        }

        // Handle regular tokens (non-whitespace)
        let start = self.position;
        while self.position < bytes.len() && !bytes[self.position].is_ascii_whitespace() {
            self.position += 1;
        }

        Some(String::from_utf8_lossy(&bytes[start..self.position]).to_string())
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

/// Get value from row with fallback for auth_* vs label_* fields.
fn get_value<'a>(
    row: &'a [String],
    field_map: &HashMap<Field, usize>,
    primary: Field,
    fallback: Option<Field>,
) -> &'a str {
    if let Some(&idx) = field_map.get(&primary)
        && idx < row.len()
    {
        return &row[idx];
    }
    if let Some(fb) = fallback
        && let Some(&idx) = field_map.get(&fb)
        && idx < row.len()
    {
        return &row[idx];
    }
    ""
}

/// Parse an atom record from a row of values.
fn parse_atom_row(row: &[String], field_map: &HashMap<Field, usize>) -> Option<AtomRecord> {
    let x: f64 = get_value(row, field_map, Field::CartnX, None)
        .parse()
        .ok()?;
    let y: f64 = get_value(row, field_map, Field::CartnY, None)
        .parse()
        .ok()?;
    let z: f64 = get_value(row, field_map, Field::CartnZ, None)
        .parse()
        .ok()?;

    let serial: i32 = get_value(row, field_map, Field::Id, None).parse().ok()?;
    let res_seq: i32 = get_value(row, field_map, Field::AuthSeqId, Some(Field::LabelSeqId))
        .parse()
        .ok()?;

    let raw_name = get_value(row, field_map, Field::AuthAtomId, Some(Field::LabelAtomId));
    if raw_name.is_empty() {
        return None;
    }

    let record_name = get_value(row, field_map, Field::GroupPdb, None);
    if record_name.is_empty() {
        return None;
    }

    Some(AtomRecord {
        record_name: record_name.to_string(),
        serial,
        name: normalize_atom_name(raw_name),
        alt_loc: fix_undefined(get_value(row, field_map, Field::LabelAltId, None)),
        res_name: fix_undefined(get_value(
            row,
            field_map,
            Field::AuthCompId,
            Some(Field::LabelCompId),
        )),
        chain_id: fix_undefined(get_value(
            row,
            field_map,
            Field::AuthAsymId,
            Some(Field::LabelAsymId),
        )),
        res_seq,
        i_code: fix_undefined(get_value(row, field_map, Field::InsCode, None)),
        x,
        y,
        z,
        element: fix_undefined(get_value(row, field_map, Field::TypeSymbol, None)),
    })
}

/// Check if atom is a hydrogen based on name or element.
fn is_hydrogen(record: &AtomRecord) -> bool {
    record.name.starts_with('H') || record.element == "H" || record.element == "D"
}

/// Check if atom should be accepted based on filter options.
fn is_acceptable(record: &AtomRecord, options: &ParseOptions) -> bool {
    if record.record_name == "HETATM" && options.exclude_heteroatoms {
        return false;
    }

    if !record.alt_loc.is_empty()
        && record.alt_loc != "A"
        && record.alt_loc != "1"
        && record.alt_loc != "."
    {
        return false;
    }

    if !options.include_hydrogens && is_hydrogen(record) {
        return false;
    }

    if record.res_name == "HOH" {
        return false;
    }

    true
}

/// Parse mmCIF format from a buffered reader.
pub fn parse_mmcif<R: BufRead>(reader: R, options: &ParseOptions) -> Vec<AtomRecord> {
    let mut tokenizer = Tokenizer::new(reader);
    let mut records = Vec::new();

    while let Some(token) = tokenizer.next_token() {
        if token != "loop_" {
            continue;
        }

        // Collect header fields
        let mut header = Vec::new();
        let mut field_map = HashMap::new();

        while let Some(tok) = tokenizer.next_token() {
            if !tok.starts_with("_atom_site.") {
                // Put back token for data reading
                if !header.is_empty() {
                    read_atom_site_data(
                        &mut tokenizer,
                        &field_map,
                        header.len(),
                        &tok,
                        options,
                        &mut records,
                    );
                }
                break;
            }

            // Extract field name after "_atom_site."
            let field_name = &tok[11..];
            if let Some(field) = field_from_name(field_name) {
                field_map.insert(field, header.len());
            }
            header.push(tok);
        }
    }

    records
}

/// Read atom site data rows.
fn read_atom_site_data<R: BufRead>(
    tokenizer: &mut Tokenizer<R>,
    field_map: &HashMap<Field, usize>,
    num_cols: usize,
    first_token: &str,
    options: &ParseOptions,
    records: &mut Vec<AtomRecord>,
) {
    let mut first_model_id: Option<String> = None;
    let mut row = Vec::with_capacity(num_cols);
    row.push(first_token.to_string());

    loop {
        // Complete current row
        while row.len() < num_cols {
            match tokenizer.next_token() {
                Some(tok) => row.push(tok),
                None => return,
            }
        }

        // Check for end of loop (new category or data block)
        if row[0].starts_with('_') || row[0].starts_with("data_") || row[0] == "loop_" {
            return;
        }

        // Get model ID for filtering
        let model_id = get_value(&row, field_map, Field::ModelNum, None).to_string();

        // Store first model ID
        if first_model_id.is_none() {
            first_model_id = Some(model_id.clone());
        }

        // Only process first model unless as_assembly
        let should_process =
            options.as_assembly || first_model_id.as_ref().is_some_and(|fm| fm == &model_id);

        if should_process
            && let Some(mut record) = parse_atom_row(&row, field_map)
            && is_acceptable(&record, options)
        {
            // Append model number for assembly mode
            if options.as_assembly && model_id != "1" && !model_id.is_empty() {
                record.chain_id = format!("{}{}", record.chain_id, model_id);
            }
            record.alt_loc.clear();
            records.push(record);
        }

        // Start next row
        row.clear();
        match tokenizer.next_token() {
            Some(tok) => row.push(tok),
            None => return,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_MMCIF: &str = r#"
data_test
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.auth_atom_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.type_symbol
_atom_site.label_alt_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N MET A 1 0.000 0.000 0.000 N . 1
ATOM 2 CA MET A 1 1.458 0.000 0.000 C . 1
HETATM 99 ZN ZN A 100 5.000 5.000 5.000 ZN . 1
"#;

    #[test]
    fn parse_atom_records() {
        let options = ParseOptions::default();
        let records = parse_mmcif(SAMPLE_MMCIF.as_bytes(), &options);
        assert_eq!(records.len(), 3); // HETATM included by default (matching C++)
        assert_eq!(records[0].name, "N");
        assert_eq!(records[0].res_name, "MET");
        assert_eq!(records[2].name, "ZN");
    }

    #[test]
    fn exclude_heteroatoms() {
        let options = ParseOptions {
            exclude_heteroatoms: true,
            ..Default::default()
        };
        let records = parse_mmcif(SAMPLE_MMCIF.as_bytes(), &options);
        assert_eq!(records.len(), 2);
        assert_eq!(records[1].name, "CA");
    }

    #[test]
    fn quoted_values() {
        let cif = r#"
data_test
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.auth_atom_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.type_symbol
_atom_site.label_alt_id
_atom_site.pdbx_PDB_model_num
ATOM 1 "C'" ALA A 1 1.0 2.0 3.0 C . 1
"#;
        let options = ParseOptions::default();
        let records = parse_mmcif(cif.as_bytes(), &options);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].name, "C'");
    }
}
