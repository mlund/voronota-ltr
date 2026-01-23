// Copyright (c) 2026 Kliment Olechnovic and Mikael Lund
// Part of the voronota-ltr project, licensed under the MIT License.
// SPDX-License-Identifier: MIT

//! VMD-like atom selection language for defining atom groups.
//!
//! Supports boolean expressions with `and`, `or`, `not`, parentheses,
//! and keywords like `chain`, `resname`, `resid`, `name`, `protein`, etc.

use log::{debug, info};

use super::AtomRecord;

/// Selection parsing error.
#[derive(Debug, Clone)]
pub struct SelectionError {
    /// Error message.
    pub message: String,
    /// Position in input where error occurred.
    pub position: usize,
}

impl std::fmt::Display for SelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at position {}", self.message, self.position)
    }
}

impl std::error::Error for SelectionError {}

/// A glob pattern for matching strings (supports `*`, `?`, `[abc]`).
#[derive(Debug, Clone)]
pub struct GlobPattern(String);

impl GlobPattern {
    /// Create a new glob pattern.
    #[must_use]
    pub fn new(pattern: &str) -> Self {
        Self(pattern.to_string())
    }

    /// Test if text matches the pattern.
    #[must_use]
    pub fn matches(&self, text: &str) -> bool {
        glob_match(&self.0, text)
    }
}

/// Match text against a glob pattern (supports `*`, `?`, `[abc]`).
#[allow(clippy::similar_names)]
fn glob_match(pattern: &str, text: &str) -> bool {
    // For backtracking on * - track where the star was in pattern and text
    let mut star_pattern_pos: Option<usize> = None;
    let mut star_text_pos: Option<usize> = None;
    let mut pat_idx = 0;
    let mut txt_idx = 0;

    let pattern: Vec<char> = pattern.chars().collect();
    let text: Vec<char> = text.chars().collect();

    while txt_idx < text.len() {
        if pat_idx < pattern.len() {
            match pattern[pat_idx] {
                '?' => {
                    // Match any single character
                    pat_idx += 1;
                    txt_idx += 1;
                    continue;
                }
                '*' => {
                    // Match zero or more characters
                    star_pattern_pos = Some(pat_idx);
                    star_text_pos = Some(txt_idx);
                    pat_idx += 1;
                    continue;
                }
                '[' => {
                    // Character class - advance if matched, else fall through to backtrack
                    if let Some((true, end_idx)) =
                        match_char_class(&pattern, pat_idx, text[txt_idx])
                    {
                        pat_idx = end_idx + 1;
                        txt_idx += 1;
                        continue;
                    }
                }
                c if c == text[txt_idx] => {
                    pat_idx += 1;
                    txt_idx += 1;
                    continue;
                }
                _ => {}
            }
        }

        // Backtrack to last * if possible
        if let (Some(sp), Some(st)) = (star_pattern_pos, star_text_pos) {
            pat_idx = sp + 1;
            star_text_pos = Some(st + 1);
            txt_idx = st + 1;
            if txt_idx > text.len() {
                return false;
            }
        } else {
            return false;
        }
    }

    // Skip trailing *
    while pat_idx < pattern.len() && pattern[pat_idx] == '*' {
        pat_idx += 1;
    }

    pat_idx == pattern.len()
}

/// Match a character class like `[abc]` or `[a-z]`.
/// Returns `(matched, end_index)` where `end_index` is position of `]`.
fn match_char_class(pattern: &[char], start: usize, c: char) -> Option<(bool, usize)> {
    if pattern.get(start) != Some(&'[') {
        return None;
    }

    let mut i = start + 1;
    let mut matched = false;

    while i < pattern.len() && pattern[i] != ']' {
        if i + 2 < pattern.len() && pattern[i + 1] == '-' && pattern[i + 2] != ']' {
            // Range like a-z
            let range_start = pattern[i];
            let range_end = pattern[i + 2];
            if c >= range_start && c <= range_end {
                matched = true;
            }
            i += 3;
        } else {
            if pattern[i] == c {
                matched = true;
            }
            i += 1;
        }
    }

    if i < pattern.len() && pattern[i] == ']' {
        Some((matched, i))
    } else {
        None
    }
}

/// Standard protein residue names.
const PROTEIN_RESIDUES: &[&str] = &[
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "HIE", "HID", "HIP", "CYX", "ASH", "GLH",
    "LYN",
];

/// DNA residue names.
const DNA_RESIDUES: &[&str] = &["DA", "DT", "DG", "DC", "DU"];

/// RNA residue names.
const RNA_RESIDUES: &[&str] = &["A", "U", "G", "C", "RA", "RU", "RG", "RC"];

/// Backbone atom names.
const BACKBONE_ATOMS: &[&str] = &["C", "CA", "N", "O"];

/// Hydrophobic residues (nonpolar sidechains).
const HYDROPHOBIC_RESIDUES: &[&str] = &[
    "ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO", "GLY",
];

/// Aromatic residues.
const AROMATIC_RESIDUES: &[&str] = &["PHE", "TYR", "TRP", "HIS", "HIE", "HID", "HIP"];

/// Acidic (negatively charged) residues.
const ACIDIC_RESIDUES: &[&str] = &["ASP", "GLU", "ASH", "GLH"];

/// Basic (positively charged) residues.
const BASIC_RESIDUES: &[&str] = &["ARG", "LYS", "HIS", "HIE", "HID", "HIP", "LYN"];

/// Polar (uncharged) residues.
const POLAR_RESIDUES: &[&str] = &["SER", "THR", "ASN", "GLN", "CYS", "CYX", "TYR"];

/// Check if residue name is in a set.
fn resname_in(resname: &str, set: &[&str]) -> bool {
    set.contains(&resname)
}

/// Token type for lexer.
#[derive(Debug, Clone, PartialEq)]
enum Token {
    // Keywords
    Chain,
    Resname,
    Resid,
    Name,
    Protein,
    Backbone,
    Sidechain,
    Nucleic,
    Hetatm,
    Hydrophobic,
    Aromatic,
    Acidic,
    Basic,
    Polar,
    Charged,
    All,
    None,
    // Boolean ops
    And,
    Or,
    Not,
    // Grouping
    LParen,
    RParen,
    // Range separator
    To,
    // Values
    Ident(String),
    Number(i32),
    Colon,
}

/// Convert identifier string to keyword token or leave as identifier.
fn ident_to_token(ident: String) -> Token {
    match ident.to_lowercase().as_str() {
        "chain" | "segid" => Token::Chain,
        "resname" | "resn" => Token::Resname,
        "resid" | "resi" | "resseq" | "resnum" => Token::Resid,
        "name" | "atomname" => Token::Name,
        "protein" => Token::Protein,
        "backbone" => Token::Backbone,
        "sidechain" => Token::Sidechain,
        "nucleic" | "nucleicacid" => Token::Nucleic,
        "hetatm" | "hetero" => Token::Hetatm,
        "hydrophobic" => Token::Hydrophobic,
        "aromatic" => Token::Aromatic,
        "acidic" => Token::Acidic,
        "basic" => Token::Basic,
        "polar" => Token::Polar,
        "charged" => Token::Charged,
        "all" | "everything" => Token::All,
        "none" | "nothing" => Token::None,
        "and" | "&&" => Token::And,
        "or" | "||" => Token::Or,
        "not" | "!" => Token::Not,
        "to" => Token::To,
        _ => Token::Ident(ident),
    }
}

/// Check if character can start an identifier.
const fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '*' | '?' | '[' | '\'')
}

/// Check if character can continue an identifier.
const fn is_ident_char(c: char, in_bracket: bool) -> bool {
    in_bracket || c.is_ascii_alphanumeric() || matches!(c, '_' | '*' | '?' | '[' | ']' | '-' | '\'')
}

/// Tokenize input string.
#[allow(clippy::too_many_lines)]
fn tokenize(input: &str) -> Result<Vec<(Token, usize)>, SelectionError> {
    let mut tokens = Vec::new();
    let mut chars = input.char_indices().peekable();

    while let Some(&(pos, c)) = chars.peek() {
        // Skip whitespace
        if c.is_whitespace() {
            chars.next();
            continue;
        }

        // Single-char tokens
        let single = match c {
            '(' => Some(Token::LParen),
            ')' => Some(Token::RParen),
            ':' => Some(Token::Colon),
            _ => None,
        };
        if let Some(token) = single {
            tokens.push((token, pos));
            chars.next();
            continue;
        }

        // Quoted strings
        if c == '"' || c == '\'' {
            let (token, start) = tokenize_quoted(&mut chars, c, pos);
            tokens.push((token, start));
            continue;
        }

        // Numbers (including negative)
        if c.is_ascii_digit()
            || (c == '-'
                && chars
                    .clone()
                    .nth(1)
                    .is_some_and(|(_, ch)| ch.is_ascii_digit()))
        {
            let (token, start) = tokenize_number(&mut chars, pos)?;
            tokens.push((token, start));
            continue;
        }

        // Identifiers and keywords
        if is_ident_start(c) {
            let (token, start) = tokenize_ident(&mut chars, pos);
            tokens.push((token, start));
            continue;
        }

        return Err(SelectionError {
            message: format!("Unexpected character: {c}"),
            position: pos,
        });
    }

    Ok(tokens)
}

/// Tokenize a quoted string.
fn tokenize_quoted(
    chars: &mut std::iter::Peekable<std::str::CharIndices>,
    quote: char,
    pos: usize,
) -> (Token, usize) {
    chars.next(); // consume opening quote
    let start = pos + 1;
    let mut value = String::new();
    while let Some(&(_, ch)) = chars.peek() {
        if ch == quote {
            chars.next();
            break;
        }
        value.push(ch);
        chars.next();
    }
    (Token::Ident(value), start)
}

/// Tokenize a number (including negative).
fn tokenize_number(
    chars: &mut std::iter::Peekable<std::str::CharIndices>,
    pos: usize,
) -> Result<(Token, usize), SelectionError> {
    let start = pos;
    let mut num_str = String::new();

    if chars.peek().is_some_and(|(_, c)| *c == '-') {
        num_str.push('-');
        chars.next();
    }
    while let Some(&(_, ch)) = chars.peek() {
        if ch.is_ascii_digit() {
            num_str.push(ch);
            chars.next();
        } else {
            break;
        }
    }
    let num: i32 = num_str.parse().map_err(|_| SelectionError {
        message: format!("Invalid number: {num_str}"),
        position: start,
    })?;
    Ok((Token::Number(num), start))
}

/// Tokenize an identifier or keyword.
fn tokenize_ident(
    chars: &mut std::iter::Peekable<std::str::CharIndices>,
    pos: usize,
) -> (Token, usize) {
    let start = pos;
    let mut ident = String::new();
    let mut in_bracket = false;

    while let Some(&(_, ch)) = chars.peek() {
        if ch == '[' {
            in_bracket = true;
        } else if ch == ']' {
            in_bracket = false;
        }
        if is_ident_char(ch, in_bracket) {
            ident.push(ch);
            chars.next();
        } else {
            break;
        }
    }

    (ident_to_token(ident), start)
}

/// Expression AST node.
#[derive(Debug, Clone)]
enum Expr {
    Chain(Vec<GlobPattern>),
    Resname(Vec<GlobPattern>),
    Resid(Vec<(i32, i32)>), // ranges (inclusive)
    Name(Vec<GlobPattern>),
    Protein,
    Backbone,
    Sidechain,
    Nucleic,
    Hetatm,
    Hydrophobic,
    Aromatic,
    Acidic,
    Basic,
    Polar,
    Charged,
    All,
    None,
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
    Not(Box<Self>),
}

impl Expr {
    fn matches(&self, record: &AtomRecord) -> bool {
        let res = &record.res_name;
        let atom = &record.name;
        match self {
            Self::Chain(patterns) => patterns.iter().any(|p| p.matches(&record.chain_id)),
            Self::Resname(patterns) => patterns.iter().any(|p| p.matches(res)),
            Self::Resid(ranges) => ranges
                .iter()
                .any(|(lo, hi)| record.res_seq >= *lo && record.res_seq <= *hi),
            Self::Name(patterns) => patterns.iter().any(|p| p.matches(atom)),
            Self::Protein => resname_in(res, PROTEIN_RESIDUES),
            Self::Backbone => {
                resname_in(res, PROTEIN_RESIDUES) && BACKBONE_ATOMS.contains(&atom.as_str())
            }
            Self::Sidechain => {
                resname_in(res, PROTEIN_RESIDUES) && !BACKBONE_ATOMS.contains(&atom.as_str())
            }
            Self::Nucleic => resname_in(res, DNA_RESIDUES) || resname_in(res, RNA_RESIDUES),
            Self::Hetatm => record.record_name == "HETATM",
            Self::Hydrophobic => resname_in(res, HYDROPHOBIC_RESIDUES),
            Self::Aromatic => resname_in(res, AROMATIC_RESIDUES),
            Self::Acidic => resname_in(res, ACIDIC_RESIDUES),
            Self::Basic => resname_in(res, BASIC_RESIDUES),
            Self::Polar => resname_in(res, POLAR_RESIDUES),
            Self::Charged => resname_in(res, ACIDIC_RESIDUES) || resname_in(res, BASIC_RESIDUES),
            Self::All => true,
            Self::None => false,
            Self::And(a, b) => a.matches(record) && b.matches(record),
            Self::Or(a, b) => a.matches(record) || b.matches(record),
            Self::Not(a) => !a.matches(record),
        }
    }
}

/// Parser state.
struct Parser<'a> {
    tokens: &'a [(Token, usize)],
    pos: usize,
}

impl<'a> Parser<'a> {
    const fn new(tokens: &'a [(Token, usize)]) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos).map(|(t, _)| t)
    }

    fn current_pos(&self) -> usize {
        self.tokens.get(self.pos).map_or(0, |(_, p)| *p)
    }

    fn advance(&mut self) -> Option<&Token> {
        if self.pos < self.tokens.len() {
            let t = &self.tokens[self.pos].0;
            self.pos += 1;
            Some(t)
        } else {
            Option::None
        }
    }

    fn parse(&mut self) -> Result<Expr, SelectionError> {
        if self.tokens.is_empty() {
            return Err(SelectionError {
                message: "Empty selection".to_string(),
                position: 0,
            });
        }
        let expr = self.parse_or()?;
        if self.pos < self.tokens.len() {
            return Err(SelectionError {
                message: "Unexpected token after expression".to_string(),
                position: self.current_pos(),
            });
        }
        Ok(expr)
    }

    fn parse_or(&mut self) -> Result<Expr, SelectionError> {
        let mut left = self.parse_and()?;
        while self.peek() == Some(&Token::Or) {
            self.advance();
            let right = self.parse_and()?;
            left = Expr::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, SelectionError> {
        let mut left = self.parse_not()?;
        while self.peek() == Some(&Token::And) {
            self.advance();
            let right = self.parse_not()?;
            left = Expr::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_not(&mut self) -> Result<Expr, SelectionError> {
        if self.peek() == Some(&Token::Not) {
            self.advance();
            let inner = self.parse_not()?;
            return Ok(Expr::Not(Box::new(inner)));
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<Expr, SelectionError> {
        let pos = self.current_pos();
        match self.peek() {
            Some(Token::LParen) => self.parse_parenthesized(),
            Some(Token::Chain) => self.parse_pattern_keyword("chain", Expr::Chain),
            Some(Token::Resname) => self.parse_pattern_keyword("resname", Expr::Resname),
            Some(Token::Name) => self.parse_pattern_keyword("name", Expr::Name),
            Some(Token::Resid) => self.parse_resid_keyword(),
            Some(Token::Protein) => self.advance_and_ok(Expr::Protein),
            Some(Token::Backbone) => self.advance_and_ok(Expr::Backbone),
            Some(Token::Sidechain) => self.advance_and_ok(Expr::Sidechain),
            Some(Token::Nucleic) => self.advance_and_ok(Expr::Nucleic),
            Some(Token::Hetatm) => self.advance_and_ok(Expr::Hetatm),
            Some(Token::Hydrophobic) => self.advance_and_ok(Expr::Hydrophobic),
            Some(Token::Aromatic) => self.advance_and_ok(Expr::Aromatic),
            Some(Token::Acidic) => self.advance_and_ok(Expr::Acidic),
            Some(Token::Basic) => self.advance_and_ok(Expr::Basic),
            Some(Token::Polar) => self.advance_and_ok(Expr::Polar),
            Some(Token::Charged) => self.advance_and_ok(Expr::Charged),
            Some(Token::All) => self.advance_and_ok(Expr::All),
            Some(Token::None) => self.advance_and_ok(Expr::None),
            Some(Token::And | Token::Or) => Err(SelectionError {
                message: "Unexpected boolean operator".to_string(),
                position: pos,
            }),
            Some(Token::Ident(s)) => Err(SelectionError {
                message: format!("Unknown keyword: {s}"),
                position: pos,
            }),
            _ => Err(SelectionError {
                message: "Expected selection expression".to_string(),
                position: pos,
            }),
        }
    }

    /// Advance past current token and return the given expression.
    /// Returns Result for consistency with other parser methods.
    #[allow(clippy::unnecessary_wraps)]
    fn advance_and_ok(&mut self, expr: Expr) -> Result<Expr, SelectionError> {
        self.advance();
        Ok(expr)
    }

    /// Parse parenthesized expression.
    fn parse_parenthesized(&mut self) -> Result<Expr, SelectionError> {
        self.advance(); // consume '('
        let inner = self.parse_or()?;
        if self.peek() != Some(&Token::RParen) {
            return Err(SelectionError {
                message: "Missing closing parenthesis".to_string(),
                position: self.current_pos(),
            });
        }
        self.advance();
        Ok(inner)
    }

    /// Parse a keyword that takes glob patterns (chain, resname, name).
    fn parse_pattern_keyword(
        &mut self,
        name: &str,
        constructor: fn(Vec<GlobPattern>) -> Expr,
    ) -> Result<Expr, SelectionError> {
        let pos = self.current_pos();
        self.advance();
        let patterns = self.parse_patterns();
        if patterns.is_empty() {
            return Err(SelectionError {
                message: format!("{name} requires at least one argument"),
                position: pos,
            });
        }
        Ok(constructor(patterns))
    }

    /// Parse resid keyword with range support.
    fn parse_resid_keyword(&mut self) -> Result<Expr, SelectionError> {
        let pos = self.current_pos();
        self.advance();
        let ranges = self.parse_resid_ranges()?;
        if ranges.is_empty() {
            return Err(SelectionError {
                message: "resid requires at least one argument".to_string(),
                position: pos,
            });
        }
        Ok(Expr::Resid(ranges))
    }

    /// Parse one or more glob patterns (identifiers).
    fn parse_patterns(&mut self) -> Vec<GlobPattern> {
        let mut patterns = Vec::new();
        while let Some(Token::Ident(s)) = self.peek() {
            patterns.push(GlobPattern::new(s));
            self.advance();
        }
        patterns
    }

    /// Parse resid ranges: individual numbers, "N to M", or "N:M".
    fn parse_resid_ranges(&mut self) -> Result<Vec<(i32, i32)>, SelectionError> {
        let mut ranges = Vec::new();

        while let Some(token) = self.peek() {
            match token {
                Token::Number(n) => {
                    let start = *n;
                    self.advance();

                    // Check for range syntax
                    match self.peek() {
                        Some(Token::To) => {
                            self.advance();
                            if let Some(Token::Number(end)) = self.peek() {
                                ranges.push((start, *end));
                                self.advance();
                            } else {
                                return Err(SelectionError {
                                    message: "Expected number after 'to'".to_string(),
                                    position: self.current_pos(),
                                });
                            }
                        }
                        Some(Token::Colon) => {
                            self.advance();
                            if let Some(Token::Number(end)) = self.peek() {
                                ranges.push((start, *end));
                                self.advance();
                            } else {
                                return Err(SelectionError {
                                    message: "Expected number after ':'".to_string(),
                                    position: self.current_pos(),
                                });
                            }
                        }
                        _ => {
                            ranges.push((start, start));
                        }
                    }
                }
                Token::Ident(s) => {
                    // Try to parse as number or "N:M" format
                    if let Some((start, end)) = parse_colon_range(s) {
                        ranges.push((start, end));
                        self.advance();
                    } else if let Ok(n) = s.parse::<i32>() {
                        ranges.push((n, n));
                        self.advance();
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }

        Ok(ranges)
    }
}

/// Parse "N:M" format from a string.
fn parse_colon_range(s: &str) -> Option<(i32, i32)> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() == 2 {
        let start: i32 = parts[0].parse().ok()?;
        let end: i32 = parts[1].parse().ok()?;
        Some((start, end))
    } else {
        Option::None
    }
}

/// A parsed atom selection expression.
#[derive(Debug, Clone)]
pub struct Selection {
    expr: Expr,
}

impl Selection {
    /// Parse a VMD-like selection expression.
    ///
    /// # Errors
    /// Returns error if the expression is invalid.
    pub fn parse(input: &str) -> Result<Self, SelectionError> {
        debug!("Parsing selection: {input:?}");
        let tokens = tokenize(input)?;
        let mut parser = Parser::new(&tokens);
        let expr = parser.parse()?;
        debug!("Parsed selection expression: {expr:?}");
        Ok(Self { expr })
    }

    /// Test if an atom record matches this selection.
    #[must_use]
    pub fn matches(&self, record: &AtomRecord) -> bool {
        self.expr.matches(record)
    }
}

/// Parse multiple selections separated by `;`.
/// # Errors
/// Returns error if parsing fails or fewer than two selections provided.
pub fn parse_selections<S: AsRef<str>>(inputs: &[S]) -> Result<Vec<Selection>, SelectionError> {
    if inputs.len() < 2 {
        return Err(SelectionError {
            message: "At least two selections required".to_string(),
            position: 0,
        });
    }

    let selections: Result<Vec<_>, _> = inputs
        .iter()
        .enumerate()
        .map(|(idx, input)| {
            Selection::parse(input.as_ref().trim()).map_err(|mut e| {
                e.position = idx;
                e
            })
        })
        .collect();

    if let Ok(ref sels) = selections {
        debug!("Successfully parsed {} selections", sels.len());
    }
    selections
}

/// Build grouping vector from selections.
///
/// Atoms matching the same selection get the same group ID.
/// Unmatched atoms each get unique group IDs (so they don't filter contacts between each other).
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub fn build_custom_grouping(records: &[AtomRecord], selections: &[Selection]) -> Vec<i32> {
    info!(
        "Building custom grouping for {} atoms with {} selections",
        records.len(),
        selections.len()
    );
    let mut groups = vec![-1i32; records.len()];
    let num_selections = selections.len() as i32;

    // Assign group IDs based on matching selections
    for (i, record) in records.iter().enumerate() {
        for (group_id, selection) in selections.iter().enumerate() {
            if selection.matches(record) {
                groups[i] = group_id as i32;
                break;
            }
        }
    }

    // Unmatched atoms get unique group IDs starting after selection groups
    let mut unmatched_count = 0;
    let mut next_unique = num_selections;
    for group in &mut groups {
        if *group == -1 {
            *group = next_unique;
            next_unique += 1;
            unmatched_count += 1;
        }
    }

    // Log per-group counts
    for (i, sel) in selections.iter().enumerate() {
        let count = groups.iter().filter(|&&g| g == i as i32).count();
        info!("Selection {i} ({sel:?}): {count} atoms");
    }
    if unmatched_count > 0 {
        info!("{unmatched_count} atoms did not match any selection");
    }

    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that a selection matches expected records and rejects others.
    macro_rules! assert_selection {
        ($sel:expr, matches: [$($m:expr),* $(,)?], rejects: [$($r:expr),* $(,)?]) => {{
            let sel = Selection::parse($sel).unwrap();
            $(
                let (chain, resname, resid, name) = $m;
                assert!(sel.matches(&make_record(chain, resname, resid, name)),
                    "Expected '{}' to match ({}, {}, {}, {})", $sel, chain, resname, resid, name);
            )*
            $(
                let (chain, resname, resid, name) = $r;
                assert!(!sel.matches(&make_record(chain, resname, resid, name)),
                    "Expected '{}' to reject ({}, {}, {}, {})", $sel, chain, resname, resid, name);
            )*
        }};
    }

    fn make_record(chain: &str, resname: &str, resid: i32, name: &str) -> AtomRecord {
        AtomRecord {
            record_name: "ATOM".to_string(),
            serial: 1,
            name: name.to_string(),
            alt_loc: String::new(),
            res_name: resname.to_string(),
            chain_id: chain.to_string(),
            res_seq: resid,
            i_code: String::new(),
            x: 0.0,
            y: 0.0,
            z: 0.0,
            element: String::new(),
        }
    }

    fn make_hetatm(chain: &str, resname: &str, resid: i32, name: &str) -> AtomRecord {
        AtomRecord {
            record_name: "HETATM".to_string(),
            serial: 1,
            name: name.to_string(),
            alt_loc: String::new(),
            res_name: resname.to_string(),
            chain_id: chain.to_string(),
            res_seq: resid,
            i_code: String::new(),
            x: 0.0,
            y: 0.0,
            z: 0.0,
            element: String::new(),
        }
    }

    mod glob_tests {
        use super::*;

        #[test]
        fn exact_match() {
            let p = GlobPattern::new("ALA");
            assert!(p.matches("ALA"));
            assert!(!p.matches("ALAB"));
            assert!(!p.matches("ALA "));
            assert!(!p.matches("ala"));
        }

        #[test]
        fn star_wildcard() {
            let p = GlobPattern::new("C*");
            assert!(p.matches("C"));
            assert!(p.matches("CA"));
            assert!(p.matches("CG1"));
            assert!(!p.matches("NC"));
        }

        #[test]
        fn question_wildcard() {
            let p = GlobPattern::new("?A");
            assert!(p.matches("CA"));
            assert!(p.matches("NA"));
            assert!(!p.matches("A"));
            assert!(!p.matches("CAA"));
        }

        #[test]
        fn char_class() {
            let p = GlobPattern::new("[CNO]");
            assert!(p.matches("C"));
            assert!(p.matches("N"));
            assert!(p.matches("O"));
            assert!(!p.matches("S"));
        }

        #[test]
        fn complex_pattern() {
            let p = GlobPattern::new("C[AG]*");
            assert!(p.matches("CA"));
            assert!(p.matches("CG"));
            assert!(p.matches("CG1"));
            assert!(!p.matches("CB"));
        }

        #[test]
        fn star_at_beginning() {
            let p = GlobPattern::new("*A");
            assert!(p.matches("A"));
            assert!(p.matches("CA"));
            assert!(p.matches("CGA"));
            assert!(!p.matches("AB"));
        }

        #[test]
        fn star_in_middle() {
            let p = GlobPattern::new("C*A");
            assert!(p.matches("CA"));
            assert!(p.matches("CGA"));
            assert!(!p.matches("CG"));
        }
    }

    mod parse_tests {
        use super::*;

        #[test]
        fn simple_chain() {
            assert_selection!("chain A",
                matches: [("A", "ALA", 1, "CA")],
                rejects: [("B", "ALA", 1, "CA")]);
        }

        #[test]
        fn multiple_chains() {
            assert_selection!("chain A B C",
                matches: [("A", "ALA", 1, "CA"), ("B", "ALA", 1, "CA")],
                rejects: [("D", "ALA", 1, "CA")]);
        }

        #[test]
        fn resname() {
            assert_selection!("resname ALA GLY",
                matches: [("A", "ALA", 1, "CA"), ("A", "GLY", 1, "CA")],
                rejects: [("A", "VAL", 1, "CA")]);
        }

        #[test]
        fn resid_range() {
            let sel = Selection::parse("resid 10 to 20").unwrap();
            assert!(!sel.matches(&make_record("A", "ALA", 9, "CA")));
            assert!(sel.matches(&make_record("A", "ALA", 10, "CA")));
            assert!(sel.matches(&make_record("A", "ALA", 15, "CA")));
            assert!(sel.matches(&make_record("A", "ALA", 20, "CA")));
            assert!(!sel.matches(&make_record("A", "ALA", 21, "CA")));
        }

        #[test]
        fn resid_colon_range() {
            let sel = Selection::parse("resid 10:20").unwrap();
            assert!(sel.matches(&make_record("A", "ALA", 15, "CA")));
        }

        #[test]
        fn atom_name_wildcard() {
            let sel = Selection::parse("name C*").unwrap();
            assert!(sel.matches(&make_record("A", "ALA", 1, "CA")));
            assert!(sel.matches(&make_record("A", "ALA", 1, "CB")));
            assert!(sel.matches(&make_record("A", "ALA", 1, "C")));
            assert!(!sel.matches(&make_record("A", "ALA", 1, "N")));
        }

        #[test]
        fn protein_keyword() {
            let sel = Selection::parse("protein").unwrap();
            assert!(sel.matches(&make_record("A", "ALA", 1, "CA")));
            assert!(sel.matches(&make_record("A", "GLY", 1, "N")));
            assert!(!sel.matches(&make_record("A", "HOH", 1, "O")));
        }

        #[test]
        fn backbone_keyword() {
            let sel = Selection::parse("backbone").unwrap();
            assert!(sel.matches(&make_record("A", "ALA", 1, "CA")));
            assert!(sel.matches(&make_record("A", "ALA", 1, "C")));
            assert!(sel.matches(&make_record("A", "ALA", 1, "N")));
            assert!(sel.matches(&make_record("A", "ALA", 1, "O")));
            assert!(!sel.matches(&make_record("A", "ALA", 1, "CB")));
        }

        #[test]
        fn sidechain_keyword() {
            let sel = Selection::parse("sidechain").unwrap();
            assert!(sel.matches(&make_record("A", "ALA", 1, "CB")));
            assert!(sel.matches(&make_record("A", "ALA", 1, "CG")));
            assert!(!sel.matches(&make_record("A", "ALA", 1, "CA")));
            assert!(!sel.matches(&make_record("A", "HOH", 1, "O")));
        }

        #[test]
        fn hetatm_keyword() {
            let sel = Selection::parse("hetatm").unwrap();
            assert!(sel.matches(&make_hetatm("A", "HOH", 1, "O")));
            assert!(!sel.matches(&make_record("A", "ALA", 1, "CA")));
        }

        #[test]
        fn all_keyword() {
            let sel = Selection::parse("all").unwrap();
            assert!(sel.matches(&make_record("A", "ALA", 1, "CA")));
            assert!(sel.matches(&make_hetatm("A", "HOH", 1, "O")));
        }

        #[test]
        fn none_keyword() {
            let sel = Selection::parse("none").unwrap();
            assert!(!sel.matches(&make_record("A", "ALA", 1, "CA")));
        }

        #[test]
        fn hydrophobic_keyword() {
            assert_selection!("hydrophobic",
                matches: [("A", "ALA", 1, "CA"), ("A", "VAL", 1, "CA"), ("A", "PHE", 1, "CA")],
                rejects: [("A", "ASP", 1, "CA"), ("A", "SER", 1, "CA")]);
        }

        #[test]
        fn aromatic_keyword() {
            assert_selection!("aromatic",
                matches: [("A", "PHE", 1, "CA"), ("A", "TYR", 1, "CA"), ("A", "TRP", 1, "CA"), ("A", "HIS", 1, "CA")],
                rejects: [("A", "ALA", 1, "CA")]);
        }

        #[test]
        fn acidic_keyword() {
            assert_selection!("acidic",
                matches: [("A", "ASP", 1, "CA"), ("A", "GLU", 1, "CA")],
                rejects: [("A", "LYS", 1, "CA"), ("A", "ALA", 1, "CA")]);
        }

        #[test]
        fn basic_keyword() {
            assert_selection!("basic",
                matches: [("A", "ARG", 1, "CA"), ("A", "LYS", 1, "CA"), ("A", "HIS", 1, "CA")],
                rejects: [("A", "ASP", 1, "CA"), ("A", "ALA", 1, "CA")]);
        }

        #[test]
        fn polar_keyword() {
            assert_selection!("polar",
                matches: [("A", "SER", 1, "CA"), ("A", "THR", 1, "CA"), ("A", "ASN", 1, "CA"), ("A", "GLN", 1, "CA")],
                rejects: [("A", "ALA", 1, "CA")]);
        }

        #[test]
        fn charged_keyword() {
            assert_selection!("charged",
                matches: [("A", "ASP", 1, "CA"), ("A", "GLU", 1, "CA"), ("A", "ARG", 1, "CA"), ("A", "LYS", 1, "CA")],
                rejects: [("A", "ALA", 1, "CA"), ("A", "SER", 1, "CA")]);
        }
    }

    mod boolean_tests {
        use super::*;

        #[test]
        fn and_operator() {
            let sel = Selection::parse("chain A and resname ALA").unwrap();
            assert!(sel.matches(&make_record("A", "ALA", 1, "CA")));
            assert!(!sel.matches(&make_record("A", "GLY", 1, "CA")));
            assert!(!sel.matches(&make_record("B", "ALA", 1, "CA")));
        }

        #[test]
        fn or_operator() {
            let sel = Selection::parse("resname ALA or resname GLY").unwrap();
            assert!(sel.matches(&make_record("A", "ALA", 1, "CA")));
            assert!(sel.matches(&make_record("A", "GLY", 1, "CA")));
            assert!(!sel.matches(&make_record("A", "VAL", 1, "CA")));
        }

        #[test]
        fn not_operator() {
            let sel = Selection::parse("not protein").unwrap();
            assert!(!sel.matches(&make_record("A", "ALA", 1, "CA")));
            assert!(sel.matches(&make_hetatm("A", "ZN", 1, "ZN")));
        }

        #[test]
        fn precedence_not_binds_tighter() {
            // "not A and B" should parse as "(not A) and B"
            let sel = Selection::parse("not resname HOH and chain A").unwrap();
            assert!(sel.matches(&make_record("A", "ALA", 1, "CA")));
            assert!(!sel.matches(&make_record("A", "HOH", 1, "O")));
            assert!(!sel.matches(&make_record("B", "ALA", 1, "CA")));
        }

        #[test]
        fn precedence_and_binds_tighter_than_or() {
            // "A or B and C" should parse as "A or (B and C)"
            let sel = Selection::parse("resname HOH or chain A and resname ALA").unwrap();
            assert!(sel.matches(&make_record("A", "HOH", 1, "O"))); // HOH matches
            assert!(sel.matches(&make_record("A", "ALA", 1, "CA"))); // A and ALA matches
            assert!(!sel.matches(&make_record("A", "GLY", 1, "CA"))); // A but not ALA
            assert!(sel.matches(&make_record("B", "HOH", 1, "O"))); // HOH matches
        }

        #[test]
        fn parentheses() {
            let sel = Selection::parse("protein and (backbone or name CB)").unwrap();
            assert!(sel.matches(&make_record("A", "ALA", 1, "CA"))); // backbone
            assert!(sel.matches(&make_record("A", "ALA", 1, "CB"))); // CB
            assert!(!sel.matches(&make_record("A", "ALA", 1, "CG"))); // neither
            assert!(!sel.matches(&make_record("A", "HOH", 1, "O"))); // not protein
        }

        #[test]
        fn nested_parentheses() {
            let sel =
                Selection::parse("(chain A or chain B) and (resname ALA or resname GLY)").unwrap();
            assert!(sel.matches(&make_record("A", "ALA", 1, "CA")));
            assert!(sel.matches(&make_record("B", "GLY", 1, "CA")));
            assert!(!sel.matches(&make_record("C", "ALA", 1, "CA")));
            assert!(!sel.matches(&make_record("A", "VAL", 1, "CA")));
        }

        #[test]
        fn not_not_protein() {
            let sel = Selection::parse("not not protein").unwrap();
            assert!(sel.matches(&make_record("A", "ALA", 1, "CA")));
            assert!(!sel.matches(&make_record("A", "HOH", 1, "O")));
        }

        #[test]
        fn not_parenthesized() {
            let sel = Selection::parse("not (protein and hetatm)").unwrap();
            // protein AND hetatm is always false, so NOT is always true
            assert!(sel.matches(&make_record("A", "ALA", 1, "CA")));
        }
    }

    mod selection_grouping_tests {
        use super::*;

        fn selections(args: &[&str]) -> Vec<String> {
            args.iter().map(|s| (*s).to_string()).collect()
        }

        #[test]
        fn parse_two_selections() {
            let sels = parse_selections(&selections(&["chain A", "chain B"])).unwrap();
            assert_eq!(sels.len(), 2);
        }

        #[test]
        fn parse_multiple_selections() {
            let sels =
                parse_selections(&selections(&["chain A B", "chain L", "resname HOH"])).unwrap();
            assert_eq!(sels.len(), 3);
        }

        #[test]
        fn build_grouping_simple() {
            let records = vec![
                make_record("A", "ALA", 1, "CA"),
                make_record("A", "ALA", 2, "CA"),
                make_record("L", "LIG", 1, "C1"),
            ];
            let sels = parse_selections(&selections(&["chain A", "chain L"])).unwrap();
            let groups = build_custom_grouping(&records, &sels);

            assert_eq!(groups[0], groups[1]); // Same group
            assert_ne!(groups[0], groups[2]); // Different groups
        }

        #[test]
        fn unmatched_atoms_get_unique_groups() {
            let records = vec![
                make_record("A", "ALA", 1, "CA"),
                make_record("X", "UNK", 1, "X1"),
                make_record("Y", "UNK", 1, "Y1"),
            ];
            let sels = parse_selections(&selections(&["chain A", "chain B"])).unwrap();
            let groups = build_custom_grouping(&records, &sels);

            // Unmatched atoms should each get unique groups
            assert_ne!(groups[1], groups[2]);
        }

        #[test]
        fn requires_at_least_two_selections() {
            assert!(parse_selections(&selections(&["chain A"])).is_err());
            assert!(parse_selections(&selections(&[])).is_err());
        }

        #[test]
        fn overlapping_selections_first_match_wins() {
            // Atom matches both "chain A" and "resname ALA" - first selection wins
            let records = vec![
                make_record("A", "ALA", 1, "CA"), // matches both
                make_record("A", "GLY", 2, "CA"), // matches only chain A
                make_record("B", "ALA", 1, "CA"), // matches only resname ALA
            ];
            let sels = parse_selections(&selections(&["chain A", "resname ALA"])).unwrap();
            let groups = build_custom_grouping(&records, &sels);

            assert_eq!(
                groups[0], 0,
                "overlapping atom assigned to first matching selection"
            );
            assert_eq!(groups[1], 0, "chain A only");
            assert_eq!(groups[2], 1, "resname ALA only");
        }
    }

    mod alias_tests {
        use super::*;

        #[test]
        fn all_and_everything() {
            let sel1 = Selection::parse("all").unwrap();
            let sel2 = Selection::parse("everything").unwrap();
            let rec = make_record("A", "ALA", 1, "CA");
            assert!(sel1.matches(&rec));
            assert!(sel2.matches(&rec));
        }

        #[test]
        fn none_and_nothing() {
            let sel1 = Selection::parse("none").unwrap();
            let sel2 = Selection::parse("nothing").unwrap();
            let rec = make_record("A", "ALA", 1, "CA");
            assert!(!sel1.matches(&rec));
            assert!(!sel2.matches(&rec));
        }

        #[test]
        fn keyword_aliases() {
            // segid = chain
            let sel = Selection::parse("segid A").unwrap();
            assert!(sel.matches(&make_record("A", "ALA", 1, "CA")));

            // resn = resname
            let sel = Selection::parse("resn ALA").unwrap();
            assert!(sel.matches(&make_record("A", "ALA", 1, "CA")));

            // resi = resid
            let sel = Selection::parse("resi 1").unwrap();
            assert!(sel.matches(&make_record("A", "ALA", 1, "CA")));
        }
    }

    mod multi_value_tests {
        use super::*;

        #[test]
        fn resname_multiple_space_separated() {
            let sel = Selection::parse("resname ALA ASP GLU").unwrap();
            assert!(sel.matches(&make_record("A", "ALA", 1, "CA")));
            assert!(sel.matches(&make_record("A", "ASP", 1, "CA")));
            assert!(sel.matches(&make_record("A", "GLU", 1, "CA")));
            assert!(!sel.matches(&make_record("A", "VAL", 1, "CA")));
        }

        #[test]
        fn resid_multiple_values() {
            let sel = Selection::parse("resid 100 101 102").unwrap();
            assert!(sel.matches(&make_record("A", "ALA", 100, "CA")));
            assert!(sel.matches(&make_record("A", "ALA", 101, "CA")));
            assert!(!sel.matches(&make_record("A", "ALA", 99, "CA")));
        }

        #[test]
        fn chain_multiple() {
            let sel = Selection::parse("chain A B C").unwrap();
            assert!(sel.matches(&make_record("A", "ALA", 1, "CA")));
            assert!(sel.matches(&make_record("B", "ALA", 1, "CA")));
            assert!(sel.matches(&make_record("C", "ALA", 1, "CA")));
            assert!(!sel.matches(&make_record("D", "ALA", 1, "CA")));
        }
    }

    mod range_tests {
        use super::*;

        #[test]
        fn resid_to_syntax() {
            let sel = Selection::parse("resid 1 to 10").unwrap();
            assert!(sel.matches(&make_record("A", "ALA", 1, "CA")));
            assert!(sel.matches(&make_record("A", "ALA", 5, "CA")));
            assert!(sel.matches(&make_record("A", "ALA", 10, "CA")));
            assert!(!sel.matches(&make_record("A", "ALA", 11, "CA")));
        }

        #[test]
        fn resid_colon_syntax() {
            let sel = Selection::parse("resid 1:10").unwrap();
            assert!(sel.matches(&make_record("A", "ALA", 5, "CA")));
        }

        #[test]
        fn resid_negative() {
            let sel = Selection::parse("resid -5 to 5").unwrap();
            assert!(sel.matches(&make_record("A", "ALA", -3, "CA")));
            assert!(sel.matches(&make_record("A", "ALA", 0, "CA")));
            assert!(!sel.matches(&make_record("A", "ALA", 6, "CA")));
        }
    }

    mod quote_tests {
        use super::*;

        #[test]
        fn single_quoted_name() {
            let sel = Selection::parse("name 'CA'").unwrap();
            assert!(sel.matches(&make_record("A", "ALA", 1, "CA")));
        }

        #[test]
        fn double_quoted_name() {
            let sel = Selection::parse("name \"CA\"").unwrap();
            assert!(sel.matches(&make_record("A", "ALA", 1, "CA")));
        }

        #[test]
        fn quoted_with_special_chars() {
            let sel = Selection::parse("name \"O5'\"").unwrap();
            assert!(sel.matches(&make_record("A", "DA", 1, "O5'")));
        }
    }

    mod error_tests {
        use super::*;

        #[test]
        fn unknown_keyword() {
            let result = Selection::parse("unknown_keyword foo");
            assert!(result.is_err());
        }

        #[test]
        fn missing_argument() {
            let result = Selection::parse("chain");
            assert!(result.is_err());
        }

        #[test]
        fn unmatched_parenthesis() {
            let result = Selection::parse("(chain A and resname ALA");
            assert!(result.is_err());
        }

        #[test]
        fn dangling_operator() {
            assert!(Selection::parse("or").is_err());
            assert!(Selection::parse("and").is_err());
            assert!(Selection::parse("chain A and").is_err());
        }

        #[test]
        fn invalid_in_boolean_context() {
            assert!(Selection::parse("not unknown").is_err());
            assert!(Selection::parse("protein or unknown").is_err());
        }
    }
}
