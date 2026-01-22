//! Input file parsing for PDB, mmCIF, and XYZR formats.
//!
//! Provides format auto-detection and unified parsing interface with radii assignment.

pub mod mmcif;
pub mod pdb;
pub mod radii;

use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::Path;

use log::{debug, info};

pub use radii::RadiiLookup;

use crate::Ball;

/// Parsed atom record with coordinates and metadata.
#[derive(Clone, Debug)]
pub struct AtomRecord {
    pub record_name: String,
    pub serial: i32,
    pub name: String,
    pub alt_loc: String,
    pub res_name: String,
    pub chain_id: String,
    pub res_seq: i32,
    pub i_code: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub element: String,
}

/// Input file format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputFormat {
    Pdb,
    Mmcif,
    Xyzr,
}

/// Options for parsing molecular files.
#[derive(Clone, Debug, Default)]
pub struct ParseOptions {
    /// Exclude HETATM records (default: false, matching C++ behavior)
    pub exclude_heteroatoms: bool,
    /// Include hydrogen atoms (default: false)
    pub include_hydrogens: bool,
    /// Treat multi-model files as assembly (default: false)
    pub as_assembly: bool,
}

/// Detect input format from file extension.
fn detect_format_from_extension(path: &Path) -> Option<InputFormat> {
    let ext = path.extension()?.to_str()?.to_lowercase();
    match ext.as_str() {
        "pdb" | "ent" | "pdb1" => Some(InputFormat::Pdb),
        "cif" | "mmcif" => Some(InputFormat::Mmcif),
        "xyzr" => Some(InputFormat::Xyzr),
        _ => None,
    }
}

/// Detect input format from file content (first non-empty line).
fn detect_format_from_content<R: BufRead>(reader: &mut R) -> Option<(InputFormat, String)> {
    let mut first_line = String::new();
    reader.read_line(&mut first_line).ok()?;

    let trimmed = first_line.trim();
    if trimmed.starts_with("data_") || trimmed.starts_with('_') || trimmed.starts_with("loop_") {
        Some((InputFormat::Mmcif, first_line))
    } else if trimmed.starts_with("ATOM")
        || trimmed.starts_with("HETATM")
        || trimmed.starts_with("HEADER")
        || trimmed.starts_with("REMARK")
    {
        Some((InputFormat::Pdb, first_line))
    } else {
        // Try XYZR: at least 4 whitespace-separated numbers
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() >= 4 && parts.iter().rev().take(4).all(|p| p.parse::<f64>().is_ok()) {
            Some((InputFormat::Xyzr, first_line))
        } else {
            None
        }
    }
}

/// Parse XYZR format (x y z radius per line, uses last 4 columns).
fn parse_xyzr<R: BufRead>(reader: R, first_line: Option<&str>) -> Vec<Ball> {
    let parse_line = |line: &str| -> Option<Ball> {
        let mut parts = line.split_whitespace().rev();
        let r: f64 = parts.next()?.parse().ok()?;
        let z: f64 = parts.next()?.parse().ok()?;
        let y: f64 = parts.next()?.parse().ok()?;
        let x: f64 = parts.next()?.parse().ok()?;
        Some(Ball::new(x, y, z, r))
    };

    let mut balls = Vec::new();

    // Parse first line if provided (already read for format detection)
    if let Some(line) = first_line
        && let Some(ball) = parse_line(line)
    {
        balls.push(ball);
    }

    balls.extend(
        reader
            .lines()
            .map_while(Result::ok)
            .filter_map(|l| parse_line(&l)),
    );
    balls
}

/// Convert atom records to balls using radii lookup.
fn records_to_balls(records: &[AtomRecord], radii: &RadiiLookup) -> Vec<Ball> {
    records
        .iter()
        .map(|r| {
            let radius = radii.get_radius(&r.res_name, &r.name);
            Ball::new(r.x, r.y, r.z, radius)
        })
        .collect()
}

/// Parse input from a file path with auto-detected format.
///
/// # Errors
/// Returns error if file cannot be opened or format cannot be detected.
pub fn parse_file(
    path: &Path,
    options: &ParseOptions,
    radii: &RadiiLookup,
) -> io::Result<Vec<Ball>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Try extension first, then content
    let format = detect_format_from_extension(path);

    if let Some(fmt) = format {
        info!("Detected format from extension: {fmt:?}");
        parse_reader(&mut reader, fmt, None, options, radii)
    } else {
        // Detect from content
        let (fmt, first_line) = detect_format_from_content(&mut reader).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "Unable to detect input format")
        })?;
        info!("Detected format from content: {fmt:?}");
        parse_reader(&mut reader, fmt, Some(&first_line), options, radii)
    }
}

/// Parse input from a buffered reader with known format.
///
/// # Errors
/// Returns error if reading from the underlying reader fails.
pub fn parse_reader<R: BufRead>(
    reader: R,
    format: InputFormat,
    first_line: Option<&str>,
    options: &ParseOptions,
    radii: &RadiiLookup,
) -> io::Result<Vec<Ball>> {
    match format {
        InputFormat::Xyzr => Ok(parse_xyzr(reader, first_line)),
        InputFormat::Pdb => {
            // For PDB, we need to prepend first_line if it was read
            let records = if let Some(line) = first_line {
                let combined = std::iter::once(Ok(line.to_string())).chain(reader.lines());
                pdb::parse_pdb(CombinedReader::new(combined), options)
            } else {
                pdb::parse_pdb(reader, options)
            };
            debug!("Parsed {} atom records", records.len());
            Ok(records_to_balls(&records, radii))
        }
        InputFormat::Mmcif => {
            let records = if let Some(line) = first_line {
                let combined = std::iter::once(Ok(line.to_string())).chain(reader.lines());
                mmcif::parse_mmcif(CombinedReader::new(combined), options)
            } else {
                mmcif::parse_mmcif(reader, options)
            };
            debug!("Parsed {} atom records", records.len());
            Ok(records_to_balls(&records, radii))
        }
    }
}

/// Parse input from stdin with auto-detected format.
///
/// # Errors
/// Returns error if format cannot be detected.
pub fn parse_stdin(options: &ParseOptions, radii: &RadiiLookup) -> io::Result<Vec<Ball>> {
    let stdin = io::stdin();
    let mut reader = stdin.lock();

    let (format, first_line) = detect_format_from_content(&mut reader).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "Unable to detect input format from stdin",
        )
    })?;

    info!("Detected format from stdin: {format:?}");
    parse_reader(reader, format, Some(&first_line), options, radii)
}

/// Adapter to prepend a first line to a reader.
struct CombinedReader<I> {
    lines: I,
    current_line: Option<String>,
}

impl<I: Iterator<Item = io::Result<String>>> CombinedReader<I> {
    const fn new(lines: I) -> Self {
        Self {
            lines,
            current_line: None,
        }
    }
}

impl<I: Iterator<Item = io::Result<String>>> BufRead for CombinedReader<I> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        if self.current_line.is_none() {
            self.current_line = self.lines.next().transpose()?;
            if let Some(ref mut line) = self.current_line
                && !line.ends_with('\n')
            {
                line.push('\n');
            }
        }
        Ok(self.current_line.as_deref().unwrap_or("").as_bytes())
    }

    fn consume(&mut self, amt: usize) {
        if let Some(ref mut line) = self.current_line {
            if amt >= line.len() {
                self.current_line = None;
            } else {
                *line = line[amt..].to_string();
            }
        }
    }
}

impl<I: Iterator<Item = io::Result<String>>> Read for CombinedReader<I> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let available = self.fill_buf()?;
        let amt = std::cmp::min(buf.len(), available.len());
        buf[..amt].copy_from_slice(&available[..amt]);
        self.consume(amt);
        Ok(amt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_pdb_extension() {
        assert_eq!(
            detect_format_from_extension(Path::new("file.pdb")),
            Some(InputFormat::Pdb)
        );
        assert_eq!(
            detect_format_from_extension(Path::new("file.pdb1")),
            Some(InputFormat::Pdb)
        );
        assert_eq!(
            detect_format_from_extension(Path::new("file.ent")),
            Some(InputFormat::Pdb)
        );
    }

    #[test]
    fn detect_mmcif_extension() {
        assert_eq!(
            detect_format_from_extension(Path::new("file.cif")),
            Some(InputFormat::Mmcif)
        );
        assert_eq!(
            detect_format_from_extension(Path::new("file.mmcif")),
            Some(InputFormat::Mmcif)
        );
    }

    #[test]
    fn detect_xyzr_extension() {
        assert_eq!(
            detect_format_from_extension(Path::new("file.xyzr")),
            Some(InputFormat::Xyzr)
        );
    }

    #[test]
    fn parse_xyzr_format() {
        let data = "1.0 2.0 3.0 4.0\n5.0 6.0 7.0 8.0\n";
        let balls = parse_xyzr(data.as_bytes(), None);
        assert_eq!(balls.len(), 2);
        assert_eq!(balls[0], Ball::new(1.0, 2.0, 3.0, 4.0));
        assert_eq!(balls[1], Ball::new(5.0, 6.0, 7.0, 8.0));
    }
}
