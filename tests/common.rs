#![allow(dead_code)]

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;

pub const EPSILON: f64 = 0.01;

#[derive(Deserialize)]
pub struct Contact {
    pub area: f64,
}

#[derive(Deserialize)]
pub struct Cell {
    pub sas_area: f64,
    pub volume: f64,
}

#[derive(Deserialize)]
pub struct TessellationResult {
    pub contacts: Vec<Contact>,
    pub cells: Vec<Cell>,
}

pub fn binary_command() -> Command {
    Command::new(env!("CARGO_BIN_EXE_voronota-ltr"))
}

pub fn parse_json(output: &str) -> TessellationResult {
    serde_json::from_str(output).expect("failed to parse JSON output")
}

pub fn run_cli(args: &[&str]) -> TessellationResult {
    let output = binary_command()
        .args(args)
        .output()
        .expect("failed to run binary");
    assert!(output.status.success());
    let stdout = std::str::from_utf8(&output.stdout).expect("stdout was not UTF-8");
    parse_json(stdout)
}

pub fn tessellation_totals(result: &TessellationResult) -> (f64, f64, f64) {
    let total_contact_area = result.contacts.iter().map(|c| c.area).sum();
    let total_sas_area = result.cells.iter().map(|c| c.sas_area).sum();
    let total_volume = result.cells.iter().map(|c| c.volume).sum();
    (total_contact_area, total_sas_area, total_volume)
}

fn test_data_path(name: &str) -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(manifest_dir).join("tests/data").join(name)
}

pub fn require_test_file(name: &str) -> Option<PathBuf> {
    let path = test_data_path(name);
    if path.exists() {
        Some(path)
    } else {
        eprintln!("Skipping test: {} not found", path.display());
        None
    }
}

pub struct CellReference {
    pub index: usize,
    pub sas_area: f64,
    pub volume: f64,
}

pub fn parse_reference_cells(path: &Path) -> Vec<CellReference> {
    let file = File::open(path).expect("Failed to open reference file");
    let reader = BufReader::new(file);

    reader
        .lines()
        .skip(1)
        .filter_map(|line| {
            let line = line.ok()?;
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() < 8 {
                return None;
            }
            Some(CellReference {
                index: parts[5].parse().ok()?,
                sas_area: parts[6].parse().ok()?,
                volume: parts[7].parse().ok()?,
            })
        })
        .collect()
}
