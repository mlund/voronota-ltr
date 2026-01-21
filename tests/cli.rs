use std::io::Write;
use std::process::{Command, Stdio};

use serde::Deserialize;

fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_voronota-ltr"))
}

fn assert_approx(name: &str, actual: f64, expected: f64, tolerance: f64) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= tolerance,
        "{name}: expected {expected}, got {actual} (diff {diff} > {tolerance})"
    );
}

#[derive(Deserialize)]
struct Contact {
    area: f64,
}

#[derive(Deserialize)]
struct Cell {
    sas_area: f64,
    volume: f64,
}

#[derive(Deserialize)]
struct TessellationResult {
    contacts: Vec<Contact>,
    cells: Vec<Cell>,
}

fn parse_json(output: &str) -> TessellationResult {
    serde_json::from_str(output).expect("failed to parse JSON output")
}

#[test]
fn test_balls_cs_1x1() {
    let output = binary()
        .args([
            "-i",
            "benches/data/balls_cs_1x1.xyzr",
            "--probe",
            "2.0",
            "-q",
        ])
        .output()
        .expect("failed to run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let result = parse_json(&stdout);

    assert_eq!(result.contacts.len(), 153);
    assert_eq!(result.cells.len(), 100);

    let total_contact_area: f64 = result.contacts.iter().map(|c| c.area).sum();
    let total_sas_area: f64 = result.cells.iter().map(|c| c.sas_area).sum();
    let total_volume: f64 = result.cells.iter().map(|c| c.volume).sum();

    assert_approx("total_contact_area", total_contact_area, 3992.55, 1.0);
    assert_approx("total_sas_area", total_sas_area, 21979.6, 10.0);
    assert_approx("total_volume", total_volume, 46419.9, 10.0);
}

#[test]
fn test_balls_cs_1x1_periodic() {
    let output = binary()
        .args([
            "-i",
            "benches/data/balls_cs_1x1.xyzr",
            "--probe",
            "2.0",
            "--periodic-box-corners",
            "0",
            "0",
            "0",
            "200",
            "250",
            "300",
            "-q",
        ])
        .output()
        .expect("failed to run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let result = parse_json(&stdout);

    assert_eq!(result.contacts.len(), 189);
    assert_eq!(result.cells.len(), 100);

    let total_contact_area: f64 = result.contacts.iter().map(|c| c.area).sum();
    let total_sas_area: f64 = result.cells.iter().map(|c| c.sas_area).sum();
    let total_volume: f64 = result.cells.iter().map(|c| c.volume).sum();

    assert_approx("total_contact_area", total_contact_area, 4812.14, 50.0);
    assert_approx("total_sas_area", total_sas_area, 20023.1, 100.0);
    assert_approx("total_volume", total_volume, 45173.2, 100.0);
}

#[test]
fn test_balls_2zsk() {
    let output = binary()
        .args(["-i", "benches/data/balls_2zsk.xyzr", "--probe", "1.4", "-q"])
        .output()
        .expect("failed to run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let result = parse_json(&stdout);

    assert_eq!(result.contacts.len(), 23855);
    assert_eq!(result.cells.len(), 3545);
}

#[test]
fn test_stdin_input() {
    let mut child = binary()
        .args(["--probe", "1.0", "-q"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to spawn binary");

    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"0 0 0 1\n0.5 0 0 1\n1 0 0 1\n")
        .unwrap();

    let output = child.wait_with_output().expect("failed to wait");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let result = parse_json(&stdout);

    assert_eq!(result.cells.len(), 3);
}

#[test]
fn test_help() {
    let output = binary()
        .arg("--help")
        .output()
        .expect("failed to run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains(".xyzr"));
    assert!(stdout.contains("--probe"));
    assert!(stdout.contains("--output"));
}
