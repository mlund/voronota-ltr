use std::io::Write;
use std::process::{Command, Stdio};

fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_voronotalt"))
}

fn assert_approx(name: &str, actual: f64, expected: f64, tolerance: f64) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= tolerance,
        "{name}: expected {expected}, got {actual} (diff {diff} > {tolerance})"
    );
}

fn parse_summary(output: &str) -> std::collections::HashMap<String, String> {
    output
        .lines()
        .filter_map(|line| {
            let mut parts = line.splitn(2, ':');
            Some((parts.next()?.trim().to_string(), parts.next()?.trim().to_string()))
        })
        .collect()
}

#[test]
fn test_balls_cs_1x1() {
    let output = binary()
        .args(["-i", "benches/data/balls_cs_1x1.xyzr", "--probe", "2.0", "-q"])
        .output()
        .expect("failed to run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let summary = parse_summary(&stdout);

    assert_eq!(summary["contacts"], "153");
    assert_eq!(summary["cells"], "100");
    assert_approx("total_contact_area", summary["total_contact_area"].parse().unwrap(), 3992.55, 1.0);
    assert_approx("total_sas_area", summary["total_sas_area"].parse().unwrap(), 21979.6, 10.0);
    assert_approx("total_volume", summary["total_volume"].parse().unwrap(), 46419.9, 10.0);
}

#[test]
fn test_balls_cs_1x1_periodic() {
    let output = binary()
        .args([
            "-i", "benches/data/balls_cs_1x1.xyzr",
            "--probe", "2.0",
            "--periodic-box-corners", "0", "0", "0", "200", "250", "300",
            "-q",
        ])
        .output()
        .expect("failed to run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let summary = parse_summary(&stdout);

    assert_eq!(summary["contacts"], "189");
    assert_eq!(summary["cells"], "100");
    assert_approx("total_contact_area", summary["total_contact_area"].parse().unwrap(), 4812.14, 50.0);
    assert_approx("total_sas_area", summary["total_sas_area"].parse().unwrap(), 20023.1, 100.0);
    assert_approx("total_volume", summary["total_volume"].parse().unwrap(), 45173.2, 100.0);
}

#[test]
fn test_balls_2zsk() {
    let output = binary()
        .args(["-i", "benches/data/balls_2zsk.xyzr", "--probe", "1.4", "-q"])
        .output()
        .expect("failed to run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let summary = parse_summary(&stdout);

    assert_eq!(summary["contacts"], "23855");
    assert_eq!(summary["cells"], "3545");
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
    let summary = parse_summary(&stdout);

    assert_eq!(summary["cells"], "3");
}

#[test]
fn test_print_contacts() {
    let output = binary()
        .args(["-i", "benches/data/balls_cs_1x1.xyzr", "--probe", "2.0", "--print-contacts", "-q"])
        .output()
        .expect("failed to run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let first_line = stdout.lines().next().unwrap();
    let parts: Vec<&str> = first_line.split_whitespace().collect();

    assert_eq!(parts.len(), 4, "contact line should have 4 columns");
    assert_eq!(parts[0], "0");
    assert_eq!(parts[1], "1");
    assert_approx("first_contact_area", parts[2].parse().unwrap(), 42.9555, 0.01);
}

#[test]
fn test_print_cells() {
    let output = binary()
        .args(["-i", "benches/data/balls_cs_1x1.xyzr", "--probe", "2.0", "--print-cells", "-q"])
        .output()
        .expect("failed to run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let first_line = stdout.lines().next().unwrap();
    let parts: Vec<&str> = first_line.split_whitespace().collect();

    assert_eq!(parts.len(), 3, "cell line should have 3 columns");
    assert_eq!(parts[0], "0");
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
}
