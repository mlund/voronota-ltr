mod common;

use std::io::{Read, Write};
use std::process::Stdio;

use approx::assert_abs_diff_eq;
use common::{EPSILON, parse_json, run_cli, tessellation_totals};

#[test]
fn cli_balls_cs_1x1() {
    #[rustfmt::skip]
    let result = run_cli(&["-i", "benches/data/balls_cs_1x1.xyzr", "--probe", "2.0", "-q"]);

    assert_eq!(result.contacts.len(), 153);
    assert_eq!(result.cells.len(), 100);

    let (total_contact_area, total_sas_area, total_volume) = tessellation_totals(&result);

    assert_abs_diff_eq!(total_contact_area, 3992.55, epsilon = EPSILON);
    assert_abs_diff_eq!(total_sas_area, 21979.64, epsilon = EPSILON);
    assert_abs_diff_eq!(total_volume, 46419.87, epsilon = EPSILON);
}

#[test]
fn cli_balls_cs_1x1_periodic() {
    #[rustfmt::skip]
    let result = run_cli(&[
        "-i", "benches/data/balls_cs_1x1.xyzr", "--probe", "2.0",
        "--periodic-box-corners", "0", "0", "0", "200", "250", "300", "-q",
    ]);

    assert_eq!(result.contacts.len(), 189);
    assert_eq!(result.cells.len(), 100);

    let (total_contact_area, total_sas_area, total_volume) = tessellation_totals(&result);

    assert_abs_diff_eq!(total_contact_area, 4812.14, epsilon = EPSILON);
    assert_abs_diff_eq!(total_sas_area, 20023.06, epsilon = EPSILON);
    assert_abs_diff_eq!(total_volume, 45173.20, epsilon = EPSILON);
}

#[test]
fn cli_balls_cs_1x1_periodic_directions() {
    #[rustfmt::skip]
    let result = run_cli(&[
        "-i", "benches/data/balls_cs_1x1.xyzr", "--probe", "2.0", "--periodic-box-directions",
        "200", "0", "0",   // vector a
        "0", "250", "0",   // vector b
        "0", "0", "300",   // vector c
        "-q",
    ]);

    assert_eq!(result.contacts.len(), 189);
    assert_eq!(result.cells.len(), 100);

    let (total_contact_area, total_sas_area, total_volume) = tessellation_totals(&result);

    assert_abs_diff_eq!(total_contact_area, 4812.14, epsilon = EPSILON);
    assert_abs_diff_eq!(total_sas_area, 20023.06, epsilon = EPSILON);
    assert_abs_diff_eq!(total_volume, 45173.20, epsilon = EPSILON);
}

#[test]
fn cli_balls_2zsk() {
    #[rustfmt::skip]
    let result = run_cli(&["-i", "benches/data/balls_2zsk.xyzr", "--probe", "1.4", "-q"]);

    assert_eq!(result.contacts.len(), 23855);
    assert_eq!(result.cells.len(), 3545);
}

#[test]
fn cli_stdin_input() {
    #[rustfmt::skip]
    let mut child = common::binary_command()
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
fn cli_help() {
    let output = common::binary_command()
        .arg("--help")
        .output()
        .expect("failed to run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("XYZR"));
    assert!(stdout.contains("--probe"));
    assert!(stdout.contains("--output"));
}

/// Run CLI with stdin input and return graphics file content
macro_rules! run_graphics_test {
    ($filename:expr, $input:expr) => {{
        let temp_file = std::env::temp_dir().join($filename);
        let mut child = common::binary_command()
            .args([
                "--probe",
                "1.0",
                "-q",
                "--graphics-output-file-for-pymol",
                temp_file.to_str().unwrap(),
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("failed to spawn binary");

        child.stdin.take().unwrap().write_all($input).unwrap();
        let output = child.wait_with_output().expect("failed to wait");
        assert!(output.status.success());

        let mut content = String::new();
        std::fs::File::open(&temp_file)
            .expect("failed to open graphics file")
            .read_to_string(&mut content)
            .expect("failed to read graphics file");
        std::fs::remove_file(&temp_file).ok();
        content
    }};
}

#[test]
fn cli_pymol_graphics_three_balls_line() {
    let content = run_graphics_test!("test_graphics.py", b"0 0 0 1\n0.5 0 0 1\n1 0 0 1\n");

    assert!(content.contains("from pymol.cgo import *"));
    assert_eq!(content.matches("SPHERE,").count(), 3);
    assert_eq!(content.matches("TRIANGLE_FAN").count(), 2);
    assert_eq!(content.matches("LINE_LOOP").count(), 2);
}

#[test]
fn cli_pymol_graphics_ring_17() {
    let balls = b"0 0 2 1\n\
        0 1 0 0.5\n0.382683 0.92388 0 0.5\n0.707107 0.707107 0 0.5\n\
        0.92388 0.382683 0 0.5\n1 0 0 0.5\n0.92388 -0.382683 0 0.5\n\
        0.707107 -0.707107 0 0.5\n0.382683 -0.92388 0 0.5\n0 -1 0 0.5\n\
        -0.382683 -0.92388 0 0.5\n-0.707107 -0.707107 0 0.5\n\
        -0.92388 -0.382683 0 0.5\n-1 0 0 0.5\n-0.92388 0.382683 0 0.5\n\
        -0.707107 0.707107 0 0.5\n-0.382683 0.92388 0 0.5\n";

    let content = run_graphics_test!("test_graphics_ring.py", balls);

    assert_eq!(content.matches("SPHERE,").count(), 17);
    assert!(content.matches("TRIANGLE_FAN").count() > 0);
}
