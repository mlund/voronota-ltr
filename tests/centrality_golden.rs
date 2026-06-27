//! Golden test for the contact `central` flag against the C++ voronota-lt
//! reference. The fixture below is identical to `tests/data/central_golden.cpp`;
//! that harness, run against the header-only voronota-lt, produced
//! `tests/data/central_golden.tsv`. This test rebuilds the same tessellation and
//! asserts every contact's `Contact.central` matches the C++ `flags` bit.

use std::collections::HashMap;

use voronota_ltr::{Ball, compute_tessellation};

/// The 8-ball fixture — MUST match `tests/data/central_golden.cpp` (radius 1.5).
const POINTS: [[f64; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [2.5, 0.0, 0.0],
    [1.25, 2.0, 0.0],
    [1.25, 0.8, 2.0],
    [0.0, 2.5, 1.0],
    [2.5, 2.5, 1.0],
    [1.25, -1.5, 1.0],
    [3.5, 1.25, 1.0],
];

fn ordered(a: usize, b: usize) -> (usize, usize) {
    if a <= b { (a, b) } else { (b, a) }
}

#[test]
fn central_flag_matches_cpp_voronota_lt() {
    let balls: Vec<Ball> = POINTS
        .iter()
        .map(|p| Ball::new(p[0], p[1], p[2], 1.5))
        .collect();
    let result = compute_tessellation(&balls, 1.4, None, None, false);

    let computed: HashMap<(usize, usize), bool> = result
        .contacts
        .iter()
        .map(|c| (ordered(c.id_a, c.id_b), c.central))
        .collect();

    let golden = include_str!("data/central_golden.tsv");
    let mut checked = 0;
    for line in golden.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let a: usize = it.next().unwrap().parse().unwrap();
        let b: usize = it.next().unwrap().parse().unwrap();
        let want = it.next().unwrap() == "1";

        let got = *computed
            .get(&ordered(a, b))
            .unwrap_or_else(|| panic!("contact ({a}, {b}) missing from Rust tessellation"));
        assert_eq!(got, want, "central mismatch for contact ({a}, {b})");
        checked += 1;
    }

    // Every Rust contact must be covered by the golden (no extras either way).
    assert_eq!(
        checked,
        computed.len(),
        "contact-count mismatch: {checked} golden vs {} computed",
        computed.len()
    );
}
