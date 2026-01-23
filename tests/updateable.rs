//! Tests for `UpdateableTessellation`

#![allow(clippy::unreadable_literal, clippy::approx_constant)]

mod common;

use std::fs;

use approx::assert_abs_diff_eq;
use voronota_ltr::{Ball, PeriodicBox, UpdateableTessellation, compute_tessellation};

fn load_xyzr(path: &str) -> Vec<Ball> {
    let content = fs::read_to_string(path).expect("Failed to read file");
    content
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 4 {
                return None;
            }
            let n = parts.len();
            Some(Ball::new(
                parts[n - 4].parse().ok()?,
                parts[n - 3].parse().ok()?,
                parts[n - 2].parse().ok()?,
                parts[n - 1].parse().ok()?,
            ))
        })
        .collect()
}

fn ring_17_balls() -> Vec<Ball> {
    vec![
        Ball::new(0.0, 0.0, 2.0, 1.0),
        Ball::new(0.0, 1.0, 0.0, 0.5),
        Ball::new(0.382683, 0.92388, 0.0, 0.5),
        Ball::new(0.707107, 0.707107, 0.0, 0.5),
        Ball::new(0.92388, 0.382683, 0.0, 0.5),
        Ball::new(1.0, 0.0, 0.0, 0.5),
        Ball::new(0.92388, -0.382683, 0.0, 0.5),
        Ball::new(0.707107, -0.707107, 0.0, 0.5),
        Ball::new(0.382683, -0.92388, 0.0, 0.5),
        Ball::new(0.0, -1.0, 0.0, 0.5),
        Ball::new(-0.382683, -0.92388, 0.0, 0.5),
        Ball::new(-0.707107, -0.707107, 0.0, 0.5),
        Ball::new(-0.92388, -0.382683, 0.0, 0.5),
        Ball::new(-1.0, 0.0, 0.0, 0.5),
        Ball::new(-0.92388, 0.382683, 0.0, 0.5),
        Ball::new(-0.707107, 0.707107, 0.0, 0.5),
        Ball::new(-0.382683, 0.92388, 0.0, 0.5),
    ]
}

fn three_line_balls() -> Vec<Ball> {
    vec![
        Ball::new(0.0, 0.0, 0.0, 1.0),
        Ball::new(2.0, 0.0, 0.0, 1.0),
        Ball::new(4.0, 0.0, 0.0, 1.0),
    ]
}

/// Runs updateable tessellation and compares with full recompute over multiple iterations.
fn verify_updateable_matches_full(
    mut balls: Vec<Ball>,
    probe: f64,
    pbox: Option<&PeriodicBox>,
    changed_ids: &[usize],
    num_iters: usize,
) {
    let mut tess = UpdateableTessellation::with_backup();
    assert!(
        tess.init(&balls, probe, pbox),
        "Failed to init tessellation"
    );

    for iter in 0..num_iters {
        // Perturb changed balls
        for &id in changed_ids {
            balls[id].x += 0.05 * (iter as f64 + 1.0);
        }

        // Update incrementally
        assert!(
            tess.update_with_changed(&balls, changed_ids),
            "Failed to update at iter {iter}"
        );

        // Compute full tessellation for comparison
        let full = compute_tessellation(&balls, probe, pbox, None, false);
        let incr = tess.summary();

        // Compare results
        assert_eq!(
            incr.contacts.len(),
            full.contacts.len(),
            "Iter {iter}: contact count mismatch"
        );
        assert_eq!(
            incr.cells.len(),
            full.cells.len(),
            "Iter {iter}: cell count mismatch"
        );

        let incr_total_area: f64 = incr.contacts.iter().map(|c| c.area).sum();
        let full_total_area: f64 = full.contacts.iter().map(|c| c.area).sum();
        assert_abs_diff_eq!(incr_total_area, full_total_area, epsilon = 0.1);

        let incr_total_sas: f64 = incr.cells.iter().map(|c| c.sas_area).sum();
        let full_total_sas: f64 = full.cells.iter().map(|c| c.sas_area).sum();
        assert_abs_diff_eq!(incr_total_sas, full_total_sas, epsilon = 0.1);
    }

    // Test restore
    assert!(tess.restore(), "Failed to restore");
    let restored = tess.summary();
    assert!(
        !restored.contacts.is_empty(),
        "Restored should have contacts"
    );
}

#[test]
fn updateable_ring_17_no_periodic() {
    verify_updateable_matches_full(ring_17_balls(), 1.0, None, &[0, 8], 3);
}

#[test]
fn updateable_three_line() {
    verify_updateable_matches_full(three_line_balls(), 0.5, None, &[1], 5);
}

/// Port of C++ `api_usage_example_updateable_periodic.cpp` pattern.
#[test]
fn updateable_ring_periodic_cpp_port() {
    let mut balls = ring_17_balls();
    let probe = 1.0;
    let pbox = PeriodicBox::from_corners((-1.6, -1.6, -0.6), (1.6, 1.6, 3.1));

    let mut tess = UpdateableTessellation::with_backup();
    assert!(tess.init(&balls, probe, Some(&pbox)));

    // Verify initial state matches compute_tessellation
    let full_init = compute_tessellation(&balls, probe, Some(&pbox), None, false);
    let up_init = tess.summary();

    assert_eq!(up_init.contacts.len(), full_init.contacts.len());
    assert_eq!(up_init.cells.len(), full_init.cells.len());

    let up_area: f64 = up_init.contacts.iter().map(|c| c.area).sum();
    let full_area: f64 = full_init.contacts.iter().map(|c| c.area).sum();
    assert_abs_diff_eq!(up_area, full_area, epsilon = 0.01);

    // Do 5 updates, moving spheres 0 and 1 by +0.1 in x each time
    let changed_ids = [0, 1];
    for _ in 0..5 {
        for &id in &changed_ids {
            balls[id].x += 0.1;
        }
        assert!(tess.update_with_changed(&balls, &changed_ids));

        // Compare with full recompute
        let full = compute_tessellation(&balls, probe, Some(&pbox), None, false);
        let up = tess.summary();

        assert_eq!(up.contacts.len(), full.contacts.len());
        assert_eq!(up.cells.len(), full.cells.len());

        let up_area: f64 = up.contacts.iter().map(|c| c.area).sum();
        let full_area: f64 = full.contacts.iter().map(|c| c.area).sum();
        assert_abs_diff_eq!(up_area, full_area, epsilon = 0.1);

        let up_sas: f64 = up.cells.iter().map(|c| c.sas_area).sum();
        let full_sas: f64 = full.cells.iter().map(|c| c.sas_area).sum();
        assert_abs_diff_eq!(up_sas, full_sas, epsilon = 0.01);

        let up_vol: f64 = up.cells.iter().map(|c| c.volume).sum();
        let full_vol: f64 = full.cells.iter().map(|c| c.volume).sum();
        assert_abs_diff_eq!(up_vol, full_vol, epsilon = 0.01);
    }

    // Test restore functionality
    let before_restore = tess.summary();
    assert!(tess.restore());
    let after_restore = tess.summary();

    let before_area: f64 = before_restore.contacts.iter().map(|c| c.area).sum();
    let after_area: f64 = after_restore.contacts.iter().map(|c| c.area).sum();
    assert!(
        (before_area - after_area).abs() > 0.01,
        "Restore should change state"
    );
}

#[test]
fn updateable_exclusion() {
    let balls = three_line_balls();

    let mut tess = UpdateableTessellation::with_backup();
    assert!(tess.init(&balls, 0.5, None));

    let before = tess.summary();
    assert_eq!(before.contacts.len(), 2);
    assert_eq!(before.cells.len(), 3);

    // Exclude middle sphere
    assert!(tess.set_exclusion(1, true));

    let after = tess.summary();
    // With middle sphere excluded, no contacts remain (spheres 0 and 2 don't overlap)
    assert_eq!(after.contacts.len(), 0);
    assert_eq!(after.cells.len(), 2);

    // Restore
    assert!(tess.restore());
    let restored = tess.summary();
    assert_eq!(restored.contacts.len(), 2);
    assert_eq!(restored.cells.len(), 3);
}

#[test]
fn updateable_auto_detect_changes() {
    let mut balls = vec![Ball::new(0.0, 0.0, 0.0, 1.0), Ball::new(2.0, 0.0, 0.0, 1.0)];

    let mut tess = UpdateableTessellation::new();
    assert!(tess.init(&balls, 0.5, None));

    let s1 = tess.summary();

    // Modify first ball and use auto-detect
    balls[0].x += 0.1;
    assert!(tess.update(&balls));

    let s2 = tess.summary();
    assert_eq!(s1.contacts.len(), s2.contacts.len());

    // Verify changed IDs were detected
    assert!(!tess.changed_ids().is_empty());
    assert!(tess.changed_ids().contains(&0));
}

#[test]
fn updateable_no_changes() {
    let balls = vec![Ball::new(0.0, 0.0, 0.0, 1.0), Ball::new(2.0, 0.0, 0.0, 1.0)];

    let mut tess = UpdateableTessellation::new();
    assert!(tess.init(&balls, 0.5, None));

    // Update with same balls - should return false
    assert!(!tess.update(&balls));
}

#[test]
fn updateable_result_api() {
    let balls = three_line_balls();

    let mut tess = UpdateableTessellation::new();
    assert!(tess.init(&balls, 0.5, None));

    let result = tess.result();
    assert_eq!(result.num_spheres(), 3);

    // Check contacts for middle sphere (contacts with 0 and 2)
    assert_eq!(result.contacts_for_sphere(1).count(), 2);
}

#[test]
fn updateable_large_change_triggers_full_reinit() {
    let n = 20;
    let balls: Vec<Ball> = (0..n)
        .map(|i| Ball::new(i as f64 * 2.0, 0.0, 0.0, 1.0))
        .collect();

    let mut tess = UpdateableTessellation::new();
    assert!(tess.init(&balls, 0.5, None));

    // Change more than half the spheres - should trigger full reinit
    let changed_ids: Vec<usize> = (0..=n / 2).collect();
    let mut modified = balls.clone();
    for &id in &changed_ids {
        modified[id].x += 0.1;
    }

    assert!(tess.update_with_changed(&modified, &changed_ids));
    assert!(tess.last_update_was_full_reinit());
}

#[test]
fn updateable_balls_2zsk() {
    let path = "benches/data/balls_2zsk.xyzr";
    if !std::path::Path::new(path).exists() {
        eprintln!("Skipping updateable_balls_2zsk: file not found");
        return;
    }

    let mut balls = load_xyzr(path);
    let probe = 1.4;

    let mut tess = UpdateableTessellation::new();
    assert!(tess.init(&balls, probe, None));

    let init_summary = tess.summary();
    assert_eq!(init_summary.contacts.len(), 23855);
    assert_eq!(init_summary.cells.len(), 3545);

    // Update a few spheres
    let changed_ids = [0, 100, 500];
    for &id in &changed_ids {
        balls[id].x += 0.1;
    }
    assert!(tess.update_with_changed(&balls, &changed_ids));

    // Should be incremental (not full reinit)
    assert!(!tess.last_update_was_full_reinit());

    // Compare with full
    let full = compute_tessellation(&balls, probe, None, None, false);
    let incr = tess.summary();

    assert_eq!(incr.contacts.len(), full.contacts.len());
    assert_eq!(incr.cells.len(), full.cells.len());
}
