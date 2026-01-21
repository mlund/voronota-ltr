//! Tests for UpdateableTessellation

use std::fs;
use voronota_ltr::{Ball, PeriodicBox, UpdateableTessellation, compute_tessellation};

/// Macro for approximate equality with context
macro_rules! assert_approx {
    ($actual:expr, $expected:expr, $eps:expr, $($arg:tt)*) => {
        let actual = $actual;
        let expected = $expected;
        let diff = (actual - expected).abs();
        assert!(
            diff < $eps,
            "{}: expected {}, got {} (diff={})",
            format!($($arg)*),
            expected,
            actual,
            diff
        );
    };
}

/// Dataset definition for reuse across tests
struct Dataset {
    balls: Vec<Ball>,
    probe: f64,
}

/// Macro to define test datasets that can be reused
macro_rules! define_datasets {
    ($($name:ident => { balls: $balls:expr, probe: $probe:expr }),* $(,)?) => {
        mod datasets {
            use super::*;

            $(
                pub fn $name() -> Dataset {
                    Dataset {
                        balls: $balls,
                        probe: $probe,
                    }
                }
            )*
        }
    };
}

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

define_datasets! {
    ring_17 => { balls: ring_17_balls(), probe: 1.0 },
    three_line => { balls: vec![
        Ball::new(0.0, 0.0, 0.0, 1.0),
        Ball::new(2.0, 0.0, 0.0, 1.0),
        Ball::new(4.0, 0.0, 0.0, 1.0),
    ], probe: 0.5 },
}

/// Macro to run updateable vs full comparison test
macro_rules! test_updateable_matches_full {
    ($name:ident, $dataset:expr, $periodic_box:expr, $changed_ids:expr, $num_iters:expr) => {
        #[test]
        fn $name() {
            let dataset = $dataset;
            let mut balls = dataset.balls;
            let probe = dataset.probe;
            let pbox: Option<&PeriodicBox> = $periodic_box;

            let mut tess = UpdateableTessellation::with_backup();
            assert!(
                tess.init(&balls, probe, pbox),
                "Failed to init tessellation"
            );

            for iter in 0..$num_iters {
                let changed: &[usize] = &$changed_ids;

                // Perturb changed balls
                for &id in changed {
                    balls[id].x += 0.05 * (iter as f64 + 1.0);
                }

                // Update incrementally
                assert!(
                    tess.update_with_changed(&balls, changed),
                    "Failed to update at iter {}",
                    iter
                );

                // Compute full tessellation for comparison
                let full = compute_tessellation(&balls, probe, pbox, None);
                let incr = tess.summary();

                // Compare results
                assert_eq!(
                    incr.contacts.len(),
                    full.contacts.len(),
                    "Iter {}: contact count mismatch",
                    iter
                );
                assert_eq!(
                    incr.cells.len(),
                    full.cells.len(),
                    "Iter {}: cell count mismatch",
                    iter
                );

                let incr_total_area: f64 = incr.contacts.iter().map(|c| c.area).sum();
                let full_total_area: f64 = full.contacts.iter().map(|c| c.area).sum();
                assert_approx!(
                    incr_total_area,
                    full_total_area,
                    0.1,
                    "Iter {}: total contact area",
                    iter
                );

                let incr_total_sas: f64 = incr.cells.iter().map(|c| c.sas_area).sum();
                let full_total_sas: f64 = full.cells.iter().map(|c| c.sas_area).sum();
                assert_approx!(
                    incr_total_sas,
                    full_total_sas,
                    0.1,
                    "Iter {}: total SAS area",
                    iter
                );
            }

            // Test restore
            assert!(tess.restore(), "Failed to restore");
            let restored = tess.summary();

            // Sanity check that restore did something
            assert!(
                !restored.contacts.is_empty(),
                "Restored should have contacts"
            );
        }
    };
}

// Tests using the macro
test_updateable_matches_full!(
    test_updateable_ring_17_no_periodic,
    datasets::ring_17(),
    None,
    [0, 8],
    3
);

test_updateable_matches_full!(
    test_updateable_three_line,
    datasets::three_line(),
    None,
    [1],
    5
);

/// Port of C++ api_usage_example_updateable_periodic.cpp pattern.
/// Verifies UpdateableTessellation produces identical results to compute_tessellation.
#[test]
fn test_updateable_ring_periodic_cpp_port() {
    let mut balls = ring_17_balls();
    let probe = 1.0;
    let pbox = PeriodicBox::from_corners((-1.6, -1.6, -0.6), (1.6, 1.6, 3.1));

    let mut tess = UpdateableTessellation::with_backup();
    assert!(tess.init(&balls, probe, Some(&pbox)));

    // Verify initial state matches compute_tessellation
    let full_init = compute_tessellation(&balls, probe, Some(&pbox), None);
    let up_init = tess.summary();

    assert_eq!(
        up_init.contacts.len(),
        full_init.contacts.len(),
        "Initial contact count"
    );
    assert_eq!(
        up_init.cells.len(),
        full_init.cells.len(),
        "Initial cell count"
    );

    let up_area: f64 = up_init.contacts.iter().map(|c| c.area).sum();
    let full_area: f64 = full_init.contacts.iter().map(|c| c.area).sum();
    assert_approx!(up_area, full_area, 0.01, "Initial total contact area");

    // Do 5 updates, moving spheres 0 and 1 by +0.1 in x each time
    let changed_ids = [0, 1];
    for iter in 0..5 {
        for &id in &changed_ids {
            balls[id].x += 0.1;
        }
        assert!(tess.update_with_changed(&balls, &changed_ids));

        // Compare with full recompute
        let full = compute_tessellation(&balls, probe, Some(&pbox), None);
        let up = tess.summary();

        assert_eq!(
            up.contacts.len(),
            full.contacts.len(),
            "Iter {}: contact count",
            iter
        );
        assert_eq!(
            up.cells.len(),
            full.cells.len(),
            "Iter {}: cell count",
            iter
        );

        let up_area: f64 = up.contacts.iter().map(|c| c.area).sum();
        let full_area: f64 = full.contacts.iter().map(|c| c.area).sum();
        assert_approx!(up_area, full_area, 0.1, "Iter {}: total contact area", iter);

        let up_sas: f64 = up.cells.iter().map(|c| c.sas_area).sum();
        let full_sas: f64 = full.cells.iter().map(|c| c.sas_area).sum();
        assert_approx!(up_sas, full_sas, 0.01, "Iter {}: total SAS area", iter);

        let up_vol: f64 = up.cells.iter().map(|c| c.volume).sum();
        let full_vol: f64 = full.cells.iter().map(|c| c.volume).sum();
        assert_approx!(up_vol, full_vol, 0.01, "Iter {}: total volume", iter);
    }

    // Test restore functionality
    let before_restore = tess.summary();
    assert!(tess.restore());
    let after_restore = tess.summary();

    // Restored state should differ from final state
    let before_area: f64 = before_restore.contacts.iter().map(|c| c.area).sum();
    let after_area: f64 = after_restore.contacts.iter().map(|c| c.area).sum();
    assert!(
        (before_area - after_area).abs() > 0.01,
        "Restore should change state"
    );
}

#[test]
fn test_updateable_exclusion() {
    let balls = vec![
        Ball::new(0.0, 0.0, 0.0, 1.0),
        Ball::new(2.0, 0.0, 0.0, 1.0),
        Ball::new(4.0, 0.0, 0.0, 1.0),
    ];

    let mut tess = UpdateableTessellation::with_backup();
    assert!(tess.init(&balls, 0.5, None));

    let before = tess.summary();
    assert_eq!(before.contacts.len(), 2);
    assert_eq!(before.cells.len(), 3);

    // Exclude middle sphere
    assert!(tess.set_exclusion(1, true));

    let after = tess.summary();
    // With middle sphere excluded, no contacts should remain
    // (spheres 0 and 2 don't overlap)
    assert_eq!(after.contacts.len(), 0);
    assert_eq!(after.cells.len(), 2);

    // Restore
    assert!(tess.restore());
    let restored = tess.summary();
    assert_eq!(restored.contacts.len(), 2);
    assert_eq!(restored.cells.len(), 3);
}

#[test]
fn test_updateable_auto_detect_changes() {
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
fn test_updateable_no_changes() {
    let balls = vec![Ball::new(0.0, 0.0, 0.0, 1.0), Ball::new(2.0, 0.0, 0.0, 1.0)];

    let mut tess = UpdateableTessellation::new();
    assert!(tess.init(&balls, 0.5, None));

    // Update with same balls - should return false
    assert!(!tess.update(&balls));
}

#[test]
fn test_updateable_result_api() {
    let balls = vec![
        Ball::new(0.0, 0.0, 0.0, 1.0),
        Ball::new(2.0, 0.0, 0.0, 1.0),
        Ball::new(4.0, 0.0, 0.0, 1.0),
    ];

    let mut tess = UpdateableTessellation::new();
    assert!(tess.init(&balls, 0.5, None));

    let result = tess.result();
    assert_eq!(result.num_spheres(), 3);

    // Check contacts for middle sphere
    let contacts: Vec<_> = result.contacts_for_sphere(1).collect();
    assert_eq!(contacts.len(), 2); // contacts with 0 and 2
}

#[test]
fn test_updateable_large_change_triggers_full_reinit() {
    let n = 20;
    let balls: Vec<Ball> = (0..n)
        .map(|i| Ball::new(i as f64 * 2.0, 0.0, 0.0, 1.0))
        .collect();

    let mut tess = UpdateableTessellation::new();
    assert!(tess.init(&balls, 0.5, None));

    // Change more than half the spheres - should trigger full reinit
    let changed_ids: Vec<usize> = (0..n / 2 + 1).collect();
    let mut modified = balls.clone();
    for &id in &changed_ids {
        modified[id].x += 0.1;
    }

    assert!(tess.update_with_changed(&modified, &changed_ids));
    assert!(tess.last_update_was_full_reinit());
}

// Test with real dataset if available
#[test]
fn test_updateable_balls_2zsk() {
    let path = "benches/data/balls_2zsk.xyzr";
    if !std::path::Path::new(path).exists() {
        eprintln!("Skipping test_updateable_balls_2zsk: file not found");
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
    let full = compute_tessellation(&balls, probe, None, None);
    let incr = tess.summary();

    assert_eq!(incr.contacts.len(), full.contacts.len());
    assert_eq!(incr.cells.len(), full.cells.len());
}
