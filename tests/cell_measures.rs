use approx::assert_abs_diff_eq;
use voronota_ltr::{Ball, CellMeasure, Results, compute_tessellation};

#[test]
fn detached_and_hidden_balls_have_distinct_volume_states() {
    let balls = vec![
        Ball::new(0.0, 0.0, 0.0, 10.0),
        Ball::new(0.0, 0.0, 0.0, 1.0),
        Ball::new(40.0, 0.0, 0.0, 2.0),
    ];

    let volumes = compute_tessellation(&balls, 0.0, None, None, false).volumes();

    assert_eq!(volumes.len(), balls.len());
    let CellMeasure::Computed(big_volume) = volumes[0] else {
        panic!("the exposed ball should have a computed cell");
    };
    assert_abs_diff_eq!(
        big_volume,
        4.0 / 3.0 * std::f64::consts::PI * 10.0f64.powi(3),
        epsilon = 1e-10
    );
    assert_eq!(volumes[1], CellMeasure::Empty);
    let CellMeasure::Computed(detached_volume) = volumes[2] else {
        panic!("the detached ball should have a computed cell");
    };
    assert_abs_diff_eq!(
        detached_volume,
        4.0 / 3.0 * std::f64::consts::PI * 2.0f64.powi(3),
        epsilon = 1e-10
    );
}

#[test]
fn group_filtered_cells_are_not_reported_as_measured() {
    let balls = vec![
        Ball::new(0.0, 0.0, 0.0, 1.0),
        Ball::new(1.5, 0.0, 0.0, 1.0),
        Ball::new(3.0, 0.0, 0.0, 1.0),
    ];
    let groups = [0, 1, 1];

    let volumes = compute_tessellation(&balls, 0.5, None, Some(&groups), false).volumes();

    assert_eq!(
        volumes,
        vec![
            CellMeasure::NotComputed,
            CellMeasure::NotComputed,
            CellMeasure::NotComputed,
        ]
    );
}

#[test]
#[cfg(feature = "serde")]
fn group_filter_state_survives_serde_roundtrip() {
    let balls = vec![
        Ball::new(0.0, 0.0, 0.0, 1.0),
        Ball::new(1.5, 0.0, 0.0, 1.0),
        Ball::new(3.0, 0.0, 0.0, 1.0),
    ];
    let groups = [0, 1, 1];
    let result = compute_tessellation(&balls, 0.5, None, Some(&groups), false);

    let json = serde_json::to_string(&result).expect("result should serialize");
    let restored: voronota_ltr::TessellationResult =
        serde_json::from_str(&json).expect("result should deserialize");

    assert_eq!(
        restored.volumes(),
        vec![
            CellMeasure::NotComputed,
            CellMeasure::NotComputed,
            CellMeasure::NotComputed,
        ]
    );
}
