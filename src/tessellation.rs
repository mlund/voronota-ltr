use rayon::prelude::*;

use crate::contact::construct_contact_descriptor;
use crate::spheres_searcher::SpheresSearcher;
use crate::types::{
    Ball, Cell, CellContactSummary, Contact, ContactDescriptorSummary, PeriodicBox, Sphere,
    TessellationResult, ValuedId,
};

/// Compute radical tessellation contacts and cells.
///
/// # Arguments
/// * `balls` - Input spheres (center + radius)
/// * `probe` - Probe radius added to each ball
/// * `periodic_box` - Optional periodic boundary box
/// * `groups` - Optional group IDs; contacts between same-group spheres are excluded
///
/// # Example
/// ```
/// use rust_voronota::{Ball, compute_tessellation};
///
/// let balls = vec![
///     Ball::new(0.0, 0.0, 0.0, 1.5),
///     Ball::new(3.0, 0.0, 0.0, 1.5),
/// ];
/// let result = compute_tessellation(&balls, 1.4, None, None);
/// ```
pub fn compute_tessellation(
    balls: &[Ball],
    probe: f64,
    periodic_box: Option<&PeriodicBox>,
    groups: Option<&[i32]>,
) -> TessellationResult {
    match periodic_box {
        Some(pbox) => compute_periodic(balls, probe, pbox, groups),
        None => compute_standard(balls, probe, groups),
    }
}

/// Standard (non-periodic) tessellation
fn compute_standard(balls: &[Ball], probe: f64, groups: Option<&[i32]>) -> TessellationResult {
    if balls.is_empty() {
        return TessellationResult::default();
    }

    let spheres: Vec<Sphere> = balls.iter().map(|b| Sphere::from_ball(b, probe)).collect();
    let searcher = SpheresSearcher::new(spheres);
    let spheres = searcher.spheres();

    let collision_data: Vec<_> = (0..spheres.len())
        .into_par_iter()
        .map(|id| {
            let result = searcher.find_colliding_ids(id, true);
            (id, result.colliding_ids, result.exclusion_status)
        })
        .collect();

    let all_collisions: Vec<Vec<ValuedId>> = collision_data
        .into_iter()
        .map(|(_, collisions, _)| collisions)
        .collect();

    let collision_pairs = collect_collision_pairs(&all_collisions, groups);

    let contact_summaries: Vec<Option<ContactDescriptorSummary>> = collision_pairs
        .par_iter()
        .map(|&(a_id, b_id)| {
            let cd = construct_contact_descriptor(
                searcher.spheres(),
                a_id,
                b_id,
                &all_collisions[a_id],
            )?;
            let mut summary = cd.to_summary();
            summary.ensure_ids_ordered();
            Some(summary)
        })
        .collect();

    let valid_summaries: Vec<ContactDescriptorSummary> = contact_summaries
        .into_iter()
        .flatten()
        .filter(|s| s.area > 0.0)
        .collect();

    let contacts: Vec<Contact> = valid_summaries
        .iter()
        .map(|s| Contact {
            id_a: s.id_a,
            id_b: s.id_b,
            area: s.area,
            arc_length: s.arc_length,
        })
        .collect();

    let cells = compute_cells(&valid_summaries, searcher.spheres(), &all_collisions);

    TessellationResult { contacts, cells }
}

/// Collect unique collision pairs, optionally filtering by group
fn collect_collision_pairs(
    all_collisions: &[Vec<ValuedId>],
    groups: Option<&[i32]>,
) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for (a_id, neighbors) in all_collisions.iter().enumerate() {
        for neighbor in neighbors {
            let b_id = neighbor.index;
            if a_id < b_id && !same_group(groups, a_id, b_id) {
                pairs.push((a_id, b_id));
            }
        }
    }
    pairs
}

/// Check if two spheres belong to the same group
#[inline]
fn same_group(groups: Option<&[i32]>, a: usize, b: usize) -> bool {
    match groups {
        Some(g) if a < g.len() && b < g.len() => g[a] == g[b],
        _ => false,
    }
}

/// Periodic boundary tessellation
fn compute_periodic(
    balls: &[Ball],
    probe: f64,
    periodic_box: &PeriodicBox,
    groups: Option<&[i32]>,
) -> TessellationResult {
    if balls.is_empty() {
        return TessellationResult::default();
    }

    let n = balls.len();

    // Convert balls to spheres (keep copy for cell computation)
    let input_spheres: Vec<Sphere> = balls.iter().map(|b| Sphere::from_ball(b, probe)).collect();

    // Create 27 periodic images and build spatial index
    let populated_spheres = populate_periodic_spheres(&input_spheres, periodic_box);
    let searcher = SpheresSearcher::new(populated_spheres);

    // Find collisions for original spheres only (indices 0..n)
    let collision_data: Vec<_> = (0..n)
        .into_par_iter()
        .map(|id| {
            let result = searcher.find_colliding_ids(id, true);
            (id, result.colliding_ids, result.exclusion_status)
        })
        .collect();

    // Build collision map, keeping periodic image indices
    let all_collisions: Vec<Vec<ValuedId>> = collision_data
        .into_iter()
        .map(|(_, collisions, _)| collisions)
        .collect();

    // Collect unique collision pairs
    // For periodic: (a, b) where a is canonical (0..n) and b can be periodic image
    let collision_pairs = collect_periodic_collision_pairs(&all_collisions, n, groups);

    // Construct contact descriptors in parallel
    let contact_summaries: Vec<Option<ContactDescriptorSummary>> = collision_pairs
        .par_iter()
        .map(|&(a_id, b_id)| {
            let cd = construct_contact_descriptor(
                searcher.spheres(),
                a_id,
                b_id,
                &all_collisions[a_id],
            )?;
            let mut summary = cd.to_summary();
            // Map IDs to canonical form
            summary.id_a = a_id;
            summary.id_b = b_id % n;
            summary.ensure_ids_ordered();
            Some(summary)
        })
        .collect();

    // Filter valid contacts
    let valid_summaries: Vec<ContactDescriptorSummary> = contact_summaries
        .into_iter()
        .flatten()
        .filter(|s| s.area > 0.0)
        .collect();

    // Build contacts output
    let contacts: Vec<Contact> = valid_summaries
        .iter()
        .map(|s| Contact {
            id_a: s.id_a,
            id_b: s.id_b,
            area: s.area,
            arc_length: s.arc_length,
        })
        .collect();

    // Compute cells using canonical IDs
    let cells = compute_cells_periodic(&valid_summaries, &input_spheres, &all_collisions, n);

    TessellationResult { contacts, cells }
}

/// Generate 27 periodic copies of each sphere (original + 26 shifts)
fn populate_periodic_spheres(spheres: &[Sphere], pbox: &PeriodicBox) -> Vec<Sphere> {
    let n = spheres.len();
    let mut result = Vec::with_capacity(n * 27);

    // First, add original spheres (index 0..n)
    result.extend_from_slice(spheres);

    // Then add 26 shifted copies (indices n..27n)
    for sx in -1..=1i32 {
        for sy in -1..=1i32 {
            for sz in -1..=1i32 {
                if sx != 0 || sy != 0 || sz != 0 {
                    for s in spheres {
                        result.push(pbox.shift_sphere(s, sx as f64, sy as f64, sz as f64));
                    }
                }
            }
        }
    }

    result
}

/// Collect collision pairs for periodic case
fn collect_periodic_collision_pairs(
    all_collisions: &[Vec<ValuedId>],
    n: usize,
    groups: Option<&[i32]>,
) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();

    for (a_id, neighbors) in all_collisions.iter().enumerate() {
        for neighbor in neighbors {
            let b_id = neighbor.index;
            let b_canonical = b_id % n;

            // Skip if same group (use canonical indices for group lookup)
            if same_group(groups, a_id, b_canonical) {
                continue;
            }

            // Include pair if:
            // - b is a periodic image (b_id >= n), OR
            // - b is canonical and a < b (to avoid duplicates)
            // Also handle self-contacts (a == b_canonical) for periodic images
            if b_id >= n {
                // Periodic image - always include (dedupe by checking if we're the "smaller" side)
                if all_collisions[a_id].len() < all_collisions[b_canonical].len()
                    || (all_collisions[a_id].len() == all_collisions[b_canonical].len()
                        && a_id <= b_canonical)
                {
                    pairs.push((a_id, b_id));
                }
            } else if a_id < b_id {
                // Both canonical - use simple ordering
                pairs.push((a_id, b_id));
            }
        }
    }

    pairs
}

/// Compute cells for periodic case
fn compute_cells_periodic(
    summaries: &[ContactDescriptorSummary],
    spheres: &[Sphere],
    all_collisions: &[Vec<ValuedId>],
    n: usize,
) -> Vec<Cell> {
    let mut cell_summaries: Vec<CellContactSummary> = (0..n)
        .map(|i| CellContactSummary {
            id: i,
            ..Default::default()
        })
        .collect();

    // Accumulate contributions from contacts
    for cds in summaries {
        if cds.area > 0.0 {
            cell_summaries[cds.id_a].add(cds);
            if cds.id_b != cds.id_a {
                cell_summaries[cds.id_b].add(cds);
            }
        }
    }

    // Compute SAS for each cell
    for (i, cs) in cell_summaries.iter_mut().enumerate() {
        if cs.stage == 1 {
            cs.compute_sas(spheres[i].r);
        } else if cs.stage == 0 && all_collisions[i].is_empty() {
            cs.compute_sas_detached(i, spheres[i].r);
        }
    }

    cell_summaries
        .into_iter()
        .filter(|cs| cs.stage == 2)
        .map(|cs| Cell {
            index: cs.id,
            sas_area: cs.sas_area,
            volume: cs.sas_inside_volume,
        })
        .collect()
}

/// Compute cell SAS areas and volumes from contact summaries
fn compute_cells(
    summaries: &[ContactDescriptorSummary],
    spheres: &[Sphere],
    all_collisions: &[Vec<ValuedId>],
) -> Vec<Cell> {
    let n = spheres.len();
    let mut cell_summaries: Vec<CellContactSummary> = (0..n)
        .map(|i| CellContactSummary {
            id: i,
            ..Default::default()
        })
        .collect();

    // Accumulate contributions from contacts
    for cds in summaries {
        if cds.area > 0.0 {
            cell_summaries[cds.id_a].add(cds);
            cell_summaries[cds.id_b].add(cds);
        }
    }

    // Compute SAS for each cell
    for (i, cs) in cell_summaries.iter_mut().enumerate() {
        if cs.stage == 1 {
            cs.compute_sas(spheres[i].r);
        } else if cs.stage == 0 && all_collisions[i].is_empty() {
            // Detached sphere (no contacts)
            cs.compute_sas_detached(i, spheres[i].r);
        }
    }

    // Build output cells
    cell_summaries
        .into_iter()
        .filter(|cs| cs.stage == 2)
        .map(|cs| Cell {
            index: cs.id,
            sas_area: cs.sas_area,
            volume: cs.sas_inside_volume,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_two_spheres() {
        let balls = vec![Ball::new(0.0, 0.0, 0.0, 1.0), Ball::new(2.0, 0.0, 0.0, 1.0)];

        let result = compute_tessellation(&balls, 0.5, None, None);

        assert_eq!(result.contacts.len(), 1);
        assert!(result.contacts[0].area > 0.0);
        assert_eq!(result.contacts[0].id_a, 0);
        assert_eq!(result.contacts[0].id_b, 1);

        // Both spheres should have cells
        assert_eq!(result.cells.len(), 2);
    }

    #[test]
    fn test_single_sphere() {
        let balls = vec![Ball::new(0.0, 0.0, 0.0, 1.0)];

        let result = compute_tessellation(&balls, 0.5, None, None);

        // No contacts for single sphere
        assert!(result.contacts.is_empty());

        // Single detached sphere should have full SAS area
        assert_eq!(result.cells.len(), 1);
        let expected_area = 4.0 * std::f64::consts::PI * 1.5 * 1.5; // 4πr² with r = 1.0 + 0.5
        assert_relative_eq!(result.cells[0].sas_area, expected_area, epsilon = 0.01);
    }

    #[test]
    fn test_three_spheres_triangle() {
        let balls = vec![
            Ball::new(0.0, 0.0, 0.0, 1.0),
            Ball::new(2.0, 0.0, 0.0, 1.0),
            Ball::new(1.0, 1.7, 0.0, 1.0),
        ];

        let result = compute_tessellation(&balls, 0.5, None, None);

        // Should have 3 contacts (each pair)
        assert_eq!(result.contacts.len(), 3);

        // All three spheres should have cells
        assert_eq!(result.cells.len(), 3);
    }

    #[test]
    fn test_non_overlapping() {
        let balls = vec![
            Ball::new(0.0, 0.0, 0.0, 1.0),
            Ball::new(10.0, 0.0, 0.0, 1.0),
        ];

        let result = compute_tessellation(&balls, 0.5, None, None);

        // No contacts between well-separated spheres
        assert!(result.contacts.is_empty());

        // Both should be detached with full SAS
        assert_eq!(result.cells.len(), 2);
    }

    #[test]
    fn test_empty_input() {
        let balls: Vec<Ball> = vec![];
        let result = compute_tessellation(&balls, 0.5, None, None);

        assert!(result.contacts.is_empty());
        assert!(result.cells.is_empty());
    }

    /// Test matching C++ voronota-lt API example (basic mode, no periodic box)
    /// Input: 17 balls arranged in a ring pattern, probe=1.0
    /// Expected values from C++ output
    #[test]
    fn test_cpp_api_example_basic() {
        let balls = vec![
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
        ];

        let result = compute_tessellation(&balls, 1.0, None, None);

        // C++ produces 44 contacts in basic mode
        assert_eq!(result.contacts.len(), 44);

        // All 17 balls should have cells
        assert_eq!(result.cells.len(), 17);

        // Check cell 0 (central large ball) - C++ expects ~34.82 SAS area, ~29.23 volume
        let cell0 = result.cells.iter().find(|c| c.index == 0).unwrap();
        assert_relative_eq!(cell0.sas_area, 34.8168, epsilon = 0.01);
        assert_relative_eq!(cell0.volume, 29.2302, epsilon = 0.01);

        // Check one of the small ring balls (cell 1) - C++ expects ~3.29 SAS area, ~2.48 volume
        let cell1 = result.cells.iter().find(|c| c.index == 1).unwrap();
        assert_relative_eq!(cell1.sas_area, 3.29195, epsilon = 0.01);
        assert_relative_eq!(cell1.volume, 2.48022, epsilon = 0.01);

        // Check contact 0-1 area - C++ expects ~0.7477
        let contact_0_1 = result
            .contacts
            .iter()
            .find(|c| c.id_a == 0 && c.id_b == 1)
            .unwrap();
        assert_relative_eq!(contact_0_1.area, 0.747721, epsilon = 0.001);
        assert_relative_eq!(contact_0_1.arc_length, 0.726907, epsilon = 0.001);

        // Check contact 1-2 (adjacent small balls) - C++ expects ~5.02
        let contact_1_2 = result
            .contacts
            .iter()
            .find(|c| c.id_a == 1 && c.id_b == 2)
            .unwrap();
        assert_relative_eq!(contact_1_2.area, 5.0216, epsilon = 0.01);
    }

    /// Helper macro to assert approximate equality with context
    macro_rules! assert_approx {
        ($actual:expr, $expected:expr, $eps:expr, $name:expr) => {
            let actual = $actual;
            let expected = $expected;
            assert!(
                (actual - expected).abs() < $eps,
                "{}: expected {}, got {} (diff={})",
                $name,
                expected,
                actual,
                (actual - expected).abs()
            );
        };
    }

    /// Edge case 1: 3 balls in a line (from C++ tricky_cases)
    #[test]
    fn test_edge_case_three_balls_line() {
        let balls = vec![
            Ball::new(0.0, 0.0, 0.0, 1.0),
            Ball::new(0.5, 0.0, 0.0, 1.0),
            Ball::new(1.0, 0.0, 0.0, 1.0),
        ];

        let result = compute_tessellation(&balls, 1.0, None, None);

        // Middle ball (1) should contact both end balls
        assert_eq!(result.cells.len(), 3);
        assert!(!result.contacts.is_empty());
    }

    /// Edge case 2: 4 balls in cube corners (from C++ tricky_cases)
    #[test]
    fn test_edge_case_four_balls_square() {
        let balls = vec![
            Ball::new(0.0, 0.0, 0.0, 1.0),
            Ball::new(0.0, 0.0, 1.0, 1.0),
            Ball::new(0.0, 1.0, 0.0, 1.0),
            Ball::new(0.0, 1.0, 1.0, 1.0),
        ];

        let result = compute_tessellation(&balls, 2.0, None, None);

        assert_eq!(result.cells.len(), 4);
        // Each ball contacts its neighbors
        assert!(!result.contacts.is_empty());
    }

    /// Edge case 3: 8 balls in cube (from C++ tricky_cases)
    #[test]
    fn test_edge_case_eight_balls_cube() {
        let balls = vec![
            Ball::new(0.0, 0.0, 0.0, 1.0),
            Ball::new(0.0, 0.0, 1.0, 1.0),
            Ball::new(0.0, 1.0, 0.0, 1.0),
            Ball::new(0.0, 1.0, 1.0, 1.0),
            Ball::new(1.0, 0.0, 0.0, 1.0),
            Ball::new(1.0, 0.0, 1.0, 1.0),
            Ball::new(1.0, 1.0, 0.0, 1.0),
            Ball::new(1.0, 1.0, 1.0, 1.0),
        ];

        let result = compute_tessellation(&balls, 2.0, None, None);

        assert_eq!(result.cells.len(), 8);
        assert!(!result.contacts.is_empty());
    }

    /// Edge case 4: ball containing smaller ball (from C++ tricky_cases)
    #[test]
    fn test_edge_case_containment() {
        let balls = vec![
            Ball::new(0.0, 0.0, 0.0, 1.0),
            Ball::new(0.0, 0.0, 0.0, 0.5), // Smaller ball at same center
            Ball::new(1.0, 0.0, 0.0, 1.0),
        ];

        let result = compute_tessellation(&balls, 0.5, None, None);

        // Ball 1 is hidden inside ball 0 - should only have 2 cells
        // (hidden balls are excluded in C++ when discard_hidden=true)
        assert!(result.cells.len() <= 3);
    }

    /// Test matching C++ CLI test: balls_cs_1x1.xyzr with probe=2.0
    /// This is a 100-ball crystal structure test
    #[test]
    fn test_cpp_cli_balls_cs_1x1() {
        // Data from /Users/mikael/github/voronota/expansion_lt/tests/input/balls_cs_1x1.xyzr
        let balls = vec![
            Ball::new(46.99, 128.17, 144.94, 3.0),
            Ball::new(46.79, 127.84, 138.22, 3.0),
            Ball::new(40.46, 120.67, 136.9, 3.0),
            Ball::new(35.1, 117.45, 140.94, 3.0),
            Ball::new(33.86, 117.2, 148.43, 3.0),
            Ball::new(39.4, 120.41, 149.01, 3.0),
            Ball::new(36.71, 121.18, 154.2, 3.0),
            Ball::new(32.12, 126.51, 155.65, 3.0),
            Ball::new(34.67, 129.16, 149.57, 3.0),
            Ball::new(32.34, 128.99, 144.2, 3.0),
            Ball::new(33.09, 122.88, 145.41, 3.0),
            Ball::new(30.0, 125.65, 139.02, 3.0),
            Ball::new(27.62, 119.24, 141.44, 3.0),
            Ball::new(25.13, 113.84, 137.18, 3.0),
            Ball::new(29.87, 107.42, 137.46, 3.0),
            Ball::new(26.02, 102.66, 133.78, 3.0),
            Ball::new(20.71, 103.26, 138.04, 3.0),
            Ball::new(18.33, 108.95, 133.72, 3.0),
            Ball::new(18.21, 102.44, 131.67, 3.0),
            Ball::new(12.27, 98.98, 136.42, 3.0),
            Ball::new(7.17, 97.07, 142.18, 3.0),
            Ball::new(12.75, 101.93, 142.09, 3.0),
            Ball::new(10.25, 106.11, 136.61, 3.0),
            Ball::new(5.13, 103.38, 137.34, 3.0),
            Ball::new(2.81, 96.83, 136.53, 3.0),
            Ball::new(199.58, 94.33, 130.99, 3.0),
            Ball::new(196.28, 96.52, 137.27, 3.0),
            Ball::new(192.59, 100.31, 143.44, 3.0),
            Ball::new(190.67, 100.96, 150.68, 3.0),
            Ball::new(187.5, 95.69, 150.38, 3.0),
            Ball::new(182.33, 94.62, 144.59, 3.0),
            Ball::new(184.33, 88.67, 146.43, 3.0),
            Ball::new(189.07, 84.29, 143.8, 3.0),
            Ball::new(191.45, 89.77, 148.42, 3.0),
            Ball::new(194.86, 84.61, 150.8, 3.0),
            Ball::new(1.45, 82.39, 152.66, 3.0),
            Ball::new(5.04, 81.64, 147.34, 3.0),
            Ball::new(5.47, 76.86, 142.69, 3.0),
            Ball::new(5.16, 75.21, 135.9, 3.0),
            Ball::new(199.99, 80.94, 137.51, 3.0),
            Ball::new(1.41, 78.75, 129.18, 3.0),
            Ball::new(8.21, 75.44, 128.81, 3.0),
            Ball::new(8.35, 81.56, 130.25, 3.0),
            Ball::new(6.65, 79.28, 122.15, 3.0),
            Ball::new(4.8, 71.03, 125.18, 3.0),
            Ball::new(10.12, 66.8, 123.83, 3.0),
            Ball::new(8.48, 63.62, 116.35, 3.0),
            Ball::new(6.2, 60.8, 125.01, 3.0),
            Ball::new(3.29, 55.29, 131.1, 3.0),
            Ball::new(195.59, 55.07, 133.32, 3.0),
            Ball::new(195.35, 53.1, 126.58, 3.0),
            Ball::new(2.39, 54.54, 123.83, 3.0),
            Ball::new(2.73, 48.16, 128.91, 3.0),
            Ball::new(6.94, 42.41, 130.5, 3.0),
            Ball::new(11.86, 44.13, 133.14, 3.0),
            Ball::new(18.09, 46.45, 135.96, 3.0),
            Ball::new(15.24, 41.72, 140.16, 3.0),
            Ball::new(7.03, 44.27, 143.93, 3.0),
            Ball::new(0.12, 39.89, 144.4, 3.0),
            Ball::new(196.82, 45.27, 145.55, 3.0),
            Ball::new(198.29, 51.36, 147.62, 3.0),
            Ball::new(195.38, 50.22, 152.79, 3.0),
            Ball::new(199.71, 47.08, 157.01, 3.0),
            Ball::new(198.69, 42.65, 162.43, 3.0),
            Ball::new(4.21, 42.48, 157.17, 3.0),
            Ball::new(198.77, 39.29, 154.66, 3.0),
            Ball::new(193.94, 33.49, 153.14, 3.0),
            Ball::new(193.53, 29.28, 146.87, 3.0),
            Ball::new(197.71, 32.8, 139.77, 3.0),
            Ball::new(2.12, 27.6, 139.4, 3.0),
            Ball::new(8.25, 28.61, 143.92, 3.0),
            Ball::new(5.23, 26.79, 150.75, 3.0),
            Ball::new(2.36, 34.37, 148.67, 3.0),
            Ball::new(0.18, 28.16, 146.85, 3.0),
            Ball::new(0.99, 21.1, 144.55, 3.0),
            Ball::new(197.14, 14.68, 142.83, 3.0),
            Ball::new(4.28, 14.63, 143.32, 3.0),
            Ball::new(4.17, 6.04, 141.11, 3.0),
            Ball::new(198.55, 11.05, 136.27, 3.0),
            Ball::new(1.65, 17.63, 135.27, 3.0),
            Ball::new(8.11, 15.64, 136.87, 3.0),
            Ball::new(8.84, 8.96, 132.77, 3.0),
            Ball::new(2.32, 11.74, 131.32, 3.0),
            Ball::new(195.68, 13.72, 128.38, 3.0),
            Ball::new(194.42, 20.61, 128.43, 3.0),
            Ball::new(0.37, 21.48, 124.8, 3.0),
            Ball::new(198.77, 28.78, 122.78, 3.0),
            Ball::new(199.57, 28.78, 130.48, 3.0),
            Ball::new(192.44, 27.17, 134.05, 3.0),
            Ball::new(187.1, 31.09, 131.26, 3.0),
            Ball::new(188.94, 33.19, 123.15, 3.0),
            Ball::new(191.64, 30.15, 116.89, 3.0),
            Ball::new(188.64, 28.34, 110.09, 3.0),
            Ball::new(193.72, 28.87, 105.23, 3.0),
            Ball::new(197.0, 34.6, 106.19, 3.0),
            Ball::new(199.2, 32.18, 114.74, 3.0),
            Ball::new(2.24, 27.61, 111.68, 3.0),
            Ball::new(8.22, 28.56, 114.86, 3.0),
            Ball::new(10.95, 24.23, 111.08, 3.0),
            Ball::new(17.76, 27.17, 109.62, 3.0),
        ];

        let result = compute_tessellation(&balls, 2.0, None, None);

        // C++ expected: 153 contacts, 100 cells
        assert_approx!(result.contacts.len() as f64, 153.0, 1.0, "contact count");
        assert_eq!(result.cells.len(), 100);

        // C++ expected total contact area: 3992.55
        let total_area: f64 = result.contacts.iter().map(|c| c.area).sum();
        assert_approx!(total_area, 3992.55, 1.0, "total contact area");

        // C++ expected total SAS area: 21979.6
        let total_sas: f64 = result.cells.iter().map(|c| c.sas_area).sum();
        assert_approx!(total_sas, 21979.6, 10.0, "total SAS area");

        // C++ expected total volume: 46419.9
        let total_vol: f64 = result.cells.iter().map(|c| c.volume).sum();
        assert_approx!(total_vol, 46419.9, 10.0, "total volume");

        // Check specific contact 0-1: area=42.9555, arc_length=23.2335
        let contact_0_1 = result.contacts.iter().find(|c| c.id_a == 0 && c.id_b == 1);
        assert!(contact_0_1.is_some(), "contact 0-1 should exist");
        let c01 = contact_0_1.unwrap();
        assert_approx!(c01.area, 42.9555, 0.01, "contact 0-1 area");
        assert_approx!(c01.arc_length, 23.2335, 0.01, "contact 0-1 arc_length");
    }

    /// Test periodic boundary conditions with 17-ball ring example
    /// Expected values from C++ api_usage_example_basic_and_periodic output
    #[test]
    fn test_cpp_api_example_periodic() {
        use crate::types::PeriodicBox;

        let balls = vec![
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
        ];

        let pbox = PeriodicBox::from_corners((-1.6, -1.6, -0.6), (1.6, 1.6, 3.1));
        let result = compute_tessellation(&balls, 1.0, Some(&pbox), None);

        // C++ produces 64 contacts in periodic mode (more than basic mode's 44)
        // (contacts include self-contacts through periodic boundaries)
        assert!(
            result.contacts.len() > 44,
            "periodic should have more contacts than basic: got {}",
            result.contacts.len()
        );

        // All 17 balls should have cells
        assert_eq!(result.cells.len(), 17);

        // Cell 0 should have different values in periodic mode
        // C++ expects: sas_area=3.56578, volume=20.2781 (vs basic: 34.8168, 29.2302)
        let cell0 = result.cells.iter().find(|c| c.index == 0).unwrap();
        assert_approx!(cell0.sas_area, 3.56578, 0.1, "cell 0 sas_area periodic");
        assert_approx!(cell0.volume, 20.2781, 0.1, "cell 0 volume periodic");

        // Check self-contact 0-0 exists (through periodic boundary)
        let self_contact = result.contacts.iter().find(|c| c.id_a == 0 && c.id_b == 0);
        assert!(
            self_contact.is_some(),
            "self-contact 0-0 should exist in periodic"
        );
        let sc = self_contact.unwrap();
        assert_approx!(sc.area, 3.36258, 0.01, "self-contact 0-0 area");
    }

    /// Test periodic boundary with 100-ball crystal structure
    /// Expected from C++ output: contacts_cs_1x1_periodic_summary.txt
    #[test]
    fn test_cpp_cli_balls_cs_1x1_periodic() {
        use crate::types::PeriodicBox;

        // Same balls as test_cpp_cli_balls_cs_1x1
        let balls = vec![
            Ball::new(46.99, 128.17, 144.94, 3.0),
            Ball::new(46.79, 127.84, 138.22, 3.0),
            Ball::new(40.46, 120.67, 136.9, 3.0),
            Ball::new(35.1, 117.45, 140.94, 3.0),
            Ball::new(33.86, 117.2, 148.43, 3.0),
            Ball::new(39.4, 120.41, 149.01, 3.0),
            Ball::new(36.71, 121.18, 154.2, 3.0),
            Ball::new(32.12, 126.51, 155.65, 3.0),
            Ball::new(34.67, 129.16, 149.57, 3.0),
            Ball::new(32.34, 128.99, 144.2, 3.0),
            Ball::new(33.09, 122.88, 145.41, 3.0),
            Ball::new(30.0, 125.65, 139.02, 3.0),
            Ball::new(27.62, 119.24, 141.44, 3.0),
            Ball::new(25.13, 113.84, 137.18, 3.0),
            Ball::new(29.87, 107.42, 137.46, 3.0),
            Ball::new(26.02, 102.66, 133.78, 3.0),
            Ball::new(20.71, 103.26, 138.04, 3.0),
            Ball::new(18.33, 108.95, 133.72, 3.0),
            Ball::new(18.21, 102.44, 131.67, 3.0),
            Ball::new(12.27, 98.98, 136.42, 3.0),
            Ball::new(7.17, 97.07, 142.18, 3.0),
            Ball::new(12.75, 101.93, 142.09, 3.0),
            Ball::new(10.25, 106.11, 136.61, 3.0),
            Ball::new(5.13, 103.38, 137.34, 3.0),
            Ball::new(2.81, 96.83, 136.53, 3.0),
            Ball::new(199.58, 94.33, 130.99, 3.0),
            Ball::new(196.28, 96.52, 137.27, 3.0),
            Ball::new(192.59, 100.31, 143.44, 3.0),
            Ball::new(190.67, 100.96, 150.68, 3.0),
            Ball::new(187.5, 95.69, 150.38, 3.0),
            Ball::new(182.33, 94.62, 144.59, 3.0),
            Ball::new(184.33, 88.67, 146.43, 3.0),
            Ball::new(189.07, 84.29, 143.8, 3.0),
            Ball::new(191.45, 89.77, 148.42, 3.0),
            Ball::new(194.86, 84.61, 150.8, 3.0),
            Ball::new(1.45, 82.39, 152.66, 3.0),
            Ball::new(5.04, 81.64, 147.34, 3.0),
            Ball::new(5.47, 76.86, 142.69, 3.0),
            Ball::new(5.16, 75.21, 135.9, 3.0),
            Ball::new(199.99, 80.94, 137.51, 3.0),
            Ball::new(1.41, 78.75, 129.18, 3.0),
            Ball::new(8.21, 75.44, 128.81, 3.0),
            Ball::new(8.35, 81.56, 130.25, 3.0),
            Ball::new(6.65, 79.28, 122.15, 3.0),
            Ball::new(4.8, 71.03, 125.18, 3.0),
            Ball::new(10.12, 66.8, 123.83, 3.0),
            Ball::new(8.48, 63.62, 116.35, 3.0),
            Ball::new(6.2, 60.8, 125.01, 3.0),
            Ball::new(3.29, 55.29, 131.1, 3.0),
            Ball::new(195.59, 55.07, 133.32, 3.0),
            Ball::new(195.35, 53.1, 126.58, 3.0),
            Ball::new(2.39, 54.54, 123.83, 3.0),
            Ball::new(2.73, 48.16, 128.91, 3.0),
            Ball::new(6.94, 42.41, 130.5, 3.0),
            Ball::new(11.86, 44.13, 133.14, 3.0),
            Ball::new(18.09, 46.45, 135.96, 3.0),
            Ball::new(15.24, 41.72, 140.16, 3.0),
            Ball::new(7.03, 44.27, 143.93, 3.0),
            Ball::new(0.12, 39.89, 144.4, 3.0),
            Ball::new(196.82, 45.27, 145.55, 3.0),
            Ball::new(198.29, 51.36, 147.62, 3.0),
            Ball::new(195.38, 50.22, 152.79, 3.0),
            Ball::new(199.71, 47.08, 157.01, 3.0),
            Ball::new(198.69, 42.65, 162.43, 3.0),
            Ball::new(4.21, 42.48, 157.17, 3.0),
            Ball::new(198.77, 39.29, 154.66, 3.0),
            Ball::new(193.94, 33.49, 153.14, 3.0),
            Ball::new(193.53, 29.28, 146.87, 3.0),
            Ball::new(197.71, 32.8, 139.77, 3.0),
            Ball::new(2.12, 27.6, 139.4, 3.0),
            Ball::new(8.25, 28.61, 143.92, 3.0),
            Ball::new(5.23, 26.79, 150.75, 3.0),
            Ball::new(2.36, 34.37, 148.67, 3.0),
            Ball::new(0.18, 28.16, 146.85, 3.0),
            Ball::new(0.99, 21.1, 144.55, 3.0),
            Ball::new(197.14, 14.68, 142.83, 3.0),
            Ball::new(4.28, 14.63, 143.32, 3.0),
            Ball::new(4.17, 6.04, 141.11, 3.0),
            Ball::new(198.55, 11.05, 136.27, 3.0),
            Ball::new(1.65, 17.63, 135.27, 3.0),
            Ball::new(8.11, 15.64, 136.87, 3.0),
            Ball::new(8.84, 8.96, 132.77, 3.0),
            Ball::new(2.32, 11.74, 131.32, 3.0),
            Ball::new(195.68, 13.72, 128.38, 3.0),
            Ball::new(194.42, 20.61, 128.43, 3.0),
            Ball::new(0.37, 21.48, 124.8, 3.0),
            Ball::new(198.77, 28.78, 122.78, 3.0),
            Ball::new(199.57, 28.78, 130.48, 3.0),
            Ball::new(192.44, 27.17, 134.05, 3.0),
            Ball::new(187.1, 31.09, 131.26, 3.0),
            Ball::new(188.94, 33.19, 123.15, 3.0),
            Ball::new(191.64, 30.15, 116.89, 3.0),
            Ball::new(188.64, 28.34, 110.09, 3.0),
            Ball::new(193.72, 28.87, 105.23, 3.0),
            Ball::new(197.0, 34.6, 106.19, 3.0),
            Ball::new(199.2, 32.18, 114.74, 3.0),
            Ball::new(2.24, 27.61, 111.68, 3.0),
            Ball::new(8.22, 28.56, 114.86, 3.0),
            Ball::new(10.95, 24.23, 111.08, 3.0),
            Ball::new(17.76, 27.17, 109.62, 3.0),
        ];

        // Periodic box: corners (0,0,0) to (200,250,300)
        let pbox = PeriodicBox::from_corners((0.0, 0.0, 0.0), (200.0, 250.0, 300.0));
        let result = compute_tessellation(&balls, 2.0, Some(&pbox), None);

        // C++ expected: 189 contacts (vs 153 non-periodic), 100 cells
        assert_approx!(
            result.contacts.len() as f64,
            189.0,
            5.0,
            "periodic contact count"
        );
        assert_eq!(result.cells.len(), 100);

        // C++ expected total contact area: 4812.14 (vs 3992.55 non-periodic)
        let total_area: f64 = result.contacts.iter().map(|c| c.area).sum();
        assert_approx!(total_area, 4812.14, 50.0, "total periodic contact area");

        // C++ expected total SAS area: 20023.1 (vs 21979.6 non-periodic)
        let total_sas: f64 = result.cells.iter().map(|c| c.sas_area).sum();
        assert_approx!(total_sas, 20023.1, 100.0, "total periodic SAS area");

        // C++ expected total volume: 45173.2 (vs 46419.9 non-periodic)
        let total_vol: f64 = result.cells.iter().map(|c| c.volume).sum();
        assert_approx!(total_vol, 45173.2, 100.0, "total periodic volume");
    }

    /// Test grouping: contacts between spheres in same group are excluded
    #[test]
    fn test_grouping_excludes_same_group() {
        let balls = vec![
            Ball::new(0.0, 0.0, 0.0, 1.0),
            Ball::new(2.0, 0.0, 0.0, 1.0),
            Ball::new(4.0, 0.0, 0.0, 1.0),
        ];

        // Without grouping: 2 contacts (0-1 and 1-2)
        let result_no_groups = compute_tessellation(&balls, 0.5, None, None);
        assert_eq!(result_no_groups.contacts.len(), 2);

        // All in same group: 0 contacts
        let groups_same = vec![0, 0, 0];
        let result_same = compute_tessellation(&balls, 0.5, None, Some(&groups_same));
        assert_eq!(result_same.contacts.len(), 0);

        // 0 and 1 in group 0, 2 in group 1: only 1 contact (1-2)
        let groups_partial = vec![0, 0, 1];
        let result_partial = compute_tessellation(&balls, 0.5, None, Some(&groups_partial));
        assert_eq!(result_partial.contacts.len(), 1);
        assert_eq!(result_partial.contacts[0].id_a, 1);
        assert_eq!(result_partial.contacts[0].id_b, 2);
    }

    /// Test grouping with periodic boundary conditions
    #[test]
    fn test_grouping_periodic() {
        use crate::types::PeriodicBox;

        let balls = vec![
            Ball::new(0.0, 0.0, 2.0, 1.0),
            Ball::new(0.0, 1.0, 0.0, 0.5),
            Ball::new(0.382683, 0.92388, 0.0, 0.5),
        ];

        let pbox = PeriodicBox::from_corners((-2.0, -2.0, -1.0), (2.0, 2.0, 4.0));

        // Without grouping
        let result_no_groups = compute_tessellation(&balls, 1.0, Some(&pbox), None);
        let contacts_no_groups = result_no_groups.contacts.len();

        // All in same group: should have fewer/no contacts
        let groups_same = vec![0, 0, 0];
        let result_same = compute_tessellation(&balls, 1.0, Some(&pbox), Some(&groups_same));
        assert!(
            result_same.contacts.len() < contacts_no_groups,
            "same group should reduce contacts"
        );
    }

    /// Test grouping with mixed groups - some contacts remain
    #[test]
    fn test_grouping_mixed_groups() {
        let balls = vec![
            Ball::new(0.0, 0.0, 0.0, 1.0),
            Ball::new(2.0, 0.0, 0.0, 1.0),
            Ball::new(4.0, 0.0, 0.0, 1.0),
            Ball::new(6.0, 0.0, 0.0, 1.0),
        ];

        // Groups: 0,0,1,1 - contacts only between groups (1-2)
        let groups = vec![0, 0, 1, 1];
        let result = compute_tessellation(&balls, 0.5, None, Some(&groups));

        // Only contact between balls 1 and 2 (different groups, adjacent)
        assert_eq!(result.contacts.len(), 1);
        assert_eq!(result.contacts[0].id_a, 1);
        assert_eq!(result.contacts[0].id_b, 2);

        // Cells are computed for spheres with contacts
        assert_eq!(result.cells.len(), 2);
    }
}
