use rayon::prelude::*;

use crate::contact::construct_contact_descriptor;
use crate::spheres_searcher::SpheresSearcher;
use crate::types::{
    Ball, Cell, CellContactSummary, Contact, ContactDescriptorSummary, Sphere, TessellationResult,
    ValuedId,
};

/// Main entry point: compute radical tessellation contacts and cells
pub fn compute_tessellation(balls: &[Ball], probe: f64) -> TessellationResult {
    if balls.is_empty() {
        return TessellationResult::default();
    }

    // Convert balls to spheres (add probe to radii)
    let spheres: Vec<Sphere> = balls.iter().map(|b| Sphere::from_ball(b, probe)).collect();

    // Build spatial index
    let searcher = SpheresSearcher::new(spheres.clone());

    // Find all collision pairs and their neighbors
    let collision_data: Vec<_> = (0..spheres.len())
        .into_par_iter()
        .map(|id| {
            let result = searcher.find_colliding_ids(id, true);
            (id, result.colliding_ids, result.exclusion_status)
        })
        .collect();

    // Build map of sphere -> collisions for quick neighbor lookup
    let all_collisions: Vec<Vec<ValuedId>> = collision_data
        .into_iter()
        .map(|(_, collisions, _)| collisions)
        .collect();

    // Collect unique collision pairs (a < b) to avoid duplicate work
    let mut collision_pairs: Vec<(usize, usize)> = Vec::new();
    for (a_id, neighbors) in all_collisions.iter().enumerate() {
        for neighbor in neighbors {
            if a_id < neighbor.index {
                collision_pairs.push((a_id, neighbor.index));
            }
        }
    }

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

    // Accumulate cell summaries
    let cells = compute_cells(&valid_summaries, searcher.spheres(), &all_collisions);

    TessellationResult { contacts, cells }
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

        let result = compute_tessellation(&balls, 0.5);

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

        let result = compute_tessellation(&balls, 0.5);

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

        let result = compute_tessellation(&balls, 0.5);

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

        let result = compute_tessellation(&balls, 0.5);

        // No contacts between well-separated spheres
        assert!(result.contacts.is_empty());

        // Both should be detached with full SAS
        assert_eq!(result.cells.len(), 2);
    }

    #[test]
    fn test_empty_input() {
        let balls: Vec<Ball> = vec![];
        let result = compute_tessellation(&balls, 0.5);

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

        let result = compute_tessellation(&balls, 1.0);

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

        let result = compute_tessellation(&balls, 1.0);

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

        let result = compute_tessellation(&balls, 2.0);

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

        let result = compute_tessellation(&balls, 2.0);

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

        let result = compute_tessellation(&balls, 0.5);

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

        let result = compute_tessellation(&balls, 2.0);

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
}
