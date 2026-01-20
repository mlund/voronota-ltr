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
}
