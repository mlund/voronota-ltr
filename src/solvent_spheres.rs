//! Generate weighted pseudo-solvent spheres around a molecule.

use std::collections::HashSet;

use nalgebra::Point3;

use crate::subdivided_icosahedron::{SubdividedIcosahedron, SubdivisionDepth};
use crate::tessellation::compute_tessellation;
use crate::types::{Ball, TessellationResult};

/// A pseudo-solvent sphere representing where a solvent molecule could sit on the molecular surface.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SolventSphere {
    /// X coordinate of the sphere center.
    pub x: f64,
    /// Y coordinate of the sphere center.
    pub y: f64,
    /// Z coordinate of the sphere center.
    pub z: f64,
    /// Radius of the solvent sphere (equals probe radius).
    pub radius: f64,
    /// Weight derived from the Voronoi cell volume in stage 2 tessellation.
    /// Larger weights indicate more accessible space.
    pub weight: f64,
    /// Index of the parent atom this solvent sphere belongs to.
    pub parent_index: usize,
}

/// Parameters for generating pseudo-solvent spheres.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SolventSpheresParams {
    /// Rolling probe radius used to define the solvent-accessible surface.
    /// Default: 1.4 Å (water molecule radius).
    pub probe: f64,

    /// Probe radius for stage 2 volume calculation.
    /// A smaller value gives tighter volume estimates around solvent spheres.
    /// Default: 0.0.
    pub volume_probe: f64,

    /// Subdivision depth of the icosahedron used for surface sampling.
    /// Higher values produce more solvent spheres but take longer.
    pub subdivision_depth: SubdivisionDepth,
}

impl Default for SolventSpheresParams {
    fn default() -> Self {
        Self {
            probe: 1.4,
            volume_probe: 0.0,
            subdivision_depth: SubdivisionDepth::default(),
        }
    }
}

/// Generate weighted pseudo-solvent spheres around a molecule.
///
/// This implements a two-stage algorithm that places solvent-sized spheres on the
/// exposed molecular surface and computes weights based on available space.
///
/// # Algorithm
///
/// **Stage 1: Find surface points belonging to each atom**
///
/// 1. Compute Voronoi tessellation of input atoms
/// 2. Build a neighbor graph from contacts (which atoms touch which)
/// 3. For each atom with exposed surface area (`sas_area` > 0):
///    - Sample points on a subdivided icosahedron at the atom's probe-extended surface
///    - For each sample point, check if it's closer to the current atom than to all neighbors
///    - Points passing this test are in the atom's Voronoi region, representing "true" exposed surface
///    - Place a solvent sphere at each valid point
///
/// **Stage 2: Compute weights via volume calculation**
///
/// 1. Combine original atoms (with adjusted radii) and pseudo-solvent spheres
/// 2. Re-tessellate to compute Voronoi cells for everything
/// 3. Each solvent sphere's cell volume becomes its weight
///
/// # Arguments
///
/// * `balls` - Input atomic spheres (atoms with positions and radii)
/// * `params` - Parameters controlling probe radius and sampling density
///
/// # Returns
///
/// Vector of [`SolventSphere`] with positions, radii, weights, and parent atom indices.
///
/// # Example
///
/// ```
/// use voronota_ltr::{Ball, SolventSpheresParams, compute_solvent_spheres};
///
/// let atoms = vec![
///     Ball::new(0.0, 0.0, 0.0, 1.5),
///     Ball::new(3.0, 0.0, 0.0, 1.5),
/// ];
///
/// let params = SolventSpheresParams::default();
/// let solvent = compute_solvent_spheres(&atoms, &params);
///
/// for s in &solvent {
///     println!("Solvent at ({:.2}, {:.2}, {:.2}), weight={:.3}, parent={}",
///              s.x, s.y, s.z, s.weight, s.parent_index);
/// }
/// ```
#[must_use]
pub fn compute_solvent_spheres(
    balls: &[Ball],
    params: &SolventSpheresParams,
) -> Vec<SolventSphere> {
    if balls.is_empty() {
        return Vec::new();
    }

    // Stage 1: tessellate to find contacts and exposed surfaces
    let stage1_result = compute_tessellation(balls, params.probe, None, None);
    let neighbors = build_neighbor_graph(balls, &stage1_result);

    // Generate pseudo-solvent spheres by sampling exposed surfaces
    let solvent_spheres = generate_solvent_spheres(balls, &stage1_result, &neighbors, params);

    if solvent_spheres.is_empty() {
        return Vec::new();
    }

    // Stage 2: compute volumes
    compute_solvent_weights(balls, solvent_spheres, params)
}

/// Build neighbor graph from tessellation contacts.
fn build_neighbor_graph(balls: &[Ball], result: &TessellationResult) -> Vec<Vec<(f64, usize)>> {
    let mut neighbors: Vec<Vec<(f64, usize)>> = vec![Vec::new(); balls.len()];

    // Pre-compute which cells have exposed surface for O(1) lookup
    let exposed: HashSet<usize> = result
        .cells
        .iter()
        .filter(|c| c.sas_area > 0.0)
        .map(|c| c.index)
        .collect();

    for contact in &result.contacts {
        let (a, b) = (contact.id_a, contact.id_b);
        let dist = nalgebra::distance(&ball_center(&balls[a]), &ball_center(&balls[b]));

        // Only add neighbors for atoms with exposed surface
        if exposed.contains(&a) {
            neighbors[a].push((dist, b));
        }
        if exposed.contains(&b) {
            neighbors[b].push((dist, a));
        }
    }

    // Sort by distance for early termination during validity checks
    for n in &mut neighbors {
        n.sort_by(|a, b| a.0.total_cmp(&b.0));
    }

    neighbors
}

/// Generate solvent spheres by sampling exposed atom surfaces.
fn generate_solvent_spheres(
    balls: &[Ball],
    result: &TessellationResult,
    neighbors: &[Vec<(f64, usize)>],
    params: &SolventSpheresParams,
) -> Vec<SolventSphere> {
    let sih = SubdividedIcosahedron::new(params.subdivision_depth);
    let mut solvent_spheres = Vec::new();

    for cell in &result.cells {
        if cell.sas_area <= 0.0 {
            continue;
        }

        let atom = &balls[cell.index];
        let center = ball_center(atom);
        let surface_radius = atom.r + params.probe;

        for sample_point in sih.points_on_sphere(center, surface_radius) {
            if is_valid_solvent_position(&sample_point, atom, &neighbors[cell.index], balls) {
                solvent_spheres.push(SolventSphere {
                    x: sample_point.x,
                    y: sample_point.y,
                    z: sample_point.z,
                    radius: params.probe,
                    weight: 0.0,
                    parent_index: cell.index,
                });
            }
        }
    }

    solvent_spheres
}

/// Check if sample point is closer to its parent atom than all neighbors.
fn is_valid_solvent_position(
    point: &Point3<f64>,
    atom: &Ball,
    neighbors: &[(f64, usize)],
    balls: &[Ball],
) -> bool {
    let center = ball_center(atom);
    let dist_to_atom = nalgebra::distance(point, &center) - atom.r;

    for &(_, neighbor_id) in neighbors {
        let neighbor = &balls[neighbor_id];
        let dist_to_neighbor = nalgebra::distance(point, &ball_center(neighbor)) - neighbor.r;

        if dist_to_neighbor <= dist_to_atom {
            return false;
        }
    }

    true
}

/// Compute weights via stage 2 tessellation.
fn compute_solvent_weights(
    balls: &[Ball],
    solvent_spheres: Vec<SolventSphere>,
    params: &SolventSpheresParams,
) -> Vec<SolventSphere> {
    let num_original = balls.len();
    let mut combined = Vec::with_capacity(num_original + solvent_spheres.len());

    // Original atoms with adjusted radii
    for ball in balls {
        combined.push(Ball::new(
            ball.x,
            ball.y,
            ball.z,
            (ball.r - params.probe + params.volume_probe).max(0.01),
        ));
    }

    // Solvent spheres
    for sphere in &solvent_spheres {
        combined.push(Ball::new(
            sphere.x,
            sphere.y,
            sphere.z,
            params.probe + params.volume_probe,
        ));
    }

    let stage2_result = compute_tessellation(&combined, 0.0, None, None);

    // Map cell index to volume for O(1) lookup
    let cell_volumes: std::collections::HashMap<usize, f64> = stage2_result
        .cells
        .iter()
        .map(|c| (c.index, c.volume))
        .collect();

    solvent_spheres
        .into_iter()
        .enumerate()
        .map(|(i, mut sphere)| {
            if let Some(&volume) = cell_volumes.get(&(num_original + i)) {
                sphere.weight = volume;
            }
            sphere
        })
        .collect()
}

/// Extract center point from a Ball.
const fn ball_center(ball: &Ball) -> Point3<f64> {
    Point3::new(ball.x, ball.y, ball.z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_sphere_produces_solvent() {
        let balls = vec![Ball::new(0.0, 0.0, 0.0, 1.5)];
        let params = SolventSpheresParams {
            subdivision_depth: SubdivisionDepth::Depth0, // 12 points
            ..Default::default()
        };

        let solvent = compute_solvent_spheres(&balls, &params);

        // Single isolated sphere should have all sample points valid
        assert_eq!(solvent.len(), 12);
        for s in &solvent {
            assert_eq!(s.parent_index, 0);
            assert!((s.radius - 1.4).abs() < 1e-10);
            assert!(s.weight > 0.0, "solvent sphere should have positive weight");
        }
    }

    #[test]
    fn two_touching_spheres() {
        // Two spheres touching - should have fewer solvent spheres than 2*12
        let balls = vec![
            Ball::new(0.0, 0.0, 0.0, 1.5),
            Ball::new(2.5, 0.0, 0.0, 1.5), // Close enough to contact
        ];
        let params = SolventSpheresParams {
            subdivision_depth: SubdivisionDepth::Depth0,
            ..Default::default()
        };

        let solvent = compute_solvent_spheres(&balls, &params);

        // Contact area means some sample points fail the distance test
        assert!(
            solvent.len() < 24,
            "touching spheres should have fewer solvent spheres"
        );
        assert!(!solvent.is_empty());
    }

    #[test]
    fn empty_input() {
        let solvent = compute_solvent_spheres(&[], &SolventSpheresParams::default());
        assert!(solvent.is_empty());
    }

    #[test]
    fn cpp_sihsolv_two_separated_spheres() {
        // Test against C++ sihsolv output (commit 33f2429e)
        // Input: two spheres at (0,0,0) and (5,0,0), radius 1.5, probe 1.4, depth 0
        let balls = vec![Ball::new(0.0, 0.0, 0.0, 1.5), Ball::new(5.0, 0.0, 0.0, 1.5)];
        let params = SolventSpheresParams {
            probe: 1.4,
            volume_probe: 0.0,
            subdivision_depth: SubdivisionDepth::Depth0,
        };

        let solvent = compute_solvent_spheres(&balls, &params);

        // C++ produces 24 solvent spheres (12 per ball)
        assert_eq!(solvent.len(), 24, "expected 24 solvent spheres");

        // Verify parent distribution
        let ball0_count = solvent.iter().filter(|s| s.parent_index == 0).count();
        let ball1_count = solvent.iter().filter(|s| s.parent_index == 1).count();
        assert_eq!(ball0_count, 12);
        assert_eq!(ball1_count, 12);

        // C++ total weight: sum of all weights ≈ 249.1 (from stage 2 volumes)
        let total_weight: f64 = solvent.iter().map(|s| s.weight).sum();
        assert!(
            (total_weight - 249.1).abs() < 5.0,
            "total weight {total_weight} should be ~249.1"
        );

        // All radii should equal probe
        for s in &solvent {
            assert!((s.radius - 1.4).abs() < 1e-10);
        }
    }

    #[test]
    fn contained_sphere_minimal_solvent() {
        // Small sphere mostly inside large sphere - should have minimal exposure
        let balls = vec![
            Ball::new(0.0, 0.0, 0.0, 3.0), // Large outer sphere
            Ball::new(2.5, 0.0, 0.0, 1.0), // Partially exposed at edge
        ];
        let params = SolventSpheresParams {
            subdivision_depth: SubdivisionDepth::Depth0,
            ..Default::default()
        };

        let solvent = compute_solvent_spheres(&balls, &params);

        // Both spheres contribute, but outer sphere should dominate
        let outer_count = solvent.iter().filter(|s| s.parent_index == 0).count();
        let inner_count = solvent.iter().filter(|s| s.parent_index == 1).count();
        assert!(
            outer_count > inner_count,
            "outer sphere should contribute more solvent: outer={outer_count}, inner={inner_count}"
        );
    }
}
