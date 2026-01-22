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

/// Error type for solvent sphere computation.
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::enum_variant_names)] // All variants represent invalid input conditions
pub enum SolventSpheresError {
    /// Probe radius must be non-negative and finite.
    InvalidProbe(f64),
    /// Volume probe must be non-negative and finite.
    InvalidVolumeProbe(f64),
    /// A ball has invalid coordinates or radius.
    InvalidBall {
        /// Index of the invalid ball.
        index: usize,
        /// Description of why the ball is invalid.
        reason: &'static str,
    },
}

impl std::fmt::Display for SolventSpheresError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidProbe(v) => write!(
                f,
                "invalid probe radius: {v} (must be non-negative and finite)"
            ),
            Self::InvalidVolumeProbe(v) => write!(
                f,
                "invalid volume probe: {v} (must be non-negative and finite)"
            ),
            Self::InvalidBall { index, reason } => {
                write!(f, "invalid ball at index {index}: {reason}")
            }
        }
    }
}

impl std::error::Error for SolventSpheresError {}

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
/// * `probe` - Rolling probe radius (e.g., 1.4 Å for water). Must be non-negative.
/// * `volume_probe` - Optional probe radius for stage 2 volume calculation. Defaults to 0.0.
/// * `subdivision_depth` - Optional icosahedron subdivision depth for surface sampling.
///   Defaults to `Depth2` (162 sample points per atom).
///
/// # Returns
///
/// Vector of [`SolventSphere`] with positions, radii, weights, and parent atom indices,
/// or an error if parameters are invalid.
///
/// # Errors
///
/// Returns [`SolventSpheresError`] if:
/// - `probe` is negative, NaN, or infinite
/// - `volume_probe` is negative, NaN, or infinite
/// - Any ball has NaN/infinite coordinates or non-positive radius
///
/// # Example
///
/// ```
/// use voronota_ltr::{Ball, compute_solvent_spheres};
///
/// let atoms = vec![
///     Ball::new(0.0, 0.0, 0.0, 1.5),
///     Ball::new(3.0, 0.0, 0.0, 1.5),
/// ];
///
/// let solvent = compute_solvent_spheres(&atoms, 1.4, None, None).unwrap();
///
/// for s in &solvent {
///     println!("Solvent at ({:.2}, {:.2}, {:.2}), weight={:.3}, parent={}",
///              s.x, s.y, s.z, s.weight, s.parent_index);
/// }
/// ```
pub fn compute_solvent_spheres(
    balls: &[Ball],
    probe: f64,
    volume_probe: Option<f64>,
    subdivision_depth: Option<SubdivisionDepth>,
) -> Result<Vec<SolventSphere>, SolventSpheresError> {
    let volume_probe = volume_probe.unwrap_or(0.0);
    let subdivision_depth = subdivision_depth.unwrap_or(SubdivisionDepth::Depth2);

    validate_params(probe, volume_probe)?;
    validate_balls(balls)?;

    if balls.is_empty() {
        return Ok(Vec::new());
    }

    // Stage 1: tessellate to find contacts and exposed surfaces
    let stage1_result = compute_tessellation(balls, probe, None, None);
    let neighbors = build_neighbor_graph(balls, &stage1_result);

    // Generate pseudo-solvent spheres by sampling exposed surfaces
    let solvent_spheres =
        generate_solvent_spheres(balls, &stage1_result, &neighbors, probe, subdivision_depth);

    if solvent_spheres.is_empty() {
        return Ok(Vec::new());
    }

    // Stage 2: compute volumes
    Ok(compute_solvent_weights(
        balls,
        solvent_spheres,
        probe,
        volume_probe,
    ))
}

/// Validate probe parameters.
fn validate_params(probe: f64, volume_probe: f64) -> Result<(), SolventSpheresError> {
    if !probe.is_finite() || probe < 0.0 {
        return Err(SolventSpheresError::InvalidProbe(probe));
    }
    if !volume_probe.is_finite() || volume_probe < 0.0 {
        return Err(SolventSpheresError::InvalidVolumeProbe(volume_probe));
    }
    Ok(())
}

/// Validate input balls.
fn validate_balls(balls: &[Ball]) -> Result<(), SolventSpheresError> {
    for (i, ball) in balls.iter().enumerate() {
        if !ball.x.is_finite() || !ball.y.is_finite() || !ball.z.is_finite() {
            return Err(SolventSpheresError::InvalidBall {
                index: i,
                reason: "coordinates must be finite",
            });
        }
        if !ball.r.is_finite() || ball.r <= 0.0 {
            return Err(SolventSpheresError::InvalidBall {
                index: i,
                reason: "radius must be positive and finite",
            });
        }
    }
    Ok(())
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

        // Indices from tessellation are guaranteed valid for the input balls
        let dist = nalgebra::distance(&ball_center(&balls[a]), &ball_center(&balls[b]));

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
    probe: f64,
    subdivision_depth: SubdivisionDepth,
) -> Vec<SolventSphere> {
    let sih = SubdividedIcosahedron::new(subdivision_depth);

    result
        .cells
        .iter()
        .filter(|cell| cell.sas_area > 0.0)
        .flat_map(|cell| {
            let atom = &balls[cell.index];
            let atom_neighbors = &neighbors[cell.index];
            let center = ball_center(atom);
            let surface_radius = atom.r + probe;
            let parent_index = cell.index;

            sih.points_on_sphere(center, surface_radius)
                .filter(move |p| is_valid_solvent_position(p, atom, atom_neighbors, balls))
                .map(move |p| SolventSphere {
                    x: p.x,
                    y: p.y,
                    z: p.z,
                    radius: probe,
                    weight: 0.0,
                    parent_index,
                })
        })
        .collect()
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
    probe: f64,
    volume_probe: f64,
) -> Vec<SolventSphere> {
    let num_original = balls.len();
    let mut combined = Vec::with_capacity(num_original + solvent_spheres.len());

    // Original atoms with adjusted radii
    for ball in balls {
        combined.push(Ball::new(
            ball.x,
            ball.y,
            ball.z,
            (ball.r - probe + volume_probe).max(0.01),
        ));
    }

    // Solvent spheres
    for sphere in &solvent_spheres {
        combined.push(Ball::new(
            sphere.x,
            sphere.y,
            sphere.z,
            probe + volume_probe,
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

        let solvent =
            compute_solvent_spheres(&balls, 1.4, None, Some(SubdivisionDepth::Depth0)).unwrap();

        // Single isolated sphere should have all sample points valid (12 for Depth0)
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

        let solvent =
            compute_solvent_spheres(&balls, 1.4, None, Some(SubdivisionDepth::Depth0)).unwrap();

        // Contact area means some sample points fail the distance test
        assert!(
            solvent.len() < 24,
            "touching spheres should have fewer solvent spheres"
        );
        assert!(!solvent.is_empty());
    }

    #[test]
    fn empty_input() {
        let solvent = compute_solvent_spheres(&[], 1.4, None, None).unwrap();
        assert!(solvent.is_empty());
    }

    #[test]
    fn cpp_sihsolv_two_separated_spheres() {
        // Test against C++ sihsolv output (commit 33f2429e)
        // Input: two spheres at (0,0,0) and (5,0,0), radius 1.5, probe 1.4, depth 0
        let balls = vec![Ball::new(0.0, 0.0, 0.0, 1.5), Ball::new(5.0, 0.0, 0.0, 1.5)];

        let solvent =
            compute_solvent_spheres(&balls, 1.4, Some(0.0), Some(SubdivisionDepth::Depth0))
                .unwrap();

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

        let solvent =
            compute_solvent_spheres(&balls, 1.4, None, Some(SubdivisionDepth::Depth0)).unwrap();

        // Both spheres contribute, but outer sphere should dominate
        let outer_count = solvent.iter().filter(|s| s.parent_index == 0).count();
        let inner_count = solvent.iter().filter(|s| s.parent_index == 1).count();
        assert!(
            outer_count > inner_count,
            "outer sphere should contribute more solvent: outer={outer_count}, inner={inner_count}"
        );
    }

    #[test]
    fn invalid_probe_returns_error() {
        let balls = vec![Ball::new(0.0, 0.0, 0.0, 1.5)];
        assert!(matches!(
            compute_solvent_spheres(&balls, -1.0, None, None),
            Err(SolventSpheresError::InvalidProbe(_))
        ));
    }

    #[test]
    fn nan_probe_returns_error() {
        let balls = vec![Ball::new(0.0, 0.0, 0.0, 1.5)];
        assert!(matches!(
            compute_solvent_spheres(&balls, f64::NAN, None, None),
            Err(SolventSpheresError::InvalidProbe(_))
        ));
    }

    #[test]
    fn invalid_ball_returns_error() {
        let balls = vec![Ball::new(0.0, 0.0, 0.0, -1.0)]; // Negative radius
        assert!(matches!(
            compute_solvent_spheres(&balls, 1.4, None, None),
            Err(SolventSpheresError::InvalidBall { .. })
        ));
    }

    #[test]
    fn nan_coordinates_returns_error() {
        let balls = vec![Ball::new(f64::NAN, 0.0, 0.0, 1.5)];
        assert!(matches!(
            compute_solvent_spheres(&balls, 1.4, None, None),
            Err(SolventSpheresError::InvalidBall { .. })
        ));
    }
}
