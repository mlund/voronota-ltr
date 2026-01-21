//! Subdivided icosahedron for uniform sphere surface sampling.

use nalgebra::{Point3, Vector3};

/// Subdivision depth controlling the number of sample points.
///
/// Higher levels produce more uniform sampling but increase computation time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SubdivisionDepth {
    /// 12 sample points (base icosahedron vertices).
    Depth0 = 0,
    /// 42 sample points (default).
    #[default]
    Depth1 = 1,
    /// 162 sample points.
    Depth2 = 2,
    /// 642 sample points.
    Depth3 = 3,
    /// 2562 sample points.
    Depth4 = 4,
}

impl From<u32> for SubdivisionDepth {
    /// Convert from integer, clamping to valid range [0, 4].
    fn from(value: u32) -> Self {
        match value {
            0 => Self::Depth0,
            1 => Self::Depth1,
            2 => Self::Depth2,
            3 => Self::Depth3,
            _ => Self::Depth4,
        }
    }
}

/// Generates uniformly distributed points on a unit sphere via icosahedron subdivision.
///
/// Each subdivision level splits every triangle into 4 smaller triangles,
/// projecting new vertices onto the unit sphere.
pub struct SubdividedIcosahedron {
    vertices: Vec<Vector3<f64>>,
}

impl SubdividedIcosahedron {
    /// Create a subdivided icosahedron with the given depth.
    #[must_use]
    #[allow(clippy::manual_midpoint)] // This is golden ratio φ = (1+√5)/2, not midpoint
    pub fn new(depth: SubdivisionDepth) -> Self {
        let t = (1.0 + 5.0_f64.sqrt()) / 2.0;

        // 12 vertices of a regular icosahedron, normalized to unit sphere
        let mut vertices: Vec<Vector3<f64>> = [
            (t, 1.0, 0.0),
            (-t, 1.0, 0.0),
            (t, -1.0, 0.0),
            (-t, -1.0, 0.0),
            (1.0, 0.0, t),
            (1.0, 0.0, -t),
            (-1.0, 0.0, t),
            (-1.0, 0.0, -t),
            (0.0, t, 1.0),
            (0.0, -t, 1.0),
            (0.0, t, -1.0),
            (0.0, -t, -1.0),
        ]
        .into_iter()
        .map(|(x, y, z)| Vector3::new(x, y, z).normalize())
        .collect();

        // 20 triangular faces of the icosahedron
        let mut triples: Vec<[usize; 3]> = vec![
            [0, 8, 4],
            [1, 10, 7],
            [2, 9, 11],
            [7, 3, 1],
            [0, 5, 10],
            [3, 9, 6],
            [3, 11, 9],
            [8, 6, 4],
            [2, 4, 9],
            [3, 7, 11],
            [4, 2, 0],
            [9, 4, 6],
            [2, 11, 5],
            [0, 10, 8],
            [5, 0, 2],
            [10, 5, 7],
            [1, 6, 8],
            [1, 8, 10],
            [6, 1, 3],
            [11, 7, 5],
        ];

        // Subdivide: each triangle splits into 4 smaller triangles
        for _ in 0..depth as u8 {
            let mut next_triples = Vec::with_capacity(triples.len() * 4);
            let mut edge_midpoints = std::collections::HashMap::new();

            for [v0, v1, v2] in &triples {
                // Get or create midpoint for each of the 3 edges
                let edges = [(*v1, *v2), (*v0, *v2), (*v0, *v1)];
                let mid: Vec<usize> = edges
                    .iter()
                    .map(|&(a, b)| {
                        let key = if a < b { (a, b) } else { (b, a) };
                        *edge_midpoints.entry(key).or_insert_with(|| {
                            let midpoint = ((vertices[a] + vertices[b]) * 0.5).normalize();
                            vertices.push(midpoint);
                            vertices.len() - 1
                        })
                    })
                    .collect();

                // Split into 4 triangles
                next_triples.push([*v0, mid[1], mid[2]]);
                next_triples.push([*v1, mid[0], mid[2]]);
                next_triples.push([*v2, mid[0], mid[1]]);
                next_triples.push([mid[0], mid[1], mid[2]]);
            }

            triples = next_triples;
        }

        Self { vertices }
    }

    /// Iterate over all points on a sphere surface.
    pub fn points_on_sphere(
        &self,
        center: Point3<f64>,
        radius: f64,
    ) -> impl Iterator<Item = Point3<f64>> + '_ {
        self.vertices.iter().map(move |v| center + v * radius)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_counts() {
        use SubdivisionDepth::*;
        assert_eq!(SubdividedIcosahedron::new(Depth0).vertices.len(), 12);
        assert_eq!(SubdividedIcosahedron::new(Depth1).vertices.len(), 42);
        assert_eq!(SubdividedIcosahedron::new(Depth2).vertices.len(), 162);
        assert_eq!(SubdividedIcosahedron::new(Depth3).vertices.len(), 642);
        assert_eq!(SubdividedIcosahedron::new(Depth4).vertices.len(), 2562);
    }

    #[test]
    fn from_u32_clamps() {
        use SubdivisionDepth::*;
        assert_eq!(SubdivisionDepth::from(0), Depth0);
        assert_eq!(SubdivisionDepth::from(1), Depth1);
        assert_eq!(SubdivisionDepth::from(2), Depth2);
        assert_eq!(SubdivisionDepth::from(3), Depth3);
        assert_eq!(SubdivisionDepth::from(4), Depth4);
        assert_eq!(SubdivisionDepth::from(5), Depth4); // Clamps to max
        assert_eq!(SubdivisionDepth::from(100), Depth4);
    }

    #[test]
    fn vertices_on_unit_sphere() {
        let sih = SubdividedIcosahedron::new(SubdivisionDepth::Depth2);
        for (i, v) in sih.vertices.iter().enumerate() {
            let dist = v.norm();
            assert!(
                (dist - 1.0).abs() < 1e-10,
                "vertex {i} not on unit sphere: {dist}"
            );
        }
    }

    #[test]
    fn points_on_arbitrary_sphere() {
        let sih = SubdividedIcosahedron::new(SubdivisionDepth::Depth1);
        let center = Point3::new(1.0, 2.0, 3.0);
        let radius = 5.0;

        for p in sih.points_on_sphere(center, radius) {
            let dist = nalgebra::distance(&p, &center);
            assert!((dist - radius).abs() < 1e-10);
        }
    }
}
