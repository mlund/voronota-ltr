//! Subdivided icosahedron for uniform sphere surface sampling.

use nalgebra::Point3;

/// Generates uniformly distributed points on a unit sphere via icosahedron subdivision.
///
/// Each subdivision level splits every triangle into 4 smaller triangles,
/// projecting new vertices onto the unit sphere.
pub struct SubdividedIcosahedron {
    vertices: Vec<Point3<f64>>,
}

impl SubdividedIcosahedron {
    /// Create a subdivided icosahedron with the given depth.
    ///
    /// Vertex counts by depth:
    /// - 0: 12 points (base icosahedron)
    /// - 1: 42 points
    /// - 2: 162 points
    /// - 3: 642 points
    /// - 4: 2562 points
    #[must_use]
    #[allow(clippy::manual_midpoint)] // This is golden ratio, not midpoint
    pub fn new(depth: u32) -> Self {
        let t = (1.0 + 5.0_f64.sqrt()) / 2.0; // Golden ratio φ = (1+√5)/2

        // 12 vertices of a regular icosahedron
        let mut vertices: Vec<Point3<f64>> = [
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
        .map(|(x, y, z)| unit_point(Point3::new(x, y, z)))
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

        // Subdivide each triangle into 4 smaller triangles
        for _ in 0..depth {
            let mut next_triples = Vec::with_capacity(triples.len() * 4);
            let mut edge_midpoints = std::collections::HashMap::new();

            for triple in &triples {
                let mut mid = [0usize; 3];

                // Find/create midpoint for each edge
                for (j, &(a, b)) in [
                    (triple[1], triple[2]),
                    (triple[0], triple[2]),
                    (triple[0], triple[1]),
                ]
                .iter()
                .enumerate()
                {
                    let key = if a < b { (a, b) } else { (b, a) };
                    mid[j] = *edge_midpoints.entry(key).or_insert_with(|| {
                        let midpoint = unit_point(Point3::new(
                            (vertices[a].x + vertices[b].x) * 0.5,
                            (vertices[a].y + vertices[b].y) * 0.5,
                            (vertices[a].z + vertices[b].z) * 0.5,
                        ));
                        vertices.push(midpoint);
                        vertices.len() - 1
                    });
                }

                // Split into 4 triangles
                next_triples.push([triple[0], mid[1], mid[2]]);
                next_triples.push([triple[1], mid[0], mid[2]]);
                next_triples.push([triple[2], mid[0], mid[1]]);
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
        self.vertices.iter().map(move |v| {
            Point3::new(
                v.x.mul_add(radius, center.x),
                v.y.mul_add(radius, center.y),
                v.z.mul_add(radius, center.z),
            )
        })
    }
}

/// Normalize point to unit sphere.
fn unit_point(p: Point3<f64>) -> Point3<f64> {
    let len = p.z.mul_add(p.z, p.x.mul_add(p.x, p.y * p.y)).sqrt();
    Point3::new(p.x / len, p.y / len, p.z / len)
}

#[cfg(test)]
mod tests {
    use super::*;

    impl SubdividedIcosahedron {
        fn len(&self) -> usize {
            self.vertices.len()
        }

        fn point_on_sphere(&self, i: usize, center: Point3<f64>, radius: f64) -> Point3<f64> {
            let v = &self.vertices[i];
            Point3::new(
                v.x.mul_add(radius, center.x),
                v.y.mul_add(radius, center.y),
                v.z.mul_add(radius, center.z),
            )
        }
    }

    #[test]
    fn vertex_counts() {
        assert_eq!(SubdividedIcosahedron::new(0).len(), 12);
        assert_eq!(SubdividedIcosahedron::new(1).len(), 42);
        assert_eq!(SubdividedIcosahedron::new(2).len(), 162);
        assert_eq!(SubdividedIcosahedron::new(3).len(), 642);
    }

    #[test]
    fn vertices_on_unit_sphere() {
        let sih = SubdividedIcosahedron::new(2);
        for i in 0..sih.len() {
            let p = sih.point_on_sphere(i, Point3::origin(), 1.0);
            let dist = p.z.mul_add(p.z, p.x.mul_add(p.x, p.y * p.y)).sqrt();
            assert!(
                (dist - 1.0).abs() < 1e-10,
                "vertex {i} not on unit sphere: {dist}"
            );
        }
    }

    #[test]
    fn points_on_arbitrary_sphere() {
        let sih = SubdividedIcosahedron::new(1);
        let center = Point3::new(1.0, 2.0, 3.0);
        let radius = 5.0;

        for p in sih.points_on_sphere(center, radius) {
            let dx = p.x - center.x;
            let dy = p.y - center.y;
            let dz = p.z - center.z;
            let dist = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();
            assert!((dist - radius).abs() < 1e-10);
        }
    }
}
