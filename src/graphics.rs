// Copyright (c) 2026 Kliment Olechnovic and Mikael Lund
// Part of the voronota-ltr project, licensed under the MIT License.
// SPDX-License-Identifier: MIT

//! `PyMOL` CGO graphics output for visualizing tessellation contacts.
//!
//! Generates Python scripts using `PyMOL`'s Compiled Graphics Objects (CGO)
//! to render contact surfaces as triangle fans and sphere boundaries.

use std::f64::consts::TAU;
use std::io::{self, Write};

use nalgebra::{Point3, Vector3};

use crate::contact::ContactDescriptor;
use crate::geometry::{any_normal_of_vector, directed_angle, rotate_point_around_axis};
use crate::types::Ball;

/// Graphics data for rendering a contact surface.
#[derive(Debug, Clone)]
pub struct ContactGraphics {
    /// Polygon vertices forming the contact boundary
    pub outer_points: Vec<Point3<f64>>,
    /// Center point for triangle fan rendering
    pub barycenter: Point3<f64>,
    /// Surface normal for lighting
    pub plane_normal: Vector3<f64>,
}

/// Collects graphics primitives and writes `PyMOL` CGO scripts.
pub struct GraphicsWriter {
    balls: Vec<Ball>,
    faces: Vec<ContactGraphics>,
}

impl GraphicsWriter {
    /// Create a new empty writer for the given input balls.
    #[must_use]
    pub const fn new(balls: Vec<Ball>) -> Self {
        Self {
            balls,
            faces: Vec::new(),
        }
    }

    /// Compute tessellation and create graphics for all contacts.
    ///
    /// # Arguments
    /// * `balls` - Input spheres
    /// * `probe` - Probe radius
    /// * `groups` - Optional grouping; contacts within same group are excluded
    /// * `angle_step` - Angular resolution for arc interpolation (radians)
    #[must_use]
    pub fn from_balls(balls: &[Ball], probe: f64, groups: Option<&[i32]>, angle_step: f64) -> Self {
        use crate::contact::construct_contact_descriptor;
        use crate::spheres_searcher::SpheresSearcher;
        use crate::types::Sphere;

        let spheres: Vec<Sphere> = balls.iter().map(|b| Sphere::from_ball(b, probe)).collect();
        let searcher = SpheresSearcher::new(spheres);
        let spheres = searcher.spheres();

        let all_collisions: Vec<Vec<crate::types::ValuedId>> = (0..spheres.len())
            .map(|id| searcher.find_colliding_ids(id, true).colliding_ids)
            .collect();

        let mut writer = Self::new(balls.to_vec());

        let same_group = |a: usize, b: usize| -> bool {
            groups.is_some_and(|g| a < g.len() && b < g.len() && g[a] == g[b])
        };

        for (a_id, neighbors) in all_collisions.iter().enumerate() {
            for neighbor in neighbors {
                let b_id = neighbor.index;
                if a_id < b_id
                    && !same_group(a_id, b_id)
                    && let Some(cd) = construct_contact_descriptor(spheres, a_id, b_id, neighbors)
                    && let Some(graphics) = cd.to_graphics(angle_step)
                {
                    writer.add_face(graphics);
                }
            }
        }

        writer
    }

    /// Add a contact face for rendering.
    pub fn add_face(&mut self, graphics: ContactGraphics) {
        self.faces.push(graphics);
    }

    /// Number of collected faces.
    #[must_use]
    pub const fn num_faces(&self) -> usize {
        self.faces.len()
    }

    /// Write `PyMOL` CGO Python script to the given writer.
    ///
    /// # Errors
    /// Returns an error if writing to the output fails.
    pub fn write_pymol<W: Write>(&self, mut writer: W, object_name: &str) -> io::Result<()> {
        writeln!(writer, "from pymol.cgo import *")?;
        writeln!(writer, "from pymol import cmd")?;
        writeln!(writer)?;

        self.write_balls_cgo(&mut writer, object_name)?;
        writeln!(writer)?;

        self.write_faces_cgo(&mut writer, object_name)?;
        writeln!(writer)?;

        self.write_wireframe_cgo(&mut writer, object_name)?;
        writeln!(writer)?;

        // PyMOL rendering settings
        writeln!(writer, "cmd.set('two_sided_lighting', 1)")?;
        writeln!(writer, "cmd.set('cgo_line_width', 1)")?;

        Ok(())
    }

    /// Write ball spheres as CGO.
    fn write_balls_cgo<W: Write>(&self, writer: &mut W, object_name: &str) -> io::Result<()> {
        writeln!(writer, "cgo_graphics_list_balls = [")?;
        writeln!(writer, "    COLOR, 0, 1, 1,")?;

        for ball in &self.balls {
            writeln!(
                writer,
                "    SPHERE, {:.6}, {:.6}, {:.6}, {:.6},",
                ball.x, ball.y, ball.z, ball.r
            )?;
        }

        writeln!(writer, "]")?;
        writeln!(
            writer,
            "cmd.load_cgo(cgo_graphics_list_balls, '{object_name}_balls')"
        )?;

        Ok(())
    }

    /// Write contact faces as CGO triangle fans.
    fn write_faces_cgo<W: Write>(&self, writer: &mut W, object_name: &str) -> io::Result<()> {
        writeln!(writer, "cgo_graphics_list_faces = [")?;
        writeln!(writer, "    COLOR, 1, 1, 0,")?;

        for face in &self.faces {
            if face.outer_points.len() < 3 {
                continue;
            }

            writeln!(writer, "    BEGIN, TRIANGLE_FAN,")?;
            writeln!(
                writer,
                "    NORMAL, {:.6}, {:.6}, {:.6},",
                face.plane_normal.x, face.plane_normal.y, face.plane_normal.z
            )?;

            // Fan center must come first in CGO TRIANGLE_FAN
            writeln!(
                writer,
                "    VERTEX, {:.6}, {:.6}, {:.6},",
                face.barycenter.x, face.barycenter.y, face.barycenter.z
            )?;

            // Close the fan by repeating first boundary vertex
            for p in &face.outer_points {
                writeln!(writer, "    VERTEX, {:.6}, {:.6}, {:.6},", p.x, p.y, p.z)?;
            }
            writeln!(
                writer,
                "    VERTEX, {:.6}, {:.6}, {:.6},",
                face.outer_points[0].x, face.outer_points[0].y, face.outer_points[0].z
            )?;

            writeln!(writer, "    END,")?;
        }

        writeln!(writer, "]")?;
        writeln!(
            writer,
            "cmd.load_cgo(cgo_graphics_list_faces, '{object_name}_faces')"
        )?;

        Ok(())
    }

    /// Write contact boundaries as CGO line loops.
    fn write_wireframe_cgo<W: Write>(&self, writer: &mut W, object_name: &str) -> io::Result<()> {
        writeln!(writer, "cgo_graphics_list_wireframe = [")?;
        writeln!(writer, "    COLOR, 1, 0, 0,")?;

        for face in &self.faces {
            if face.outer_points.len() < 3 {
                continue;
            }

            writeln!(writer, "    BEGIN, LINE_LOOP,")?;
            for p in &face.outer_points {
                writeln!(writer, "    VERTEX, {:.6}, {:.6}, {:.6},", p.x, p.y, p.z)?;
            }
            writeln!(writer, "    END,")?;
        }

        writeln!(writer, "]")?;
        writeln!(
            writer,
            "cmd.load_cgo(cgo_graphics_list_wireframe, '{object_name}_wireframe')"
        )?;

        Ok(())
    }
}

impl ContactDescriptor {
    /// Convert contact to graphics representation for visualization.
    ///
    /// Returns `None` if the contact has zero area or cannot be visualized.
    ///
    /// # Arguments
    /// * `angle_step` - Angular resolution for arc interpolation (radians).
    ///   Smaller values produce smoother curves.
    #[must_use]
    pub fn to_graphics(&self, angle_step: f64) -> Option<ContactGraphics> {
        if self.area <= 0.0 {
            return None;
        }

        let outer_points = if self.contour.is_empty() {
            // No neighbors cut this contact, so it's a complete circle
            generate_full_circle_polygon(
                &self.intersection_circle.center,
                self.intersection_circle.r,
                &self.axis,
                angle_step,
            )
        } else {
            // Neighbors cut the contact into a polygon; interpolate arcs on SAS edges
            construct_contact_polygon(self, angle_step)
        };

        // Degenerate contacts with <3 boundary points (thin slivers) can't form polygons
        if outer_points.len() < 3 {
            return None;
        }

        Some(ContactGraphics {
            outer_points,
            barycenter: self.contour_barycenter,
            plane_normal: self.axis,
        })
    }
}

/// Generate vertices for a full circular contact.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn generate_full_circle_polygon(
    center: &Point3<f64>,
    radius: f64,
    axis: &Vector3<f64>,
    angle_step: f64,
) -> Vec<Point3<f64>> {
    let first_offset = any_normal_of_vector(axis) * radius;
    let num_segments = (TAU / angle_step).ceil() as usize;
    let actual_step = TAU / num_segments as f64;

    (0..num_segments)
        .map(|i| {
            let angle = i as f64 * actual_step;
            let rotated = rotate_point_around_axis(axis, angle, &first_offset);
            center + rotated
        })
        .collect()
}

/// Construct polygon vertices from contact contour with arc interpolation.
fn construct_contact_polygon(cd: &ContactDescriptor, angle_step: f64) -> Vec<Point3<f64>> {
    let mut points = Vec::with_capacity(cd.contour.len() * 2);

    for i in 0..cd.contour.len() {
        let cp = &cd.contour[i];
        let next = &cd.contour[(i + 1) % cd.contour.len()];

        points.push(cp.p);

        // Interpolate arc if this edge lies on the SAS boundary
        if cp.right_id == cd.id_a && cp.angle > 0.0 {
            let arc_points = interpolate_arc_points(
                &cd.intersection_circle.center,
                cd.intersection_circle.r,
                &cd.axis,
                &cp.p,
                &next.p,
                cp.angle,
                angle_step,
            );
            points.extend(arc_points);
        }
    }

    points
}

/// Interpolate points along a circular arc.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn interpolate_arc_points(
    center: &Point3<f64>,
    radius: f64,
    axis: &Vector3<f64>,
    start: &Point3<f64>,
    end: &Point3<f64>,
    total_angle: f64,
    angle_step: f64,
) -> Vec<Point3<f64>> {
    if total_angle <= angle_step {
        return Vec::new();
    }

    // Compute the actual arc angle using directed_angle to ensure proper winding
    let computed_angle = directed_angle(center, start, end, &(center + axis)).min(total_angle);

    let num_segments = (computed_angle / angle_step).floor() as usize;
    if num_segments <= 1 {
        return Vec::new();
    }

    let start_offset = start - center;
    let actual_step = computed_angle / num_segments as f64;

    (1..num_segments)
        .map(|i| {
            let angle = i as f64 * actual_step;
            let rotated = rotate_point_around_axis(axis, angle, &start_offset);
            center + rotated.normalize() * radius
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Ball;
    use approx::assert_relative_eq;

    #[test]
    fn test_full_circle_polygon() {
        let center = Point3::new(0.0, 0.0, 0.0);
        let axis = Vector3::new(0.0, 0.0, 1.0);
        let radius = 1.0;
        let angle_step = std::f64::consts::PI / 6.0; // 30 degrees

        let points = generate_full_circle_polygon(&center, radius, &axis, angle_step);

        // 12 segments for 30-degree steps
        assert_eq!(points.len(), 12);

        // All points should be at radius distance from center
        for p in &points {
            let dist = (p - center).norm();
            assert_relative_eq!(dist, radius, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_graphics_writer_output() {
        let balls = vec![Ball::new(0.0, 0.0, 0.0, 1.0)];
        let writer = GraphicsWriter::new(balls);

        let mut output = Vec::new();
        writer.write_pymol(&mut output, "test").unwrap();

        let content = String::from_utf8(output).unwrap();
        assert!(content.contains("from pymol.cgo import *"));
        assert!(content.contains("SPHERE"));
        assert!(content.contains("test_balls"));
    }

    /// Test 3 balls in a line - should produce 2 faces
    #[test]
    fn test_three_balls_line_graphics() {
        let balls = vec![
            Ball::new(0.0, 0.0, 0.0, 1.0),
            Ball::new(0.5, 0.0, 0.0, 1.0),
            Ball::new(1.0, 0.0, 0.0, 1.0),
        ];

        // Get internal contact descriptors by using the tessellation internals
        let spheres: Vec<crate::types::Sphere> = balls
            .iter()
            .map(|b| crate::types::Sphere::from_ball(b, 1.0))
            .collect();
        let searcher = crate::spheres_searcher::SpheresSearcher::new(spheres);

        let all_collisions: Vec<Vec<crate::types::ValuedId>> = (0..3)
            .map(|id| searcher.find_colliding_ids(id, true).colliding_ids)
            .collect();

        let mut face_count = 0;
        for a_id in 0..3 {
            for neighbor in &all_collisions[a_id] {
                let b_id = neighbor.index;
                if a_id < b_id {
                    if let Some(cd) = crate::contact::construct_contact_descriptor(
                        searcher.spheres(),
                        a_id,
                        b_id,
                        &all_collisions[a_id],
                    ) {
                        if cd.to_graphics(0.5).is_some() {
                            face_count += 1;
                        }
                    }
                }
            }
        }

        assert_eq!(face_count, 2, "3 balls in line should produce 2 faces");
    }
}
