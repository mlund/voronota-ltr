use std::f64::consts::TAU;

#[cfg(test)]
use std::f64::consts::PI;

use nalgebra::{Point3, Unit, UnitQuaternion, Vector3};

use crate::types::Sphere;

/// Tolerance for floating-point comparisons.
/// This algorithm requires consistent epsilon across all comparisons
/// to maintain geometric consistency (matching C++ voronota-lt behavior).
pub const EPSILON: f64 = 1e-10;

/// Epsilon-based floating point comparisons.
/// These exist because the tessellation algorithm requires consistent
/// "fuzzy" comparisons to handle geometric degeneracies robustly.
pub mod float_cmp {
    use super::EPSILON;

    #[inline]
    pub const fn eq(a: f64, b: f64) -> bool {
        (a - b).abs() <= EPSILON
    }

    #[inline]
    pub const fn lt(a: f64, b: f64) -> bool {
        a + EPSILON < b
    }

    #[inline]
    pub const fn gt(a: f64, b: f64) -> bool {
        a - EPSILON > b
    }

    #[inline]
    pub const fn le(a: f64, b: f64) -> bool {
        a < b + EPSILON
    }

    #[inline]
    pub const fn ge(a: f64, b: f64) -> bool {
        a + EPSILON > b
    }
}

use float_cmp::{eq, ge, gt, le, lt};

#[inline]
pub fn point_equals(a: &Point3<f64>, b: &Point3<f64>) -> bool {
    eq(a.x, b.x) && eq(a.y, b.y) && eq(a.z, b.z)
}

/// Check if two spheres intersect (overlap)
#[inline]
pub fn sphere_intersects_sphere(a: &Sphere, b: &Sphere) -> bool {
    let sum_r = a.r + b.r;
    lt((b.center - a.center).norm_squared(), sum_r * sum_r)
}

/// Check if spheres are equal
#[inline]
pub fn sphere_equals_sphere(a: &Sphere, b: &Sphere) -> bool {
    eq(a.r, b.r) && point_equals(&a.center, &b.center)
}

/// Check if sphere `a` contains sphere `b`
#[inline]
pub fn sphere_contains_sphere(a: &Sphere, b: &Sphere) -> bool {
    let diff_r = a.r - b.r;
    ge(a.r, b.r) && le((b.center - a.center).norm_squared(), diff_r * diff_r)
}

/// Signed distance from point to plane (normalizes `plane_normal`)
#[inline]
pub fn signed_distance_to_plane(
    plane_point: &Point3<f64>,
    plane_normal: &Vector3<f64>,
    x: &Point3<f64>,
) -> f64 {
    signed_distance_to_plane_unit(plane_point, &plane_normal.normalize(), x)
}

/// Signed distance from point to plane (assumes `plane_normal` is unit length)
#[inline]
pub fn signed_distance_to_plane_unit(
    plane_point: &Point3<f64>,
    plane_normal: &Vector3<f64>,
    x: &Point3<f64>,
) -> f64 {
    plane_normal.dot(&(x - plane_point))
}

/// Determine which halfspace a point lies in relative to a plane
/// Returns: 1 if positive side, -1 if negative side, 0 if on plane
#[allow(dead_code)] // Public API, used internally via _unit variant
#[inline]
pub fn halfspace_of_point(
    plane_point: &Point3<f64>,
    plane_normal: &Vector3<f64>,
    x: &Point3<f64>,
) -> i32 {
    halfspace_of_point_unit(plane_point, &plane_normal.normalize(), x)
}

/// Determine which halfspace a point lies in (assumes `plane_normal` is unit length)
#[inline]
pub fn halfspace_of_point_unit(
    plane_point: &Point3<f64>,
    plane_normal: &Vector3<f64>,
    x: &Point3<f64>,
) -> i32 {
    let sd = signed_distance_to_plane_unit(plane_point, plane_normal, x);
    if gt(sd, 0.0) {
        1
    } else if lt(sd, 0.0) {
        -1
    } else {
        0
    }
}

/// Find intersection of a plane and a line segment
pub fn intersection_of_plane_and_segment(
    plane_point: &Point3<f64>,
    plane_normal: &Vector3<f64>,
    a: &Point3<f64>,
    b: &Point3<f64>,
) -> Point3<f64> {
    let da = signed_distance_to_plane(plane_point, plane_normal, a);
    let db = signed_distance_to_plane(plane_point, plane_normal, b);
    if (da - db).abs() < EPSILON {
        *a
    } else {
        let t = da / (da - db);
        a + (b - a) * t
    }
}

/// Triangle area from three points
#[inline]
pub fn triangle_area(a: &Point3<f64>, b: &Point3<f64>, c: &Point3<f64>) -> f64 {
    (b - a).cross(&(c - a)).norm() / 2.0
}

/// Minimum angle at vertex o between rays to a and b
pub fn min_angle(o: &Point3<f64>, a: &Point3<f64>, b: &Point3<f64>) -> f64 {
    let v1 = (a - o).normalize();
    let v2 = (b - o).normalize();
    v1.dot(&v2).clamp(-1.0, 1.0).acos()
}

/// Directed angle from ray oa to ray ob, using c to determine direction
#[allow(clippy::many_single_char_names)]
pub fn directed_angle(o: &Point3<f64>, a: &Point3<f64>, b: &Point3<f64>, c: &Point3<f64>) -> f64 {
    let angle = min_angle(o, a, b);
    let v1 = (a - o).normalize();
    let v2 = (b - o).normalize();
    let n = v1.cross(&v2);
    if (c - o).dot(&n) >= 0.0 {
        angle
    } else {
        TAU - angle
    }
}

/// Find any vector perpendicular to the given vector
pub fn any_normal_of_vector(a: &Vector3<f64>) -> Vector3<f64> {
    let mut b = *a;

    // Find a non-parallel vector to cross with
    if !eq(b.x, 0.0) && (!eq(b.y, 0.0) || !eq(b.z, 0.0)) {
        b.x = -b.x;
        return a.cross(&b).normalize();
    } else if !eq(b.y, 0.0) && (!eq(b.x, 0.0) || !eq(b.z, 0.0)) {
        b.y = -b.y;
        return a.cross(&b).normalize();
    } else if !eq(b.x, 0.0) {
        return Vector3::new(0.0, 1.0, 0.0);
    }
    Vector3::new(1.0, 0.0, 0.0)
}

/// Rotate point around an axis by angle (radians) using quaternion
pub fn rotate_point_around_axis(axis: &Vector3<f64>, angle: f64, p: &Vector3<f64>) -> Vector3<f64> {
    if axis.norm_squared() <= 0.0 {
        return *p;
    }
    let unit_axis = Unit::new_normalize(*axis);
    let rotation = UnitQuaternion::from_axis_angle(&unit_axis, angle);
    rotation * p
}

/// Distance from sphere a center to intersection circle center with sphere b
pub fn distance_to_intersection_circle_center(a: &Sphere, b: &Sphere) -> f64 {
    let cm = (b.center - a.center).norm();
    if cm < EPSILON {
        return 0.0;
    }
    let cos_g = b.r.mul_add(-b.r, a.r.mul_add(a.r, cm * cm)) / (2.0 * a.r * cm);
    a.r * cos_g
}

/// Center of intersection circle of two spheres
pub fn center_of_intersection_circle(a: &Sphere, b: &Sphere) -> Point3<f64> {
    let cv = b.center - a.center;
    let cm = cv.norm();
    if cm < EPSILON {
        return a.center;
    }
    let cos_g = b.r.mul_add(-b.r, a.r.mul_add(a.r, cm * cm)) / (2.0 * a.r * cm);
    a.center + cv * (a.r * cos_g / cm)
}

/// Intersection circle of two spheres as a Sphere (center + radius)
pub fn intersection_circle_of_two_spheres(a: &Sphere, b: &Sphere) -> Sphere {
    let cv = b.center - a.center;
    let cm = cv.norm();
    if cm < EPSILON {
        return Sphere::new(a.center, 0.0);
    }
    let cos_g = b.r.mul_add(-b.r, a.r.mul_add(a.r, cm * cm)) / (2.0 * a.r * cm);
    let sin_g = (1.0 - cos_g * cos_g).max(0.0).sqrt();
    let center = a.center + cv * (a.r * cos_g / cm);
    Sphere::new(center, a.r * sin_g)
}

/// Project point o onto line segment ab, return Some if projection is inside segment
#[allow(clippy::many_single_char_names)]
pub fn project_point_inside_line(
    o: &Point3<f64>,
    a: &Point3<f64>,
    b: &Point3<f64>,
) -> Option<Point3<f64>> {
    let v = (b - a).normalize();
    let l = v.dot(&(o - a));
    if l > 0.0 && l * l <= (b - a).norm_squared() {
        Some(a + v * l)
    } else {
        None
    }
}

/// Intersect segment (from `p_out` toward `p_in`) with circle, finding the intersection closest to `p_out`
pub fn intersect_segment_with_circle(
    circle: &Sphere,
    p_in: &Point3<f64>,
    p_out: &Point3<f64>,
) -> Option<Point3<f64>> {
    let dist = (p_in - p_out).norm();
    if dist <= 0.0 {
        return None;
    }

    let v = (p_in - p_out) / dist;
    let u = circle.center - p_out;
    let s = p_out + v * v.dot(&u);
    let ll = circle
        .r
        .mul_add(circle.r, -(circle.center - s).norm_squared());

    if ll >= 0.0 {
        Some(s - v * ll.sqrt())
    } else {
        None
    }
}

/// Minimum dihedral angle
pub fn min_dihedral_angle(
    o: &Point3<f64>,
    a: &Point3<f64>,
    b1: &Point3<f64>,
    b2: &Point3<f64>,
) -> f64 {
    let oa = (a - o).normalize();
    let d1 = b1 - (o + oa * oa.dot(&(b1 - o)));
    let d2 = b2 - (o + oa * oa.dot(&(b2 - o)));
    d1.normalize().dot(&d2.normalize()).clamp(-1.0, 1.0).acos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sphere_intersects() {
        let s1 = Sphere::from_coords(0.0, 0.0, 0.0, 1.0);
        let s2 = Sphere::from_coords(1.5, 0.0, 0.0, 1.0);
        assert!(sphere_intersects_sphere(&s1, &s2));

        let s3 = Sphere::from_coords(3.0, 0.0, 0.0, 1.0);
        assert!(!sphere_intersects_sphere(&s1, &s3));
    }

    #[test]
    fn test_sphere_contains() {
        let outer = Sphere::from_coords(0.0, 0.0, 0.0, 3.0);
        let inner = Sphere::from_coords(0.5, 0.0, 0.0, 1.0);
        assert!(sphere_contains_sphere(&outer, &inner));
        assert!(!sphere_contains_sphere(&inner, &outer));
    }

    #[test]
    fn test_intersection_circle() {
        let s1 = Sphere::from_coords(0.0, 0.0, 0.0, 1.0);
        let s2 = Sphere::from_coords(1.0, 0.0, 0.0, 1.0);
        let ic = intersection_circle_of_two_spheres(&s1, &s2);
        assert_relative_eq!(ic.center.x, 0.5, epsilon = 1e-9);
        assert_relative_eq!(ic.center.y, 0.0, epsilon = 1e-9);
        assert!(ic.r > 0.0);
    }

    #[test]
    fn test_rotate_point() {
        let axis = Vector3::new(0.0, 0.0, 1.0);
        let p = Vector3::new(1.0, 0.0, 0.0);
        let rotated = rotate_point_around_axis(&axis, PI / 2.0, &p);
        assert_relative_eq!(rotated.x, 0.0, epsilon = 1e-9);
        assert_relative_eq!(rotated.y, 1.0, epsilon = 1e-9);
        assert_relative_eq!(rotated.z, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn test_any_normal() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let n = any_normal_of_vector(&v);
        assert_relative_eq!(v.dot(&n), 0.0, epsilon = 1e-9);
        assert_relative_eq!(n.norm(), 1.0, epsilon = 1e-9);
    }
}
