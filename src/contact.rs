use std::f64::consts::{FRAC_PI_3, PI, TAU};

use nalgebra::{Point3, Vector3};

use crate::geometry::{
    any_normal_of_vector, center_of_intersection_circle, directed_angle, float_cmp,
    halfspace_of_point, intersect_segment_with_circle, intersection_circle_of_two_spheres,
    intersection_of_plane_and_segment, min_dihedral_angle, project_point_inside_line,
    rotate_point_around_axis, signed_distance_to_plane, sphere_contains_sphere,
    sphere_intersects_sphere, triangle_area,
};
use crate::types::{ContactDescriptorSummary, Sphere, ValuedId};

/// A point on the contact contour
#[derive(Debug, Clone)]
struct ContourPoint {
    p: Point3<f64>,
    angle: f64,
    left_id: usize,
    right_id: usize,
    indicator: i32,
}

impl ContourPoint {
    fn new(p: Point3<f64>, left_id: usize, right_id: usize) -> Self {
        Self {
            p,
            angle: 0.0,
            left_id,
            right_id,
            indicator: 0,
        }
    }
}

type Contour = Vec<ContourPoint>;

/// Full contact descriptor (internal use)
#[derive(Debug, Clone)]
pub struct ContactDescriptor {
    contour: Contour,
    pub intersection_circle: Sphere,
    pub axis: Vector3<f64>,
    pub contour_barycenter: Point3<f64>,
    pub sum_of_arc_angles: f64,
    pub area: f64,
    pub solid_angle_a: f64,
    pub solid_angle_b: f64,
    pub pyramid_volume_a: f64,
    pub pyramid_volume_b: f64,
    pub distance: f64,
    pub id_a: usize,
    pub id_b: usize,
}

impl Default for ContactDescriptor {
    fn default() -> Self {
        Self {
            contour: Vec::new(),
            intersection_circle: Sphere::from_coords(0.0, 0.0, 0.0, 0.0),
            axis: Vector3::zeros(),
            contour_barycenter: Point3::origin(),
            sum_of_arc_angles: 0.0,
            area: 0.0,
            solid_angle_a: 0.0,
            solid_angle_b: 0.0,
            pyramid_volume_a: 0.0,
            pyramid_volume_b: 0.0,
            distance: 0.0,
            id_a: 0,
            id_b: 0,
        }
    }
}

impl ContactDescriptor {
    pub fn to_summary(&self) -> ContactDescriptorSummary {
        ContactDescriptorSummary {
            area: self.area,
            arc_length: self.sum_of_arc_angles * self.intersection_circle.r,
            solid_angle_a: self.solid_angle_a,
            solid_angle_b: self.solid_angle_b,
            pyramid_volume_a: self.pyramid_volume_a,
            pyramid_volume_b: self.pyramid_volume_b,
            distance: self.distance,
            id_a: self.id_a,
            id_b: self.id_b,
        }
    }
}

/// Construct contact descriptor between spheres a and b
pub fn construct_contact_descriptor(
    spheres: &[Sphere],
    a_id: usize,
    b_id: usize,
    neighbors: &[ValuedId],
) -> Option<ContactDescriptor> {
    if a_id >= spheres.len() || b_id >= spheres.len() {
        return None;
    }

    let a = &spheres[a_id];
    let b = &spheres[b_id];

    // Check basic intersection conditions
    if !sphere_intersects_sphere(a, b)
        || sphere_contains_sphere(a, b)
        || sphere_contains_sphere(b, a)
    {
        return None;
    }

    let mut cd = ContactDescriptor {
        id_a: a_id,
        id_b: b_id,
        intersection_circle: intersection_circle_of_two_spheres(a, b),
        ..Default::default()
    };

    if cd.intersection_circle.r <= 0.0 {
        return None;
    }

    let mut discarded = false;
    let mut contour_initialized = false;

    // Process neighbor spheres that might cut the contact
    for neighbor in neighbors {
        if discarded {
            break;
        }

        let neighbor_id = neighbor.index;
        if neighbor_id == b_id {
            continue;
        }

        let c = &spheres[neighbor_id];

        // Check if neighbor affects the contact
        if !sphere_intersects_sphere(&cd.intersection_circle, c) || !sphere_intersects_sphere(b, c)
        {
            continue;
        }

        // If neighbor contains a or b, contact is invalid
        if sphere_contains_sphere(c, a) || sphere_contains_sphere(c, b) {
            discarded = true;
            continue;
        }

        let ac_plane_center = center_of_intersection_circle(a, c);
        let ac_plane_normal = (c.center - a.center).normalize();

        // Check angle between intersection circles
        let cos_val = (cd.intersection_circle.center - a.center)
            .normalize()
            .dot(&(ac_plane_center - a.center).normalize());

        if cos_val.abs() >= 1.0 {
            // Planes are parallel
            if halfspace_of_point(
                &ac_plane_center,
                &ac_plane_normal,
                &cd.intersection_circle.center,
            ) > 0
            {
                discarded = true;
            }
            continue;
        }

        // Calculate distance from intersection circle center to the cutting plane
        let l = signed_distance_to_plane(
            &ac_plane_center,
            &ac_plane_normal,
            &cd.intersection_circle.center,
        )
        .abs();
        let xl = l / (1.0 - cos_val * cos_val).sqrt();

        if xl >= cd.intersection_circle.r {
            // Cutting plane doesn't intersect the circle
            if halfspace_of_point(
                &ac_plane_center,
                &ac_plane_normal,
                &cd.intersection_circle.center,
            ) >= 0
            {
                discarded = true;
            }
            continue;
        }

        // Initialize contour if needed
        if !contour_initialized {
            cd.axis = (b.center - a.center).normalize();
            init_contour(&mut cd.contour, a_id, &cd.intersection_circle, &cd.axis);
            contour_initialized = true;
        } else if !test_contour_cuttable(&a.center, &ac_plane_center, &cd.contour) {
            continue;
        }

        // Cut the contour
        mark_and_cut_contour(
            &mut cd.contour,
            &ac_plane_center,
            &ac_plane_normal,
            neighbor_id,
        );

        if cd.contour.is_empty() {
            discarded = true;
        }
    }

    if discarded {
        return None;
    }

    // Finalize the contact descriptor
    if !contour_initialized {
        // Full circle contact (no cuts)
        cd.axis = (b.center - a.center).normalize();
        cd.contour_barycenter = cd.intersection_circle.center;
        cd.sum_of_arc_angles = TAU;
        cd.area = cd.intersection_circle.r * cd.intersection_circle.r * PI;
    } else if !cd.contour.is_empty() {
        restrict_contour_to_circle(
            &mut cd.contour,
            &cd.intersection_circle,
            &cd.axis,
            a_id,
            &mut cd.sum_of_arc_angles,
        );

        if !cd.contour.is_empty() {
            cd.area = calculate_contour_area(
                &cd.contour,
                &cd.intersection_circle,
                &mut cd.contour_barycenter,
            );
        }
    }

    if cd.area <= 0.0 {
        return None;
    }

    // Calculate solid angles and pyramid volumes
    cd.solid_angle_a = calculate_solid_angle(a, b, &cd.intersection_circle, &cd.contour);
    cd.solid_angle_b = calculate_solid_angle(b, a, &cd.intersection_circle, &cd.contour);

    let sign_a = if cd.solid_angle_a < 0.0 { -1.0 } else { 1.0 };
    let sign_b = if cd.solid_angle_b < 0.0 { -1.0 } else { 1.0 };

    cd.pyramid_volume_a =
        (cd.intersection_circle.center - a.center).norm() * cd.area / 3.0 * sign_a;
    cd.pyramid_volume_b =
        (cd.intersection_circle.center - b.center).norm() * cd.area / 3.0 * sign_b;

    cd.distance = (b.center - a.center).norm();

    Some(cd)
}

/// Initialize hexagonal contour from intersection circle.
/// The contour starts as a regular hexagon inscribed in a circle slightly
/// larger than the intersection circle (factor 1.19 ≈ 1/cos(30°) ensures
/// the hexagon fully contains the circle for robust cutting).
fn init_contour(contour: &mut Contour, a_id: usize, base: &Sphere, axis: &Vector3<f64>) {
    contour.clear();

    // 1.19 ≈ 1/cos(30°): ensures hexagon vertices are outside the circle
    const HEXAGON_SCALE: f64 = 1.19;
    let first_point = any_normal_of_vector(axis) * base.r * HEXAGON_SCALE;
    let angle_step = FRAC_PI_3;

    contour.push(ContourPoint::new(base.center + first_point, a_id, a_id));

    let mut rotation_angle = angle_step;
    while rotation_angle < TAU {
        let rotated = rotate_point_around_axis(axis, rotation_angle, &first_point);
        contour.push(ContourPoint::new(base.center + rotated, a_id, a_id));
        rotation_angle += angle_step;
    }
}

/// Check if contour can still be cut (any point beyond cut plane)
fn test_contour_cuttable(
    a_center: &Point3<f64>,
    closest_cut_point: &Point3<f64>,
    contour: &Contour,
) -> bool {
    let threshold = (closest_cut_point - a_center).norm_squared();
    contour
        .iter()
        .any(|cp| (cp.p - a_center).norm_squared() >= threshold)
}

/// Mark contour points that are on the outside of the cutting plane and cut
fn mark_and_cut_contour(
    contour: &mut Contour,
    plane_center: &Point3<f64>,
    plane_normal: &Vector3<f64>,
    c_id: usize,
) -> bool {
    let outsiders = mark_contour(contour, plane_center, plane_normal, c_id);

    if outsiders == 0 {
        return false;
    }

    if outsiders >= contour.len() {
        contour.clear();
        return true;
    }

    cut_contour(contour, plane_center, plane_normal, c_id);
    true
}

/// Mark points outside the cutting plane
fn mark_contour(
    contour: &mut Contour,
    plane_center: &Point3<f64>,
    plane_normal: &Vector3<f64>,
    c_id: usize,
) -> usize {
    let mut count = 0;
    for cp in contour.iter_mut() {
        if halfspace_of_point(plane_center, plane_normal, &cp.p) >= 0 {
            cp.left_id = c_id;
            cp.right_id = c_id;
            count += 1;
        }
    }
    count
}

/// Cut contour by removing outside points and adding intersection points
fn cut_contour(
    contour: &mut Contour,
    plane_center: &Point3<f64>,
    plane_normal: &Vector3<f64>,
    c_id: usize,
) {
    if contour.len() < 3 {
        return;
    }

    // Find first and last outsider indices
    let i_start = contour
        .iter()
        .position(|cp| cp.left_id == c_id && cp.right_id == c_id);
    let i_end = contour
        .iter()
        .rposition(|cp| cp.left_id == c_id && cp.right_id == c_id);

    let (Some(mut start), Some(mut end)) = (i_start, i_end) else {
        return;
    };

    // Handle wrap-around case
    if start == 0 && end == contour.len() - 1 {
        // Find the actual start/end accounting for wrap
        end = 0;
        while end + 1 < contour.len()
            && contour[end + 1].left_id == c_id
            && contour[end + 1].right_id == c_id
        {
            end += 1;
        }

        start = contour.len() - 1;
        while start > 0 && contour[start - 1].left_id == c_id && contour[start - 1].right_id == c_id
        {
            start -= 1;
        }
    }

    // Handle different contour cut cases
    match start.cmp(&end) {
        std::cmp::Ordering::Equal => {
            // Single point case
            contour.insert(start, contour[start].clone());
            end = start + 1;
        }
        std::cmp::Ordering::Less => {
            // Remove points between start and end (exclusive)
            if start + 1 < end {
                contour.drain((start + 1)..end);
            }
            end = start + 1;
        }
        std::cmp::Ordering::Greater => {
            // Wrap-around: remove after start and before end
            if start + 1 < contour.len() {
                contour.drain((start + 1)..);
            }
            if end > 0 {
                contour.drain(0..end);
            }
            start = contour.len() - 1;
            end = 0;
        }
    }

    // Calculate intersection points
    let n = contour.len();

    let i_left = if start > 0 { start - 1 } else { n - 1 };
    let left_intersection = intersection_of_plane_and_segment(
        plane_center,
        plane_normal,
        &contour[start].p,
        &contour[i_left].p,
    );
    let new_start = ContourPoint::new(
        left_intersection,
        contour[i_left].right_id,
        contour[start].left_id,
    );
    contour[start] = new_start;

    let i_right = if end + 1 < n { end + 1 } else { 0 };
    let right_intersection = intersection_of_plane_and_segment(
        plane_center,
        plane_normal,
        &contour[end].p,
        &contour[i_right].p,
    );
    let new_end = ContourPoint::new(
        right_intersection,
        contour[end].right_id,
        contour[i_right].left_id,
    );
    contour[end] = new_end;

    // Merge very close points
    if !float_cmp::gt((contour[end].p - contour[start].p).norm_squared(), 0.0) {
        contour[start].right_id = contour[end].right_id;
        contour.remove(end);
    }
}

/// Restrict contour to lie on the intersection circle
fn restrict_contour_to_circle(
    contour: &mut Contour,
    ic: &Sphere,
    axis: &Vector3<f64>,
    a_id: usize,
    sum_angles: &mut f64,
) {
    *sum_angles = 0.0;

    // Mark points inside/outside circle
    let mut outsiders = 0;
    for cp in contour.iter_mut() {
        if (cp.p - ic.center).norm_squared() <= ic.r * ic.r {
            cp.indicator = 0;
        } else {
            cp.indicator = 1;
            outsiders += 1;
        }
    }

    if outsiders == 0 {
        return;
    }

    // Insert intersection points
    let mut insertions = 0;
    let mut i = 0;
    while i < contour.len() {
        let next_i = (i + 1) % contour.len();
        let pr1_indicator = contour[i].indicator;
        let pr2_indicator = contour[next_i].indicator;

        if pr1_indicator == 1 || pr2_indicator == 1 {
            if pr1_indicator == 1 && pr2_indicator == 1 {
                // Both outside: check if segment crosses circle
                if let Some(mp) =
                    project_point_inside_line(&ic.center, &contour[i].p, &contour[next_i].p)
                    && (mp - ic.center).norm_squared() <= ic.r * ic.r
                    && let (Some(ip1), Some(ip2)) = (
                        intersect_segment_with_circle(ic, &mp, &contour[i].p),
                        intersect_segment_with_circle(ic, &mp, &contour[next_i].p),
                    )
                {
                    let right_id = contour[i].right_id;
                    let left_id = contour[next_i].left_id;
                    let insert_pos = if i + 1 < contour.len() {
                        i + 1
                    } else {
                        contour.len()
                    };
                    contour.insert(insert_pos, ContourPoint::new(ip1, a_id, right_id));
                    contour.insert(insert_pos + 1, ContourPoint::new(ip2, left_id, a_id));
                    insertions += 2;
                    i += 2;
                }
            } else if pr1_indicator == 1 {
                // First outside, second inside
                if let Some(ip) =
                    intersect_segment_with_circle(ic, &contour[next_i].p, &contour[i].p)
                {
                    let right_id = contour[i].right_id;
                    let insert_pos = if i + 1 < contour.len() {
                        i + 1
                    } else {
                        contour.len()
                    };
                    contour.insert(insert_pos, ContourPoint::new(ip, a_id, right_id));
                    insertions += 1;
                    i += 1;
                } else {
                    contour[next_i].left_id = a_id;
                    contour[next_i].right_id = contour[i].right_id;
                }
            } else {
                // First inside, second outside
                if let Some(ip) =
                    intersect_segment_with_circle(ic, &contour[i].p, &contour[next_i].p)
                {
                    let left_id = contour[next_i].left_id;
                    let insert_pos = if i + 1 < contour.len() {
                        i + 1
                    } else {
                        contour.len()
                    };
                    contour.insert(insert_pos, ContourPoint::new(ip, left_id, a_id));
                    insertions += 1;
                    i += 1;
                } else {
                    contour[i].left_id = contour[next_i].left_id;
                    contour[i].right_id = a_id;
                }
            }
        }
        i += 1;
    }

    if insertions == 0 {
        contour.clear();
        return;
    }

    // Remove outside points
    contour.retain(|cp| cp.indicator != 1);

    if contour.len() < 2 {
        contour.clear();
        return;
    }

    // Calculate arc angles
    for i in 0..contour.len() {
        let next_i = (i + 1) % contour.len();
        if contour[i].right_id == a_id && contour[next_i].left_id == a_id {
            let angle = directed_angle(
                &ic.center,
                &contour[i].p,
                &contour[next_i].p,
                &(ic.center + axis),
            );
            contour[i].angle = angle;
            *sum_angles += angle;
            contour[i].indicator = 2;
            // Note: we modify next point's indicator but can't do it here due to borrow
        }
    }

    // Mark end points of arcs
    for i in 0..contour.len() {
        let prev_i = if i > 0 { i - 1 } else { contour.len() - 1 };
        if contour[prev_i].indicator == 2 {
            contour[i].indicator = 3;
        }
    }

    // If sum of arc angles >= 2π, the contour is a full circle
    if float_cmp::ge(*sum_angles, TAU)
        || (contour.len() > 2 && float_cmp::eq(*sum_angles, TAU))
    {
        *sum_angles = TAU;
        contour.clear();
    }
}

/// Calculate area of the contour
fn calculate_contour_area(contour: &Contour, ic: &Sphere, barycenter: &mut Point3<f64>) -> f64 {
    // Compute barycenter
    let mut sum = Vector3::zeros();
    for cp in contour {
        sum += cp.p.coords;
    }
    *barycenter = Point3::from(sum / contour.len() as f64);

    let mut area = 0.0;
    for i in 0..contour.len() {
        let next_i = (i + 1) % contour.len();
        area += triangle_area(barycenter, &contour[i].p, &contour[next_i].p);

        if contour[i].angle > 0.0 {
            // Add circular segment area
            area += ic.r * ic.r * (contour[i].angle - contour[i].angle.sin()) * 0.5;
        }
    }

    area
}

/// Calculate solid angle contribution
fn calculate_solid_angle(a: &Sphere, b: &Sphere, ic: &Sphere, contour: &Contour) -> f64 {
    let mut turn_angle = 0.0;

    if contour.is_empty() {
        // Full circle case
        turn_angle = TAU * (ic.center - a.center).norm() / a.r;
    } else {
        for i in 0..contour.len() {
            let prev_i = if i > 0 { i - 1 } else { contour.len() - 1 };
            let next_i = (i + 1) % contour.len();

            let pr0 = &contour[prev_i];
            let pr1 = &contour[i];
            let pr2 = &contour[next_i];

            if pr0.angle > 0.0 {
                let mut d = (b.center - a.center).cross(&(pr1.p - ic.center));
                let should_flip = (pr0.angle < PI && d.dot(&(pr0.p - pr1.p)) < 0.0)
                    || (pr0.angle > PI && d.dot(&(pr0.p - pr1.p)) > 0.0);
                if should_flip {
                    d = -d;
                }
                turn_angle += PI - min_dihedral_angle(&a.center, &pr1.p, &(pr1.p + d), &pr2.p);
            } else if pr1.angle > 0.0 {
                let mut d = (b.center - a.center).cross(&(pr1.p - ic.center));
                let should_flip = (pr1.angle < PI && d.dot(&(pr2.p - pr1.p)) < 0.0)
                    || (pr1.angle > PI && d.dot(&(pr2.p - pr1.p)) > 0.0);
                if should_flip {
                    d = -d;
                }
                turn_angle += PI - min_dihedral_angle(&a.center, &pr1.p, &pr0.p, &(pr1.p + d));
                turn_angle += pr1.angle * ((ic.center - a.center).norm() / a.r);
            } else {
                turn_angle += PI - min_dihedral_angle(&a.center, &pr1.p, &pr0.p, &pr2.p);
            }
        }
    }

    let mut solid_angle = TAU - turn_angle;

    // Check if contact is on far side of sphere a
    let ic_to_a = ic.center - a.center;
    let ic_to_b = ic.center - b.center;
    if ic_to_a.dot(&ic_to_b) > 0.0
        && (a.center - ic.center).norm_squared() < (b.center - ic.center).norm_squared()
    {
        solid_angle = -solid_angle;
    }

    solid_angle
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_contact() {
        let spheres = vec![
            Sphere::from_coords(0.0, 0.0, 0.0, 1.5),
            Sphere::from_coords(2.0, 0.0, 0.0, 1.5),
        ];

        let neighbors = vec![ValuedId::new(0.5, 1)];
        let cd = construct_contact_descriptor(&spheres, 0, 1, &neighbors);

        assert!(cd.is_some());
        let cd = cd.unwrap();
        assert!(cd.area > 0.0);
        assert_eq!(cd.id_a, 0);
        assert_eq!(cd.id_b, 1);
    }

    #[test]
    fn test_no_contact_non_intersecting() {
        let spheres = vec![
            Sphere::from_coords(0.0, 0.0, 0.0, 1.0),
            Sphere::from_coords(5.0, 0.0, 0.0, 1.0),
        ];

        let neighbors = vec![];
        let cd = construct_contact_descriptor(&spheres, 0, 1, &neighbors);
        assert!(cd.is_none());
    }

    #[test]
    fn test_contact_with_neighbor() {
        let spheres = vec![
            Sphere::from_coords(0.0, 0.0, 0.0, 1.5),
            Sphere::from_coords(2.0, 0.0, 0.0, 1.5),
            Sphere::from_coords(1.0, 1.5, 0.0, 1.5), // Neighbor that cuts the contact
        ];

        let neighbors = vec![ValuedId::new(0.5, 1), ValuedId::new(0.8, 2)];
        let cd = construct_contact_descriptor(&spheres, 0, 1, &neighbors);

        assert!(cd.is_some());
        let cd = cd.unwrap();
        // Contact area should be reduced by neighbor
        assert!(cd.area > 0.0);
    }
}
