use nalgebra::{Point3, Vector3};

/// Input ball (center + radius), user-facing type
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ball {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub r: f64,
}

impl Ball {
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64, r: f64) -> Self {
        Self { x, y, z, r }
    }
}

/// Internal sphere representation with nalgebra Point3
#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    pub center: Point3<f64>,
    pub r: f64,
}

impl Sphere {
    pub const fn new(center: Point3<f64>, r: f64) -> Self {
        Self { center, r }
    }

    pub const fn from_coords(x: f64, y: f64, z: f64, r: f64) -> Self {
        Self {
            center: Point3::new(x, y, z),
            r,
        }
    }

    /// Convert Ball to Sphere with optional probe radius added
    #[must_use]
    pub fn from_ball(ball: &Ball, probe: f64) -> Self {
        Self {
            center: Point3::new(ball.x, ball.y, ball.z),
            r: ball.r + probe,
        }
    }
}

/// Contact between two spheres
#[derive(Debug, Clone)]
pub struct Contact {
    pub id_a: usize,
    pub id_b: usize,
    pub area: f64,
    pub arc_length: f64,
}

/// Cell (Voronoi cell) around a sphere
#[derive(Debug, Clone)]
pub struct Cell {
    pub index: usize,
    pub sas_area: f64,
    pub volume: f64,
}

/// Sorted collision pair: distance to intersection circle center + sphere index
#[derive(Debug, Clone, Copy)]
pub struct ValuedId {
    pub value: f64,
    pub index: usize,
}

impl ValuedId {
    pub const fn new(value: f64, index: usize) -> Self {
        Self { value, index }
    }
}

impl PartialEq for ValuedId {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.index == other.index
    }
}

impl Eq for ValuedId {}

impl PartialOrd for ValuedId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ValuedId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other)
            .unwrap_or_else(|| self.index.cmp(&other.index))
    }
}

/// Tessellation result containing contacts and cells
#[derive(Debug, Clone, Default)]
pub struct TessellationResult {
    pub contacts: Vec<Contact>,
    pub cells: Vec<Cell>,
}

/// Periodic boundary box defined by three shift vectors
#[derive(Debug, Clone, Copy, Default)]
pub struct PeriodicBox {
    pub shift_a: Vector3<f64>,
    pub shift_b: Vector3<f64>,
    pub shift_c: Vector3<f64>,
}

impl PeriodicBox {
    /// Create from two corner points (axis-aligned box)
    #[must_use]
    pub const fn from_corners(min: (f64, f64, f64), max: (f64, f64, f64)) -> Self {
        Self {
            shift_a: Vector3::new(max.0 - min.0, 0.0, 0.0),
            shift_b: Vector3::new(0.0, max.1 - min.1, 0.0),
            shift_c: Vector3::new(0.0, 0.0, max.2 - min.2),
        }
    }

    /// Create from three shift direction vectors (for non-orthogonal boxes)
    #[must_use]
    pub const fn from_vectors(a: (f64, f64, f64), b: (f64, f64, f64), c: (f64, f64, f64)) -> Self {
        Self {
            shift_a: Vector3::new(a.0, a.1, a.2),
            shift_b: Vector3::new(b.0, b.1, b.2),
            shift_c: Vector3::new(c.0, c.1, c.2),
        }
    }

    /// Shift a sphere by weighted direction vectors
    pub(crate) fn shift_sphere(&self, s: &Sphere, wa: f64, wb: f64, wc: f64) -> Sphere {
        Sphere {
            center: Point3::new(
                self.shift_c.x.mul_add(
                    wc,
                    self.shift_b
                        .x
                        .mul_add(wb, self.shift_a.x.mul_add(wa, s.center.x)),
                ),
                self.shift_c.y.mul_add(
                    wc,
                    self.shift_b
                        .y
                        .mul_add(wb, self.shift_a.y.mul_add(wa, s.center.y)),
                ),
                self.shift_c.z.mul_add(
                    wc,
                    self.shift_b
                        .z
                        .mul_add(wb, self.shift_a.z.mul_add(wa, s.center.z)),
                ),
            ),
            r: s.r,
        }
    }
}

/// Internal contact descriptor summary (matches C++ `ContactDescriptorSummary`)
#[derive(Debug, Clone, Default)]
pub struct ContactDescriptorSummary {
    pub area: f64,
    pub arc_length: f64,
    pub solid_angle_a: f64,
    pub solid_angle_b: f64,
    pub pyramid_volume_a: f64,
    pub pyramid_volume_b: f64,
    #[allow(dead_code)]
    pub distance: f64,
    pub id_a: usize,
    pub id_b: usize,
}

impl ContactDescriptorSummary {
    /// Ensure `id_a` < `id_b`, swapping related values if needed
    pub const fn ensure_ids_ordered(&mut self) {
        if self.id_a > self.id_b {
            std::mem::swap(&mut self.id_a, &mut self.id_b);
            std::mem::swap(&mut self.solid_angle_a, &mut self.solid_angle_b);
            std::mem::swap(&mut self.pyramid_volume_a, &mut self.pyramid_volume_b);
        }
    }
}

/// Cell contact summary for computing SAS area and volume
#[derive(Debug, Clone)]
pub struct CellContactSummary {
    pub id: usize,
    pub area: f64,
    pub arc_length: f64,
    pub explained_solid_angle_positive: f64,
    pub explained_solid_angle_negative: f64,
    pub explained_pyramid_volume_positive: f64,
    pub explained_pyramid_volume_negative: f64,
    pub sas_area: f64,
    pub sas_inside_volume: f64,
    pub count: usize,
    pub stage: i32,
}

impl Default for CellContactSummary {
    fn default() -> Self {
        Self {
            id: 0,
            area: 0.0,
            arc_length: 0.0,
            explained_solid_angle_positive: 0.0,
            explained_solid_angle_negative: 0.0,
            explained_pyramid_volume_positive: 0.0,
            explained_pyramid_volume_negative: 0.0,
            sas_area: 0.0,
            sas_inside_volume: 0.0,
            count: 0,
            stage: 0,
        }
    }
}

impl CellContactSummary {
    pub fn add(&mut self, cds: &ContactDescriptorSummary) {
        if cds.area > 0.0 && (cds.id_a == self.id || cds.id_b == self.id) {
            self.count += 1;
            self.area += cds.area;
            self.arc_length += cds.arc_length;

            let (solid_angle, pyramid_volume) = if cds.id_a == self.id {
                (cds.solid_angle_a, cds.pyramid_volume_a)
            } else {
                (cds.solid_angle_b, cds.pyramid_volume_b)
            };

            self.explained_solid_angle_positive += solid_angle.max(0.0);
            self.explained_solid_angle_negative -= solid_angle.min(0.0);
            self.explained_pyramid_volume_positive += pyramid_volume.max(0.0);
            self.explained_pyramid_volume_negative -= pyramid_volume.min(0.0);
            self.stage = 1;
        }
    }

    #[allow(dead_code)]
    pub fn add_with_id(&mut self, new_id: usize, cds: &ContactDescriptorSummary) {
        if cds.area > 0.0 {
            if self.stage == 0 {
                self.id = new_id;
            }
            self.add(cds);
        }
    }

    pub fn compute_sas(&mut self, r: f64) {
        use std::f64::consts::PI;

        if self.stage != 1 {
            return;
        }

        self.sas_area = 0.0;
        self.sas_inside_volume = 0.0;

        let diff =
            (self.explained_solid_angle_positive - self.explained_solid_angle_negative).abs();
        if self.arc_length > 0.0 && diff > 1e-10 {
            if self.explained_solid_angle_positive > self.explained_solid_angle_negative {
                let angle_diff =
                    self.explained_solid_angle_positive - self.explained_solid_angle_negative;
                self.sas_area = 4.0f64.mul_add(PI, -angle_diff.max(0.0)) * r * r;
            } else {
                let angle_diff =
                    self.explained_solid_angle_negative - self.explained_solid_angle_positive;
                self.sas_area = angle_diff.max(0.0) * r * r;
            }
            self.sas_inside_volume = (self.sas_area * r / 3.0)
                + self.explained_pyramid_volume_positive
                - self.explained_pyramid_volume_negative;

            // Sanity check: volume shouldn't exceed full sphere
            let full_sphere_vol = 4.0 / 3.0 * PI * r * r * r;
            if self.sas_inside_volume > full_sphere_vol {
                self.sas_area = 0.0;
                self.sas_inside_volume =
                    self.explained_pyramid_volume_positive - self.explained_pyramid_volume_negative;
            }
        } else {
            self.sas_inside_volume =
                self.explained_pyramid_volume_positive - self.explained_pyramid_volume_negative;
        }
        self.stage = 2;
    }

    /// Compute SAS for a detached (non-contacting) sphere
    pub fn compute_sas_detached(&mut self, new_id: usize, r: f64) {
        use std::f64::consts::PI;

        if self.stage == 0 {
            self.id = new_id;
            self.sas_area = 4.0 * PI * r * r;
            self.sas_inside_volume = self.sas_area * r / 3.0;
            self.stage = 2;
        }
    }
}
