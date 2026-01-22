use nalgebra::{Point3, Vector3};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Input sphere defined by center coordinates and radius.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Ball {
    /// X coordinate of center.
    pub x: f64,
    /// Y coordinate of center.
    pub y: f64,
    /// Z coordinate of center.
    pub z: f64,
    /// Radius.
    pub r: f64,
}

impl Ball {
    /// Create a new ball with center (x, y, z) and radius r.
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64, r: f64) -> Self {
        Self { x, y, z, r }
    }
}

/// Internal sphere representation with nalgebra Point3.
#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    /// Center point of the sphere.
    pub center: Point3<f64>,
    /// Radius (may include probe radius).
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

/// Contact area between two neighboring spheres.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Contact {
    /// Index of first sphere (always less than `id_b`).
    pub id_a: usize,
    /// Index of second sphere.
    pub id_b: usize,
    /// Contact area between the two spheres.
    pub area: f64,
    /// Arc length of the contact boundary.
    pub arc_length: f64,
}

/// Voronoi cell properties for a sphere.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Cell {
    /// Index of the sphere this cell belongs to.
    pub index: usize,
    /// Solvent-accessible surface area.
    pub sas_area: f64,
    /// Volume of the Voronoi cell.
    pub volume: f64,
}

/// Sorted collision pair: distance to intersection circle center + sphere index.
#[derive(Debug, Clone, Copy)]
pub struct ValuedId {
    /// Distance from sphere center to intersection circle center.
    pub value: f64,
    /// Index of the colliding sphere.
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
        // Sort by distance first; fall back to index for NaN or ties to ensure stable ordering
        match self.value.partial_cmp(&other.value) {
            Some(std::cmp::Ordering::Equal) | None => self.index.cmp(&other.index),
            Some(ord) => ord,
        }
    }
}

/// Trait for accessing tessellation results (contacts and cells).
pub trait Results {
    /// Number of balls in the tessellation.
    fn num_balls(&self) -> usize;

    /// Access to computed cells.
    fn cells(&self) -> &[Cell];

    /// Number of contacts.
    fn num_contacts(&self) -> usize;

    /// Get all contacts.
    fn contacts(&self) -> Vec<Contact>;

    /// Get solvent-accessible surface area for each ball as a Vec.
    ///
    /// Returns a Vec of length `num_balls` where index `i` contains the
    /// SAS area for ball `i`. Balls without computed cells get 0.0.
    #[must_use]
    fn sas_areas(&self) -> Vec<f64> {
        let mut result = vec![0.0; self.num_balls()];
        for cell in self.cells() {
            result[cell.index] = cell.sas_area;
        }
        result
    }

    /// Get cell volumes for each ball as a Vec.
    ///
    /// Returns a Vec of length `num_balls` where index `i` contains the
    /// volume for ball `i`. Balls without computed cells get 0.0.
    #[must_use]
    fn volumes(&self) -> Vec<f64> {
        let mut result = vec![0.0; self.num_balls()];
        for cell in self.cells() {
            result[cell.index] = cell.volume;
        }
        result
    }

    /// Get total solvent-accessible surface area across all balls.
    #[must_use]
    fn total_sas_area(&self) -> f64 {
        self.cells().iter().map(|c| c.sas_area).sum()
    }

    /// Get total volume across all balls.
    #[must_use]
    fn total_volume(&self) -> f64 {
        self.cells().iter().map(|c| c.volume).sum()
    }

    /// Get total contact area.
    #[must_use]
    fn total_contact_area(&self) -> f64 {
        self.contacts().iter().map(|c| c.area).sum()
    }
}

/// Result of a tessellation computation.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TessellationResult {
    /// Number of input balls.
    pub num_balls: usize,
    /// Contact areas between neighboring spheres.
    pub contacts: Vec<Contact>,
    /// Voronoi cell properties for each sphere.
    pub cells: Vec<Cell>,
}

impl Results for TessellationResult {
    fn num_balls(&self) -> usize {
        self.num_balls
    }

    fn cells(&self) -> &[Cell] {
        &self.cells
    }

    fn num_contacts(&self) -> usize {
        self.contacts.len()
    }

    fn contacts(&self) -> Vec<Contact> {
        self.contacts.clone()
    }
}

/// Periodic boundary conditions defined by three lattice vectors.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PeriodicBox {
    /// First lattice vector.
    pub shift_a: Vector3<f64>,
    /// Second lattice vector.
    pub shift_b: Vector3<f64>,
    /// Third lattice vector.
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

    /// The 26 periodic shift combinations (3x3x3 grid excluding origin).
    #[rustfmt::skip]
    pub const NEIGHBOR_SHIFTS: [(i32, i32, i32); 26] = [
        (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
        (-1,  0, -1), (-1,  0, 0), (-1,  0, 1),
        (-1,  1, -1), (-1,  1, 0), (-1,  1, 1),
        ( 0, -1, -1), ( 0, -1, 0), ( 0, -1, 1),
        ( 0,  0, -1),             ( 0,  0, 1),
        ( 0,  1, -1), ( 0,  1, 0), ( 0,  1, 1),
        ( 1, -1, -1), ( 1, -1, 0), ( 1, -1, 1),
        ( 1,  0, -1), ( 1,  0, 0), ( 1,  0, 1),
        ( 1,  1, -1), ( 1,  1, 0), ( 1,  1, 1),
    ];

    /// Generate all 27 periodic copies of spheres (original + 26 shifts).
    pub(crate) fn populate_periodic_spheres(&self, spheres: &[Sphere]) -> Vec<Sphere> {
        let n = spheres.len();
        let mut result = Vec::with_capacity(n * 27);
        result.extend_from_slice(spheres);

        for (sx, sy, sz) in Self::NEIGHBOR_SHIFTS {
            for s in spheres {
                result.push(self.shift_sphere(s, f64::from(sx), f64::from(sy), f64::from(sz)));
            }
        }
        result
    }
}

/// Internal contact descriptor summary (matches C++ `ContactDescriptorSummary`).
#[derive(Debug, Clone, Default)]
pub struct ContactDescriptorSummary {
    /// Contact area between the two spheres.
    pub area: f64,
    /// Arc length of the contact boundary.
    pub arc_length: f64,
    /// Solid angle contribution from sphere A's perspective.
    pub solid_angle_a: f64,
    /// Solid angle contribution from sphere B's perspective.
    pub solid_angle_b: f64,
    /// Pyramid volume contribution from sphere A's perspective.
    pub pyramid_volume_a: f64,
    /// Pyramid volume contribution from sphere B's perspective.
    pub pyramid_volume_b: f64,
    /// Distance between sphere centers.
    #[allow(dead_code)]
    pub distance: f64,
    /// Index of the first sphere.
    pub id_a: usize,
    /// Index of the second sphere.
    pub id_b: usize,
}

impl ContactDescriptorSummary {
    /// Ensure `id_a` < `id_b`, swapping related values if needed.
    /// Note: `solid_angle` and `pyramid_volume` are perspective-dependent (from sphere A vs B),
    /// so they must be swapped. Distance is symmetric and unchanged.
    pub const fn ensure_ids_ordered(&mut self) {
        if self.id_a > self.id_b {
            std::mem::swap(&mut self.id_a, &mut self.id_b);
            std::mem::swap(&mut self.solid_angle_a, &mut self.solid_angle_b);
            std::mem::swap(&mut self.pyramid_volume_a, &mut self.pyramid_volume_b);
        }
    }
}

/// Processing stage for cell contact summaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CellStage {
    /// Initial state, no contacts added yet.
    #[default]
    Init,
    /// Contacts have been added, ready for SAS computation.
    ContactsAdded,
    /// SAS area and volume have been computed.
    SasComputed,
}

/// Cell contact summary for computing SAS area and volume.
#[derive(Debug, Clone)]
pub struct CellContactSummary {
    /// Sphere index this summary belongs to.
    pub id: usize,
    /// Total contact area with neighbors.
    pub area: f64,
    /// Total arc length of contact boundaries.
    pub arc_length: f64,
    /// Sum of positive solid angle contributions.
    pub explained_solid_angle_positive: f64,
    /// Sum of negative solid angle contributions (stored as positive).
    pub explained_solid_angle_negative: f64,
    /// Sum of positive pyramid volume contributions.
    pub explained_pyramid_volume_positive: f64,
    /// Sum of negative pyramid volume contributions (stored as positive).
    pub explained_pyramid_volume_negative: f64,
    /// Computed solvent-accessible surface area.
    pub sas_area: f64,
    /// Computed volume inside the SAS.
    pub sas_inside_volume: f64,
    /// Number of contacts added to this summary.
    pub count: usize,
    /// Processing stage.
    pub stage: CellStage,
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
            stage: CellStage::Init,
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

            // Accumulate positive/negative separately for numerical stability
            // and to determine dominant orientation later in compute_sas
            self.explained_solid_angle_positive += solid_angle.max(0.0);
            self.explained_solid_angle_negative -= solid_angle.min(0.0);
            self.explained_pyramid_volume_positive += pyramid_volume.max(0.0);
            self.explained_pyramid_volume_negative -= pyramid_volume.min(0.0);
            self.stage = CellStage::ContactsAdded;
        }
    }

    #[allow(dead_code)]
    pub fn add_with_id(&mut self, new_id: usize, cds: &ContactDescriptorSummary) {
        if cds.area > 0.0 {
            if self.stage == CellStage::Init {
                self.id = new_id;
            }
            self.add(cds);
        }
    }

    pub fn compute_sas(&mut self, r: f64) {
        use std::f64::consts::PI;

        if self.stage != CellStage::ContactsAdded {
            return;
        }

        self.sas_area = 0.0;
        self.sas_inside_volume = 0.0;

        let diff =
            (self.explained_solid_angle_positive - self.explained_solid_angle_negative).abs();
        if self.arc_length > 0.0 && diff > 1e-10 {
            // Positive dominance: sphere mostly exposed, SAS = full sphere minus covered part
            // Negative dominance: sphere mostly buried, SAS = just the exposed segment
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
        self.stage = CellStage::SasComputed;
    }

    /// Compute SAS for a detached (non-contacting) sphere
    pub fn compute_sas_detached(&mut self, new_id: usize, r: f64) {
        use std::f64::consts::PI;

        if self.stage == CellStage::Init {
            self.id = new_id;
            self.sas_area = 4.0 * PI * r * r;
            self.sas_inside_volume = self.sas_area * r / 3.0;
            self.stage = CellStage::SasComputed;
        }
    }
}
