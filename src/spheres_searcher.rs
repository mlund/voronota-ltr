use crate::geometry::{
    distance_to_intersection_circle_center, sphere_contains_sphere, sphere_equals_sphere,
    sphere_intersects_sphere,
};
use crate::types::{Sphere, ValuedId};

/// Grid coordinates for spatial indexing
#[derive(Debug, Clone, Copy, Default)]
struct GridPoint {
    x: i32,
    y: i32,
    z: i32,
}

impl GridPoint {
    #[allow(clippy::cast_possible_truncation)]
    fn from_sphere(s: &Sphere, box_size: f64) -> Self {
        Self {
            x: (s.center.x / box_size).floor() as i32,
            y: (s.center.y / box_size).floor() as i32,
            z: (s.center.z / box_size).floor() as i32,
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_sphere_with_offset(s: &Sphere, box_size: f64, offset: &Self) -> Self {
        Self {
            x: (s.center.x / box_size).floor() as i32 - offset.x,
            y: (s.center.y / box_size).floor() as i32 - offset.y,
            z: (s.center.z / box_size).floor() as i32 - offset.z,
        }
    }

    #[allow(clippy::cast_sign_loss)]
    const fn index(&self, grid_size: &Self) -> Option<usize> {
        if self.x >= 0
            && self.y >= 0
            && self.z >= 0
            && self.x < grid_size.x
            && self.y < grid_size.y
            && self.z < grid_size.z
        {
            Some((self.z * grid_size.x * grid_size.y + self.y * grid_size.x + self.x) as usize)
        } else {
            None
        }
    }
}

/// Grid parameters for spatial indexing
#[derive(Debug, Clone)]
struct GridParameters {
    grid_offset: GridPoint,
    grid_size: GridPoint,
    box_size: f64,
}

impl GridParameters {
    fn new(spheres: &[Sphere]) -> Self {
        let mut params = Self {
            grid_offset: GridPoint::default(),
            grid_size: GridPoint { x: 1, y: 1, z: 1 },
            box_size: 1.0,
        };

        if spheres.is_empty() {
            return params;
        }

        // Box size = max(2*r + margin): ensures overlapping spheres are in adjacent cells
        for s in spheres {
            params.box_size = params.box_size.max(s.r.mul_add(2.0, 0.25));
        }

        // Compute grid bounds
        let padding = 1;
        for (i, s) in spheres.iter().enumerate() {
            let gp = GridPoint::from_sphere(s, params.box_size);
            if i == 0 {
                params.grid_offset = gp;
                params.grid_size = gp;
            } else {
                params.grid_offset.x = params.grid_offset.x.min(gp.x - padding);
                params.grid_offset.y = params.grid_offset.y.min(gp.y - padding);
                params.grid_offset.z = params.grid_offset.z.min(gp.z - padding);
                params.grid_size.x = params.grid_size.x.max(gp.x + padding);
                params.grid_size.y = params.grid_size.y.max(gp.y + padding);
                params.grid_size.z = params.grid_size.z.max(gp.z + padding);
            }
        }

        params.grid_size.x = params.grid_size.x - params.grid_offset.x + 1;
        params.grid_size.y = params.grid_size.y - params.grid_offset.y + 1;
        params.grid_size.z = params.grid_size.z - params.grid_offset.z + 1;

        params
    }
}

/// Result of collision search
pub struct CollisionResult {
    pub colliding_ids: Vec<ValuedId>,
    /// Whether this sphere is excluded (contained within another).
    pub excluded: bool,
}

/// Grid-based spatial index for finding sphere collisions
pub struct SpheresSearcher {
    spheres: Vec<Sphere>,
    grid_params: GridParameters,
    /// Map from grid cell index to box index (-1 = empty)
    map_of_boxes: Vec<i32>,
    /// Each box contains sphere indices
    boxes: Vec<Vec<usize>>,
}

impl SpheresSearcher {
    pub fn new(spheres: Vec<Sphere>) -> Self {
        let grid_params = GridParameters::new(&spheres);
        let mut searcher = Self {
            spheres,
            grid_params,
            map_of_boxes: Vec::new(),
            boxes: Vec::new(),
        };
        searcher.init_boxes();
        searcher
    }

    pub fn spheres(&self) -> &[Sphere] {
        &self.spheres
    }

    /// Update sphere positions and rebuild spatial index for changed spheres.
    pub fn update(&mut self, spheres: &[Sphere], changed_ids: &[usize]) {
        // Update sphere positions
        for &id in changed_ids {
            if id < self.spheres.len() && id < spheres.len() {
                self.spheres[id] = spheres[id];
            }
        }

        // Check if grid parameters changed significantly
        let new_params = GridParameters::new(&self.spheres);
        if (new_params.box_size - self.grid_params.box_size).abs() > 0.01
            || new_params.grid_size.x != self.grid_params.grid_size.x
            || new_params.grid_size.y != self.grid_params.grid_size.y
            || new_params.grid_size.z != self.grid_params.grid_size.z
        {
            // Full rebuild needed
            self.grid_params = new_params;
            self.init_boxes();
        } else {
            // Incremental update: remove and re-add changed spheres
            for &id in changed_ids {
                if id < self.spheres.len() {
                    self.remove_sphere_from_grid(id);
                }
            }
            for &id in changed_ids {
                if id < self.spheres.len() {
                    self.add_sphere_to_grid(id);
                }
            }
        }
    }

    /// Clone for backup purposes.
    pub fn clone_for_backup(&self) -> Self {
        Self {
            spheres: self.spheres.clone(),
            grid_params: self.grid_params.clone(),
            map_of_boxes: self.map_of_boxes.clone(),
            boxes: self.boxes.clone(),
        }
    }

    #[allow(clippy::cast_sign_loss)]
    fn remove_sphere_from_grid(&mut self, sphere_id: usize) {
        let sphere = &self.spheres[sphere_id];
        let gp = GridPoint::from_sphere_with_offset(
            sphere,
            self.grid_params.box_size,
            &self.grid_params.grid_offset,
        );
        if let Some(index) = gp.index(&self.grid_params.grid_size) {
            let box_id = self.map_of_boxes[index];
            if box_id >= 0 {
                let box_vec = &mut self.boxes[box_id as usize];
                if let Some(pos) = box_vec.iter().position(|&id| id == sphere_id) {
                    box_vec.swap_remove(pos);
                }
            }
        }
    }

    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap
    )]
    fn add_sphere_to_grid(&mut self, sphere_id: usize) {
        let sphere = &self.spheres[sphere_id];
        let gp = GridPoint::from_sphere_with_offset(
            sphere,
            self.grid_params.box_size,
            &self.grid_params.grid_offset,
        );
        if let Some(index) = gp.index(&self.grid_params.grid_size) {
            let box_id = self.map_of_boxes[index];
            if box_id < 0 {
                self.map_of_boxes[index] = self.boxes.len() as i32;
                self.boxes.push(vec![sphere_id]);
            } else {
                self.boxes[box_id as usize].push(sphere_id);
            }
        }
    }

    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap
    )]
    fn init_boxes(&mut self) {
        let total_cells = (self.grid_params.grid_size.x
            * self.grid_params.grid_size.y
            * self.grid_params.grid_size.z) as usize;

        self.map_of_boxes = vec![-1; total_cells];
        self.boxes.clear();

        for (i, sphere) in self.spheres.iter().enumerate() {
            let gp = GridPoint::from_sphere_with_offset(
                sphere,
                self.grid_params.box_size,
                &self.grid_params.grid_offset,
            );
            if let Some(index) = gp.index(&self.grid_params.grid_size) {
                let box_id = self.map_of_boxes[index];
                if box_id < 0 {
                    self.map_of_boxes[index] = self.boxes.len() as i32;
                    self.boxes.push(vec![i]);
                } else {
                    self.boxes[box_id as usize].push(i);
                }
            }
        }
    }

    /// Find all spheres that collide with sphere at `central_id`
    /// Returns sorted by distance to intersection circle center
    #[allow(clippy::cast_sign_loss)]
    pub fn find_colliding_ids(&self, central_id: usize, discard_hidden: bool) -> CollisionResult {
        let mut result = CollisionResult {
            colliding_ids: Vec::new(),
            excluded: false,
        };

        if central_id >= self.spheres.len() {
            return result;
        }

        let central_sphere = &self.spheres[central_id];
        let gp = GridPoint::from_sphere_with_offset(
            central_sphere,
            self.grid_params.box_size,
            &self.grid_params.grid_offset,
        );

        // Search 27-cell neighborhood (3x3x3) to catch all potential overlaps
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let neighbor = GridPoint {
                        x: gp.x + dx,
                        y: gp.y + dy,
                        z: gp.z + dz,
                    };

                    if let Some(index) = neighbor.index(&self.grid_params.grid_size) {
                        let box_id = self.map_of_boxes[index];
                        if box_id >= 0 {
                            for &id in &self.boxes[box_id as usize] {
                                if id == central_id {
                                    continue;
                                }

                                let candidate = &self.spheres[id];
                                if !sphere_intersects_sphere(central_sphere, candidate) {
                                    continue;
                                }

                                // Check if central sphere is hidden by candidate
                                if discard_hidden
                                    && sphere_contains_sphere(candidate, central_sphere)
                                    && (!sphere_equals_sphere(candidate, central_sphere)
                                        || central_id > id)
                                {
                                    result.colliding_ids.clear();
                                    result.excluded = true;
                                    return result;
                                }

                                // Skip candidates that are contained by central sphere
                                if discard_hidden
                                    && sphere_contains_sphere(central_sphere, candidate)
                                {
                                    continue;
                                }

                                let dist = distance_to_intersection_circle_center(
                                    central_sphere,
                                    candidate,
                                );
                                result.colliding_ids.push(ValuedId::new(dist, id));
                            }
                        }
                    }
                }
            }
        }

        result.colliding_ids.sort();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_searcher_basic() {
        let spheres = vec![
            Sphere::from_coords(0.0, 0.0, 0.0, 1.0),
            Sphere::from_coords(1.5, 0.0, 0.0, 1.0),
            Sphere::from_coords(5.0, 0.0, 0.0, 1.0),
        ];

        let searcher = SpheresSearcher::new(spheres);
        let result = searcher.find_colliding_ids(0, true);

        assert_eq!(result.colliding_ids.len(), 1);
        assert_eq!(result.colliding_ids[0].index, 1);
    }

    #[test]
    fn test_searcher_hidden() {
        let spheres = vec![
            Sphere::from_coords(0.0, 0.0, 0.0, 1.0),
            Sphere::from_coords(0.0, 0.0, 0.0, 2.0), // Contains sphere 0
        ];

        let searcher = SpheresSearcher::new(spheres);
        let result = searcher.find_colliding_ids(0, true);

        assert!(result.excluded);
        assert!(result.colliding_ids.is_empty());
    }
}
