//! Internal sphere state management for updateable tessellation.
//!
//! Manages sphere positions, periodic images, exclusion statuses, and collision detection.

use rayon::prelude::*;

use crate::geometry::sphere_equals_sphere;
use crate::spheres_searcher::SpheresSearcher;
use crate::types::{PeriodicBox, Sphere, ValuedId};

/// Result of an update operation
pub struct UpdateResult {
    /// IDs of spheres that were changed
    pub changed_ids: Vec<usize>,
    /// IDs of spheres affected (changed + their neighbors)
    pub affected_ids: Vec<usize>,
    /// Whether a full reinit was performed
    pub was_full_reinit: bool,
}

/// Manages sphere state including periodic images and collision detection.
pub struct SpheresContainer {
    spheres: Vec<Sphere>,
    periodic_box: Option<PeriodicBox>,
    /// Spheres including periodic copies (27x for periodic case)
    populated_spheres: Vec<Sphere>,
    /// Whether each populated sphere is excluded (contained within another).
    exclusion_statuses: Vec<bool>,
    /// Collision lists per input sphere
    colliding_ids: Vec<Vec<ValuedId>>,
    total_collisions: usize,
    searcher: Option<SpheresSearcher>,
}

impl SpheresContainer {
    pub const fn new() -> Self {
        Self {
            spheres: Vec::new(),
            periodic_box: None,
            populated_spheres: Vec::new(),
            exclusion_statuses: Vec::new(),
            colliding_ids: Vec::new(),
            total_collisions: 0,
            searcher: None,
        }
    }

    /// Initialize container with spheres and optional periodic box.
    pub fn init(&mut self, spheres: Vec<Sphere>, periodic_box: Option<PeriodicBox>) {
        self.spheres = spheres;
        self.periodic_box = periodic_box;

        self.populate_spheres();
        self.exclusion_statuses = vec![false; self.populated_spheres.len()];
        self.searcher = Some(SpheresSearcher::new(self.populated_spheres.clone()));

        self.detect_all_collisions();
    }

    /// Update spheres and return affected IDs, or None if full reinit was performed.
    ///
    /// If `changed_ids` is provided, only those spheres are checked for changes.
    /// Otherwise, all spheres are compared to detect changes.
    #[allow(clippy::too_many_lines, clippy::option_if_let_else)]
    pub fn update(
        &mut self,
        new_spheres: &[Sphere],
        changed_ids: Option<&[usize]>,
    ) -> Option<UpdateResult> {
        if new_spheres.len() != self.spheres.len() {
            self.init(new_spheres.to_vec(), self.periodic_box);
            return Some(UpdateResult {
                changed_ids: Vec::new(),
                affected_ids: Vec::new(),
                was_full_reinit: true,
            });
        }

        let threshold = self.size_threshold_for_full_reinit();

        // Determine which spheres changed
        let changed: Vec<usize> = match changed_ids {
            Some(ids) => ids
                .iter()
                .filter(|&&id| {
                    id < self.spheres.len()
                        && !sphere_equals_sphere(&new_spheres[id], &self.spheres[id])
                })
                .copied()
                .collect(),
            None => (0..new_spheres.len())
                .filter(|&i| !sphere_equals_sphere(&new_spheres[i], &self.spheres[i]))
                .take(threshold + 1)
                .collect(),
        };

        if changed.is_empty() {
            return None;
        }

        // Full reinit is faster than incremental when many spheres change
        if changed.len() > threshold {
            self.init(new_spheres.to_vec(), self.periodic_box);
            return Some(UpdateResult {
                changed_ids: Vec::new(),
                affected_ids: Vec::new(),
                was_full_reinit: true,
            });
        }

        // Collect affected IDs (changed + their current neighbors)
        let mut affected: Vec<usize> = changed.clone();
        affected.sort_unstable();

        for &sphere_id in &changed {
            for neighbor in &self.colliding_ids[sphere_id] {
                let canonical_id = neighbor.index % self.spheres.len();
                if let Err(pos) = affected.binary_search(&canonical_id) {
                    if affected.len() < threshold {
                        affected.insert(pos, canonical_id);
                    } else {
                        self.init(new_spheres.to_vec(), self.periodic_box);
                        return Some(UpdateResult {
                            changed_ids: Vec::new(),
                            affected_ids: Vec::new(),
                            was_full_reinit: true,
                        });
                    }
                }
            }
        }

        // Update sphere positions
        let mut changed_populated: Vec<usize> = Vec::new();
        for &sphere_id in &changed {
            self.spheres[sphere_id] = new_spheres[sphere_id];
            self.update_sphere_periodic_instances(sphere_id, &mut changed_populated);
        }

        // Update spatial index
        if let Some(ref mut searcher) = self.searcher {
            searcher.update(&self.populated_spheres, &changed_populated);
        }

        // Redetect collisions for affected spheres
        self.update_collisions_for_spheres(&affected);

        // Find new neighbors that became affected
        let mut more_affected: Vec<usize> = Vec::new();
        for &sphere_id in &changed {
            for neighbor in &self.colliding_ids[sphere_id] {
                let canonical_id = neighbor.index % self.spheres.len();
                if affected.binary_search(&canonical_id).is_err()
                    && let Err(pos) = more_affected.binary_search(&canonical_id)
                {
                    more_affected.insert(pos, canonical_id);
                }
            }
        }

        if !more_affected.is_empty() {
            self.update_collisions_for_spheres(&more_affected);

            affected.extend(more_affected);
            affected.sort_unstable();
            affected.dedup();
        }

        self.recount_collisions();

        Some(UpdateResult {
            changed_ids: changed,
            affected_ids: affected,
            was_full_reinit: false,
        })
    }

    /// Set exclusion status for a sphere.
    /// Returns affected sphere IDs if successful, None if already in requested state.
    pub fn set_exclusion(&mut self, id: usize, excluded: bool) -> Option<Vec<usize>> {
        if id >= self.spheres.len() || id >= self.exclusion_statuses.len() {
            return None;
        }

        if self.exclusion_statuses[id] == excluded {
            return None;
        }

        self.exclusion_statuses[id] = excluded;
        self.set_exclusion_status_periodic_instances(id);

        // Collect affected IDs
        let mut affected = vec![id];
        for neighbor in &self.colliding_ids[id] {
            let canonical_id = neighbor.index % self.spheres.len();
            if let Err(pos) = affected.binary_search(&canonical_id) {
                affected.insert(pos, canonical_id);
            }
        }

        Some(affected)
    }

    #[inline]
    pub fn spheres(&self) -> &[Sphere] {
        &self.spheres
    }

    #[inline]
    pub fn populated_spheres(&self) -> &[Sphere] {
        &self.populated_spheres
    }

    #[inline]
    pub fn colliding_ids(&self, id: usize) -> &[ValuedId] {
        &self.colliding_ids[id]
    }

    #[inline]
    pub fn all_colliding_ids(&self) -> &[Vec<ValuedId>] {
        &self.colliding_ids
    }

    #[inline]
    pub fn exclusion_statuses(&self) -> &[bool] {
        &self.exclusion_statuses
    }

    #[inline]
    pub fn is_excluded(&self, id: usize) -> bool {
        id < self.exclusion_statuses.len() && self.exclusion_statuses[id]
    }

    #[inline]
    pub const fn periodic_box(&self) -> Option<&PeriodicBox> {
        self.periodic_box.as_ref()
    }

    /// Clone this container's state for backup purposes.
    pub fn clone_state(&self) -> Self {
        Self {
            spheres: self.spheres.clone(),
            periodic_box: self.periodic_box,
            populated_spheres: self.populated_spheres.clone(),
            exclusion_statuses: self.exclusion_statuses.clone(),
            colliding_ids: self.colliding_ids.clone(),
            total_collisions: self.total_collisions,
            searcher: self
                .searcher
                .as_ref()
                .map(super::spheres_searcher::SpheresSearcher::clone_for_backup),
        }
    }

    /// Restore state from a backup for a subset of sphere IDs.
    pub fn restore_from(&mut self, backup: &Self, affected_ids: &[usize]) {
        if affected_ids.is_empty()
            || self.spheres.len() != backup.spheres.len()
            || affected_ids.len() > self.size_threshold_for_full_reinit()
        {
            *self = backup.clone_state();
            return;
        }

        let n = self.spheres.len();

        for &id in affected_ids {
            if id >= n {
                *self = backup.clone_state();
                return;
            }
        }

        // Restore only affected spheres
        for &id in affected_ids {
            self.spheres[id] = backup.spheres[id];
            self.colliding_ids[id].clone_from(&backup.colliding_ids[id]);
            self.exclusion_statuses[id] = backup.exclusion_statuses[id];

            if self.periodic_box.is_some() {
                for m in 1..27 {
                    let shifted_id = m * n + id;
                    if shifted_id < self.populated_spheres.len() {
                        self.populated_spheres[shifted_id] = backup.populated_spheres[shifted_id];
                        self.exclusion_statuses[shifted_id] = backup.exclusion_statuses[shifted_id];
                    }
                }
            }
            self.populated_spheres[id] = backup.populated_spheres[id];
        }

        self.total_collisions = backup.total_collisions;
        if let Some(ref backup_searcher) = backup.searcher {
            self.searcher = Some(backup_searcher.clone_for_backup());
        }
    }

    fn size_threshold_for_full_reinit(&self) -> usize {
        // At least 10 to avoid triggering full reinit on small datasets
        (self.spheres.len() / 2).max(10)
    }

    /// Create populated spheres including periodic images.
    fn populate_spheres(&mut self) {
        if let Some(ref pbox) = self.periodic_box {
            self.populated_spheres = pbox.populate_periodic_spheres(&self.spheres);
        } else {
            self.populated_spheres = self.spheres.clone();
        }
    }

    /// Update periodic instances for a single sphere.
    fn update_sphere_periodic_instances(&mut self, id: usize, changed_ids: &mut Vec<usize>) {
        if id >= self.spheres.len() {
            return;
        }

        let n = self.spheres.len();
        self.populated_spheres[id] = self.spheres[id];
        changed_ids.push(id);

        if let Some(ref pbox) = self.periodic_box {
            for (g, (sx, sy, sz)) in PeriodicBox::NEIGHBOR_SHIFTS.iter().enumerate() {
                let shifted_id = (g + 1) * n + id;
                self.populated_spheres[shifted_id] = pbox.shift_sphere(
                    &self.spheres[id],
                    f64::from(*sx),
                    f64::from(*sy),
                    f64::from(*sz),
                );
                changed_ids.push(shifted_id);
            }
        }
    }

    /// Set exclusion status for periodic instances.
    fn set_exclusion_status_periodic_instances(&mut self, id: usize) {
        if self.periodic_box.is_none() {
            return;
        }

        let n = self.spheres.len();
        if self.exclusion_statuses.len() != n * 27 {
            return;
        }

        let status = self.exclusion_statuses[id];
        for m in 1..27 {
            self.exclusion_statuses[m * n + id] = status;
        }
    }

    /// Detect collisions for all spheres.
    fn detect_all_collisions(&mut self) {
        let n = self.spheres.len();
        self.colliding_ids = vec![Vec::new(); n];

        if let Some(ref searcher) = self.searcher {
            let results: Vec<_> = (0..n)
                .into_par_iter()
                .map(|id| {
                    let result = searcher.find_colliding_ids(id, true);
                    (id, result.colliding_ids, result.excluded)
                })
                .collect();

            for (id, collisions, excluded) in results {
                self.colliding_ids[id] = collisions;
                self.exclusion_statuses[id] = excluded;
            }

            if self.periodic_box.is_some() {
                for i in 0..n {
                    self.set_exclusion_status_periodic_instances(i);
                }
            }
        }

        self.recount_collisions();
    }

    /// Update collisions for a subset of spheres.
    fn update_collisions_for_spheres(&mut self, sphere_ids: &[usize]) {
        if let Some(ref searcher) = self.searcher {
            let results: Vec<_> = sphere_ids
                .par_iter()
                .map(|&id| {
                    let result = searcher.find_colliding_ids(id, true);
                    (id, result.colliding_ids, result.excluded)
                })
                .collect();

            for (id, collisions, excluded) in results {
                self.colliding_ids[id] = collisions;
                self.exclusion_statuses[id] = excluded;
            }

            if self.periodic_box.is_some() {
                for &id in sphere_ids {
                    self.set_exclusion_status_periodic_instances(id);
                }
            }
        }
    }

    /// Recount total collisions.
    fn recount_collisions(&mut self) {
        self.total_collisions = self
            .colliding_ids
            .iter()
            .map(std::vec::Vec::len)
            .sum::<usize>()
            / 2;
    }
}
