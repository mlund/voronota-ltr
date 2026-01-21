//! Updateable radical tessellation for incremental updates.
//!
//! This module provides [`UpdateableTessellation`] which enables efficient
//! incremental updates when only a subset of spheres change.

use rayon::prelude::*;

use crate::contact::construct_contact_descriptor;
use crate::spheres_container::SpheresContainer;
use crate::types::{
    Ball, Cell, CellContactSummary, Contact, ContactDescriptorSummary, PeriodicBox, Sphere,
    TessellationResult,
};

/// Result of updateable tessellation with per-sphere contact storage.
#[derive(Debug, Clone, Default)]
pub struct UpdateableResult {
    /// Cell summaries for each sphere
    pub cells: Vec<Cell>,
    /// Contacts organized by sphere ID (internal representation)
    pub(crate) contacts_by_sphere: Vec<Vec<ContactDescriptorSummary>>,
}

impl UpdateableResult {
    const fn is_empty(&self) -> bool {
        self.cells.is_empty() || self.contacts_by_sphere.is_empty()
    }

    /// Get contacts involving a specific sphere.
    ///
    /// Returns an iterator over Contact structs for the given sphere ID.
    pub fn contacts_for_sphere(&self, sphere_id: usize) -> impl Iterator<Item = Contact> + '_ {
        self.contacts_by_sphere
            .get(sphere_id)
            .into_iter()
            .flatten()
            .map(|cds| Contact {
                id_a: cds.id_a,
                id_b: cds.id_b,
                area: cds.area,
                arc_length: cds.arc_length,
            })
    }

    /// Get number of spheres in the result.
    #[must_use]
    pub const fn num_spheres(&self) -> usize {
        self.contacts_by_sphere.len()
    }
}

/// Internal state for `UpdateableTessellation`.
struct State {
    container: SpheresContainer,
    result: UpdateableResult,
    cell_summaries: Vec<CellContactSummary>,
    changed_ids: Vec<usize>,
    affected_ids: Vec<usize>,
    was_full_reinit: bool,
}

impl State {
    fn new() -> Self {
        Self {
            container: SpheresContainer::new(),
            result: UpdateableResult::default(),
            cell_summaries: Vec::new(),
            changed_ids: Vec::new(),
            affected_ids: Vec::new(),
            was_full_reinit: true,
        }
    }
}

/// Updateable radical tessellation that supports incremental updates.
///
/// When only a subset of spheres change positions, the tessellation can be
/// updated more efficiently than computing from scratch.
///
/// # Example
///
/// ```
/// use voronotalt::{Ball, UpdateableTessellation};
///
/// let mut balls = vec![
///     Ball::new(0.0, 0.0, 0.0, 1.0),
///     Ball::new(2.0, 0.0, 0.0, 1.0),
///     Ball::new(4.0, 0.0, 0.0, 1.0),
/// ];
///
/// let mut tess = UpdateableTessellation::with_backup();
/// tess.init(&balls, 1.0, None);
///
/// // Move first ball
/// balls[0].x += 0.1;
/// tess.update_with_changed(&balls, &[0]);
///
/// // Get results
/// let summary = tess.summary();
/// println!("Total contacts: {}", summary.contacts.len());
/// ```
pub struct UpdateableTessellation {
    state: State,
    backup: Option<State>,
    backup_enabled: bool,
    probe: f64,
}

impl UpdateableTessellation {
    /// Create a new updateable tessellation without backup support.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: State::new(),
            backup: None,
            backup_enabled: false,
            probe: 0.0,
        }
    }

    /// Create a new updateable tessellation with backup support for restore.
    #[must_use]
    pub fn with_backup() -> Self {
        Self {
            state: State::new(),
            backup: None,
            backup_enabled: true,
            probe: 0.0,
        }
    }

    /// Initialize tessellation with spheres.
    ///
    /// Returns true if tessellation was computed successfully.
    pub fn init(&mut self, balls: &[Ball], probe: f64, periodic_box: Option<&PeriodicBox>) -> bool {
        self.probe = probe;
        self.prepare_for_update();

        let spheres: Vec<Sphere> = balls.iter().map(|b| Sphere::from_ball(b, probe)).collect();
        self.state.container.init(spheres, periodic_box.copied());

        self.state.was_full_reinit = true;
        self.state.changed_ids.clear();
        self.state.affected_ids.clear();

        self.compute_full_tessellation();

        !self.state.result.is_empty()
    }

    /// Update tessellation by detecting which spheres changed.
    ///
    /// Compares new positions with stored positions to find changes.
    /// Returns true if update was successful.
    pub fn update(&mut self, balls: &[Ball]) -> bool {
        self.update_internal(balls, None)
    }

    /// Update tessellation with explicit list of changed sphere IDs.
    ///
    /// More efficient than `update()` when caller knows which spheres changed.
    /// Returns true if update was successful.
    pub fn update_with_changed(&mut self, balls: &[Ball], changed_ids: &[usize]) -> bool {
        if changed_ids.is_empty() {
            return false;
        }
        self.update_internal(balls, Some(changed_ids))
    }

    /// Exclude or include a sphere without removing it.
    ///
    /// Excluded spheres don't participate in contact/cell calculations.
    /// Returns true if the exclusion state was changed.
    pub fn set_exclusion(&mut self, id: usize, excluded: bool) -> bool {
        if self.state.result.is_empty()
            || id >= self.state.result.contacts_by_sphere.len()
            || id >= self.state.container.exclusion_statuses().len()
        {
            return false;
        }

        let current_excluded = self.state.container.is_excluded(id);
        if current_excluded == excluded {
            return false;
        }

        self.prepare_for_update();

        let affected = if excluded {
            // When excluding: affected = sphere + its current neighbors
            let mut affected = vec![id];
            for contact in &self.state.result.contacts_by_sphere[id] {
                let neighbor_id = if contact.id_a == id {
                    contact.id_b
                } else {
                    contact.id_a
                };
                if let Err(pos) = affected.binary_search(&neighbor_id) {
                    affected.insert(pos, neighbor_id);
                }
            }
            self.state.container.set_exclusion(id, true);
            affected
        } else {
            // When including: get affected from container
            match self.state.container.set_exclusion(id, false) {
                Some(affected) => affected,
                None => return false,
            }
        };

        self.state.changed_ids = vec![id];
        self.state.affected_ids = affected;
        self.state.was_full_reinit = false;

        self.update_using_affected();

        true
    }

    /// Restore the last backed-up state.
    ///
    /// Requires that the tessellation was created with `with_backup()`.
    /// Returns true if restore was successful.
    pub fn restore(&mut self) -> bool {
        if !self.backup_enabled {
            return false;
        }

        // Take backup out to avoid borrow issues
        let Some(backup) = self.backup.take() else {
            return false;
        };

        // Restore container state
        let affected = self.state.affected_ids.clone();
        self.state
            .container
            .restore_from(&backup.container, &affected);

        // Restore result state
        let n = self.state.result.contacts_by_sphere.len();
        let needs_rebuild = if self.state.was_full_reinit
            || backup.result.contacts_by_sphere.len() != n
            || backup.cell_summaries.len() != n
        {
            self.state.result = backup.result;
            self.state.cell_summaries = backup.cell_summaries;
            false
        } else {
            for &id in &affected {
                if id < n {
                    self.state.result.contacts_by_sphere[id]
                        .clone_from(&backup.result.contacts_by_sphere[id]);
                    self.state.cell_summaries[id].clone_from(&backup.cell_summaries[id]);
                }
            }
            true
        };

        self.state.changed_ids = backup.changed_ids;
        self.state.affected_ids = backup.affected_ids;
        self.state.was_full_reinit = backup.was_full_reinit;

        if needs_rebuild {
            self.rebuild_cells_from_summaries();
        }

        true
    }

    /// Get the detailed result with per-sphere contact storage.
    #[must_use]
    pub const fn result(&self) -> &UpdateableResult {
        &self.state.result
    }

    /// Get aggregated summary as `TessellationResult`.
    #[must_use]
    pub fn summary(&self) -> TessellationResult {
        let n = self.state.result.contacts_by_sphere.len();
        let mut contacts = Vec::new();

        // Collect unique contacts (only where id_a == sphere_id to avoid duplicates)
        for (i, sphere_contacts) in self.state.result.contacts_by_sphere.iter().enumerate() {
            for cds in sphere_contacts {
                if cds.id_a == i {
                    contacts.push(Contact {
                        id_a: cds.id_a,
                        id_b: cds.id_b,
                        area: cds.area,
                        arc_length: cds.arc_length,
                    });
                }
            }
        }

        let cells: Vec<Cell> = (0..n)
            .filter_map(|i| {
                let cs = &self.state.cell_summaries[i];
                if cs.stage == 2 {
                    Some(Cell {
                        index: cs.id,
                        sas_area: cs.sas_area,
                        volume: cs.sas_inside_volume,
                    })
                } else {
                    None
                }
            })
            .collect();

        TessellationResult { contacts, cells }
    }

    /// Get IDs of spheres that changed in the last update.
    #[must_use]
    pub fn changed_ids(&self) -> &[usize] {
        &self.state.changed_ids
    }

    /// Get IDs of spheres affected by the last update (changed + neighbors).
    #[must_use]
    pub fn affected_ids(&self) -> &[usize] {
        &self.state.affected_ids
    }

    /// Whether the last update required a full recomputation.
    #[must_use]
    pub const fn last_update_was_full_reinit(&self) -> bool {
        self.state.was_full_reinit
    }

    /// Whether backup is enabled.
    #[must_use]
    pub const fn backup_enabled(&self) -> bool {
        self.backup_enabled
    }

    fn prepare_for_update(&mut self) {
        // Save current state as backup if backup is enabled and result exists
        if self.backup_enabled && !self.state.result.is_empty() {
            self.backup = Some(State {
                container: self.state.container.clone_state(),
                result: self.state.result.clone(),
                cell_summaries: self.state.cell_summaries.clone(),
                changed_ids: self.state.changed_ids.clone(),
                affected_ids: self.state.affected_ids.clone(),
                was_full_reinit: self.state.was_full_reinit,
            });
        }

        self.state.changed_ids.clear();
        self.state.affected_ids.clear();
        self.state.was_full_reinit = false;
    }

    fn update_internal(&mut self, balls: &[Ball], changed_ids: Option<&[usize]>) -> bool {
        self.prepare_for_update();

        let new_spheres: Vec<Sphere> = balls
            .iter()
            .map(|b| Sphere::from_ball(b, self.probe))
            .collect();

        let update_result = self.state.container.update(&new_spheres, changed_ids);

        match update_result {
            None => false, // No changes
            Some(result) if result.was_full_reinit => {
                self.state.was_full_reinit = true;
                self.compute_full_tessellation();
                true
            }
            Some(result) => {
                self.state.changed_ids = result.changed_ids;
                self.state.affected_ids = result.affected_ids;
                self.state.was_full_reinit = false;
                self.update_using_affected();
                true
            }
        }
    }

    fn compute_full_tessellation(&mut self) {
        let n = self.state.container.spheres().len();
        let periodic = self.state.container.periodic_box().is_some();

        // Collect collision pairs
        let collision_pairs = self.collect_collision_pairs(None);

        // Compute contact descriptors (with original IDs for periodic)
        let (all_summaries, deduped_summaries) =
            self.compute_contact_summaries_with_dedup(&collision_pairs, periodic, n);

        // Organize contacts by sphere (use deduped canonical IDs)
        self.state.result.contacts_by_sphere = vec![Vec::new(); n];
        for cds in &deduped_summaries {
            if cds.area > 0.0 {
                self.state.result.contacts_by_sphere[cds.id_a].push(cds.clone());
                if cds.id_b != cds.id_a {
                    self.state.result.contacts_by_sphere[cds.id_b].push(cds.clone());
                }
            }
        }

        // Compute cells using all summaries with original IDs (for periodic)
        self.compute_cells_from_all_summaries(&all_summaries, periodic, n);
    }

    fn update_using_affected(&mut self) {
        let n = self.state.container.spheres().len();
        let periodic = self.state.container.periodic_box().is_some();

        // Create involvement mask
        let mut involvement = vec![false; n];
        for &id in &self.state.affected_ids {
            if id < n {
                involvement[id] = true;
            }
        }

        // Collect collision pairs for involved spheres
        let collision_pairs = self.collect_collision_pairs(Some(&involvement));

        // Compute new contact descriptors
        let new_contacts = self.compute_contact_summaries(&collision_pairs, periodic, n);

        // Remove old contacts where both spheres are affected
        for &sphere_id in &self.state.affected_ids {
            if sphere_id < n {
                self.state.result.contacts_by_sphere[sphere_id]
                    .retain(|cds| !(involvement[cds.id_a % n] && involvement[cds.id_b % n]));
            }
        }

        // Add new contacts
        for cds in &new_contacts {
            if cds.area > 0.0 {
                self.state.result.contacts_by_sphere[cds.id_a].push(cds.clone());
                if cds.id_b != cds.id_a {
                    self.state.result.contacts_by_sphere[cds.id_b].push(cds.clone());
                }
            }
        }

        // Recompute cells for affected spheres
        self.recompute_cells_for_affected();
    }

    fn collect_collision_pairs(&self, involvement: Option<&[bool]>) -> Vec<(usize, usize)> {
        let n = self.state.container.spheres().len();
        let all_collisions = self.state.container.all_colliding_ids();
        let exclusion_statuses = self.state.container.exclusion_statuses();
        let periodic = self.state.container.periodic_box().is_some();

        let mut pairs = Vec::new();

        for (id_a, neighbors) in all_collisions.iter().enumerate().take(n) {
            let a_involved = involvement.is_none_or(|inv| inv.get(id_a).copied().unwrap_or(false));
            if !a_involved || exclusion_statuses.get(id_a).copied().unwrap_or(0) != 0 {
                continue;
            }

            for neighbor in neighbors {
                let id_b = neighbor.index;
                let id_b_canonical = id_b % n;

                let b_involved =
                    involvement.is_none_or(|inv| inv.get(id_b_canonical).copied().unwrap_or(false));
                if !b_involved || exclusion_statuses.get(id_b_canonical).copied().unwrap_or(0) != 0
                {
                    continue;
                }

                // Include pair if:
                // - For periodic: always include periodic images (dedupe happens later)
                // - For non-periodic or canonical: standard a < b ordering
                let include_pair = if periodic && id_b >= n {
                    true // Periodic image - include for later deduplication
                } else {
                    id_a < id_b // Canonical pair - use ordering
                };

                if include_pair {
                    pairs.push((id_a, id_b));
                }
            }
        }

        pairs
    }

    /// Compute contact summaries, returning both original (with original IDs for cells)
    /// and deduplicated (with canonical IDs for contact storage).
    fn compute_contact_summaries_with_dedup(
        &self,
        pairs: &[(usize, usize)],
        periodic: bool,
        n: usize,
    ) -> (Vec<ContactDescriptorSummary>, Vec<ContactDescriptorSummary>) {
        let spheres = self.state.container.populated_spheres();
        let all_collisions = self.state.container.all_colliding_ids();

        let summaries: Vec<ContactDescriptorSummary> = pairs
            .par_iter()
            .filter_map(|&(a_id, b_id)| {
                let neighbors = &all_collisions[a_id];
                let cd = construct_contact_descriptor(spheres, a_id, b_id, neighbors)?;
                let mut summary = cd.to_summary();

                // Keep original IDs for deduplication (matches C++ behavior)
                summary.id_a = a_id;
                summary.id_b = b_id;
                summary.ensure_ids_ordered();

                if summary.area > 0.0 {
                    Some(summary)
                } else {
                    None
                }
            })
            .collect();

        // For periodic, deduplicate for contact storage but keep originals for cells
        if periodic {
            let deduped = Self::deduplicate_periodic_contacts(&summaries, n);
            (summaries, deduped)
        } else {
            (summaries.clone(), summaries)
        }
    }

    /// Compute contact summaries for incremental updates (returns only deduped canonical).
    fn compute_contact_summaries(
        &self,
        pairs: &[(usize, usize)],
        periodic: bool,
        n: usize,
    ) -> Vec<ContactDescriptorSummary> {
        let (_, deduped) = self.compute_contact_summaries_with_dedup(pairs, periodic, n);
        deduped
    }

    /// Deduplicate periodic boundary contacts following C++ algorithm.
    fn deduplicate_periodic_contacts(
        summaries: &[ContactDescriptorSummary],
        n: usize,
    ) -> Vec<ContactDescriptorSummary> {
        // Build map from canonical spheres to boundary contacts involving them
        let mut sphere_to_boundary_contacts: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, summary) in summaries.iter().enumerate() {
            if summary.id_a >= n || summary.id_b >= n {
                sphere_to_boundary_contacts[summary.id_a % n].push(i);
                sphere_to_boundary_contacts[summary.id_b % n].push(i);
            }
        }

        // For each contact, determine its canonical index
        let mut canonical_ids: Vec<usize> = (0..summaries.len()).collect();

        for (i, summary) in summaries.iter().enumerate() {
            if summary.id_a >= n || summary.id_b >= n {
                let sphere_id_a = summary.id_a % n;
                let sphere_id_b = summary.id_b % n;

                let candidates = if sphere_to_boundary_contacts[sphere_id_a].len()
                    <= sphere_to_boundary_contacts[sphere_id_b].len()
                {
                    &sphere_to_boundary_contacts[sphere_id_a]
                } else {
                    &sphere_to_boundary_contacts[sphere_id_b]
                };

                for &candidate_idx in candidates {
                    let candidate = &summaries[candidate_idx];
                    let cand_a = candidate.id_a % n;
                    let cand_b = candidate.id_b % n;
                    if (cand_a == sphere_id_a && cand_b == sphere_id_b)
                        || (cand_a == sphere_id_b && cand_b == sphere_id_a)
                    {
                        canonical_ids[i] = candidate_idx;
                        break;
                    }
                }
            }
        }

        // Keep only contacts where canonical_id == index, then canonicalize IDs
        summaries
            .iter()
            .enumerate()
            .filter(|(i, _)| canonical_ids[*i] == *i)
            .map(|(_, s)| {
                let mut cs = s.clone();
                cs.id_a = s.id_a % n;
                cs.id_b = s.id_b % n;
                cs.ensure_ids_ordered();
                cs
            })
            .collect()
    }

    /// Compute cells from all summaries (handles periodic boundary correctly).
    fn compute_cells_from_all_summaries(
        &mut self,
        all_summaries: &[ContactDescriptorSummary],
        periodic: bool,
        n: usize,
    ) {
        let spheres = self.state.container.spheres();

        self.state.cell_summaries = (0..n)
            .map(|i| CellContactSummary {
                id: i,
                ..Default::default()
            })
            .collect();

        if periodic {
            // For periodic: only add to cells where the ID is canonical (< n)
            // This matches C++ behavior and avoids double-counting boundary contacts
            for cds in all_summaries {
                if cds.area > 0.0 {
                    if cds.id_a < n {
                        self.state.cell_summaries[cds.id_a].add(cds);
                    }
                    if cds.id_b < n && cds.id_b != cds.id_a {
                        self.state.cell_summaries[cds.id_b].add(cds);
                    }
                }
            }
        } else {
            // Non-periodic: use contacts_by_sphere which has canonical IDs
            for (i, contacts) in self.state.result.contacts_by_sphere.iter().enumerate() {
                for cds in contacts {
                    self.state.cell_summaries[i].add(cds);
                }
            }
        }

        // Compute SAS for each cell
        for (i, (cs, sphere)) in self
            .state
            .cell_summaries
            .iter_mut()
            .zip(spheres.iter())
            .enumerate()
        {
            if cs.stage == 1 {
                cs.compute_sas(sphere.r);
            } else if cs.stage == 0
                && !self.state.container.is_excluded(i)
                && self.state.container.colliding_ids(i).is_empty()
            {
                cs.compute_sas_detached(i, sphere.r);
            }
        }

        self.rebuild_cells_from_summaries();
    }

    fn recompute_cells_for_affected(&mut self) {
        let spheres = self.state.container.spheres();

        for &sphere_id in &self.state.affected_ids {
            if sphere_id >= spheres.len() {
                continue;
            }

            // Reset cell summary
            let mut cs = CellContactSummary {
                id: sphere_id,
                ..Default::default()
            };

            // Accumulate from contacts
            for cds in &self.state.result.contacts_by_sphere[sphere_id] {
                cs.add(cds);
            }

            // Compute SAS
            if cs.stage == 1 {
                cs.compute_sas(spheres[sphere_id].r);
            } else if cs.stage == 0 && !self.state.container.is_excluded(sphere_id) {
                // Check if sphere has any non-excluded neighbors with contacts
                // If no contacts and not excluded, treat as detached
                let has_active_contacts =
                    !self.state.result.contacts_by_sphere[sphere_id].is_empty();
                if !has_active_contacts {
                    cs.compute_sas_detached(sphere_id, spheres[sphere_id].r);
                }
            }

            self.state.cell_summaries[sphere_id] = cs;
        }

        self.rebuild_cells_from_summaries();
    }

    fn rebuild_cells_from_summaries(&mut self) {
        self.state.result.cells = self
            .state
            .cell_summaries
            .iter()
            .filter(|cs| cs.stage == 2)
            .map(|cs| Cell {
                index: cs.id,
                sas_area: cs.sas_area,
                volume: cs.sas_inside_volume,
            })
            .collect();
    }
}

impl Default for UpdateableTessellation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_updateable_basic() {
        let balls = vec![
            Ball::new(0.0, 0.0, 0.0, 1.0),
            Ball::new(2.0, 0.0, 0.0, 1.0),
            Ball::new(4.0, 0.0, 0.0, 1.0),
        ];

        let mut tess = UpdateableTessellation::new();
        assert!(tess.init(&balls, 0.5, None));

        let summary = tess.summary();
        assert_eq!(summary.contacts.len(), 2);
        assert_eq!(summary.cells.len(), 3);
    }

    #[test]
    fn test_updateable_with_update() {
        let mut balls = vec![Ball::new(0.0, 0.0, 0.0, 1.0), Ball::new(2.0, 0.0, 0.0, 1.0)];

        let mut tess = UpdateableTessellation::new();
        assert!(tess.init(&balls, 0.5, None));

        let summary1 = tess.summary();

        balls[0].x += 0.1;
        assert!(tess.update_with_changed(&balls, &[0]));

        let summary2 = tess.summary();
        assert_eq!(summary1.contacts.len(), summary2.contacts.len());
    }

    #[test]
    fn test_updateable_backup_restore() {
        let mut balls = vec![Ball::new(0.0, 0.0, 0.0, 1.0), Ball::new(2.0, 0.0, 0.0, 1.0)];

        let mut tess = UpdateableTessellation::with_backup();
        assert!(tess.init(&balls, 0.5, None));

        let summary_init = tess.summary();
        let init_area = summary_init.contacts[0].area;

        // First update to create backup
        balls[0].x += 0.1;
        assert!(tess.update_with_changed(&balls, &[0]));

        // Second update
        balls[0].x += 0.1;
        assert!(tess.update_with_changed(&balls, &[0]));

        let summary_after = tess.summary();

        // Restore should go back to state before last update
        assert!(tess.restore());

        let summary_restored = tess.summary();
        // Area should be different from after but not necessarily equal to init
        // (restored to state after first update)
        assert!(
            (summary_restored.contacts[0].area - summary_after.contacts[0].area).abs() > 0.0001
                || summary_restored.contacts[0].area == init_area
        );
    }
}
