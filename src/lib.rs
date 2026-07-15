// Copyright (c) 2026 Kliment Olechnovic and Mikael Lund
// Part of the voronota-ltr project, licensed under the MIT License.
// SPDX-License-Identifier: MIT

#![deny(missing_docs)]

//! Native rust port of [voronota-lt](https://github.com/kliment-olechnovic/voronota/tree/master/expansion_lt)
//! for computing radical tessellation contacts and cells.
//!
//! This library computes the radical Voronoi tessellation of a set of spheres,
//! providing contact areas between neighboring spheres and solvent-accessible
//! surface (SAS) areas and volumes for each sphere.
//!
//! # Cell measures
//!
//! Detailed [`Cell`] records are sparse. The [`Results::sas_areas`] and [`Results::volumes`]
//! methods instead return one [`CellMeasure`] per input ball. Detached balls are
//! [`CellMeasure::Computed`] with their full-sphere measure, geometrically absent cells are
//! [`CellMeasure::Empty`], and filtered results are [`CellMeasure::NotComputed`].
//!
//! # Example
//!
//! ```
//! use voronota_ltr::{Ball, CellMeasure, Results, compute_tessellation};
//!
//! let balls = vec![
//!     Ball::new(0.0, 0.0, 0.0, 1.5),
//!     Ball::new(3.0, 0.0, 0.0, 1.5),
//!     Ball::new(1.5, 2.5, 0.0, 1.5),
//! ];
//!
//! let result = compute_tessellation(&balls, 1.4, None, None, false);
//!
//! // Per-ball measures distinguish computed, empty, and unavailable cells.
//! let sas_areas: Vec<CellMeasure> = result.sas_areas();
//! let volumes: Vec<CellMeasure> = result.volumes();
//!
//! // Total SAS area across computed cells.
//! let total_sas: f64 = result.total_sas_area();
//!
//! for contact in &result.contacts {
//!     println!("Contact {}-{}: area={:.2}", contact.id_a, contact.id_b, contact.area);
//! }
//! ```

pub(crate) mod contact;
pub(crate) mod geometry;
pub(crate) mod graphics;
/// Input file parsing (PDB, mmCIF, XYZR formats).
pub mod input;
#[cfg(feature = "python")]
mod python;
mod solvent_spheres;
mod spheres_container;
pub(crate) mod spheres_searcher;
mod subdivided_icosahedron;
mod tessellation;
pub(crate) mod types;
mod updateable;

pub use graphics::GraphicsWriter;
pub use input::{FileComputeError, compute_tessellation_from_file};
pub use solvent_spheres::{SolventSphere, SolventSpheresError, compute_solvent_spheres};
pub use subdivided_icosahedron::SubdivisionDepth;
pub use tessellation::{compute_contacts_only, compute_tessellation};
pub use types::{
    Ball, Cell, CellEdge, CellMeasure, CellVertex, Contact, PeriodicBox, Results,
    TessellationResult,
};
pub use updateable::{UpdateableResult, UpdateableTessellation};
