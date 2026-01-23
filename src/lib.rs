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
//! # Example
//!
//! ```
//! use voronota_ltr::{Ball, Results, compute_tessellation};
//!
//! let balls = vec![
//!     Ball::new(0.0, 0.0, 0.0, 1.5),
//!     Ball::new(3.0, 0.0, 0.0, 1.5),
//!     Ball::new(1.5, 2.5, 0.0, 1.5),
//! ];
//!
//! let result = compute_tessellation(&balls, 1.4, None, None, false);
//!
//! // Per-ball SAS areas and volumes (indexed by ball)
//! let sas_areas: Vec<f64> = result.sas_areas();
//! let volumes: Vec<f64> = result.volumes();
//!
//! // Total SAS area
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
mod solvent_spheres;
mod spheres_container;
pub(crate) mod spheres_searcher;
mod subdivided_icosahedron;
mod tessellation;
pub(crate) mod types;
mod updateable;

pub use graphics::GraphicsWriter;
pub use solvent_spheres::{SolventSphere, SolventSpheresError, compute_solvent_spheres};
pub use subdivided_icosahedron::SubdivisionDepth;
pub use tessellation::compute_tessellation;
pub use types::{
    Ball, Cell, CellEdge, CellVertex, Contact, PeriodicBox, Results, TessellationResult,
};
pub use updateable::{UpdateableResult, UpdateableTessellation};
