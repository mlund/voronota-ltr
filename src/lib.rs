//! Rust port of voronota-lt for computing radical tessellation contacts and cells.
//!
//! This library computes the radical Voronoi tessellation of a set of spheres,
//! providing contact areas between neighboring spheres and solvent-accessible
//! surface (SAS) areas and volumes for each sphere.
//!
//! # Example
//!
//! ```
//! use rust_voronota::{Ball, compute_tessellation};
//!
//! let balls = vec![
//!     Ball::new(0.0, 0.0, 0.0, 1.5),
//!     Ball::new(3.0, 0.0, 0.0, 1.5),
//!     Ball::new(1.5, 2.5, 0.0, 1.5),
//! ];
//!
//! let result = compute_tessellation(&balls, 1.4);
//!
//! for contact in &result.contacts {
//!     println!("Contact {}-{}: area={:.2}", contact.id_a, contact.id_b, contact.area);
//! }
//!
//! for cell in &result.cells {
//!     println!("Cell {}: SAS area={:.2}, volume={:.2}", cell.index, cell.sas_area, cell.volume);
//! }
//! ```

mod contact;
mod geometry;
mod spheres_searcher;
mod tessellation;
mod types;

pub use tessellation::{compute_tessellation, compute_tessellation_periodic};
pub use types::{Ball, Cell, Contact, PeriodicBox, TessellationResult};
