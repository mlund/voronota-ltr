//! Comparison test between voronota-ltr and rust-sasa SASA calculations.
//!
//! Both libraries compute solvent-accessible surface area but use different algorithms:
//! - voronota-ltr: Radical Voronoi tessellation
//! - rust-sasa: Shrake-Rupley algorithm (numerical sphere point sampling)

use rust_sasa::{Atom, calculate_sasa_internal};
use voronota_ltr::{Ball, Results, compute_tessellation};

/// Standard atomic radii (Bondi radii, commonly used for SASA)
fn get_atom_radius(element: &str) -> f64 {
    match element.to_uppercase().as_str() {
        "C" => 1.7,
        "N" => 1.55,
        "O" => 1.52,
        "S" => 1.8,
        "H" => 1.2,
        "P" => 1.8,
        _ => 1.5,
    }
}

#[test]
fn test_sasa_comparison_with_rust_sasa() {
    // Create a simple test case: a few atoms in known positions
    let atoms_data = [
        // (x, y, z, element)
        (0.0, 0.0, 0.0, "C"),
        (1.5, 0.0, 0.0, "C"),
        (0.75, 1.3, 0.0, "N"),
        (2.25, 1.3, 0.0, "O"),
        (-0.75, 1.3, 0.0, "C"),
    ];

    let probe = 1.4;

    // Create atoms for rust-sasa
    let rust_sasa_atoms: Vec<Atom> = atoms_data
        .iter()
        .enumerate()
        .map(|(i, (x, y, z, elem))| Atom {
            position: [*x as f32, *y as f32, *z as f32],
            radius: get_atom_radius(elem) as f32,
            id: i,
            parent_id: None,
        })
        .collect();

    // Calculate SASA with rust-sasa (100 points, use all threads)
    let rust_sasa_result = calculate_sasa_internal(&rust_sasa_atoms, probe as f32, 100, -1);

    // Create balls for voronota-ltr
    let balls: Vec<Ball> = atoms_data
        .iter()
        .map(|(x, y, z, elem)| Ball::new(*x, *y, *z, get_atom_radius(elem)))
        .collect();

    // Calculate SASA with voronota-ltr
    let voronota_result = compute_tessellation(&balls, probe, None, None, false);
    let voronota_sas = voronota_result.sas_areas();

    // Compare results
    assert_eq!(
        rust_sasa_result.len(),
        voronota_sas.len(),
        "Number of atoms should match"
    );

    eprintln!("\n=== Per-atom SASA Comparison ===");
    let mut total_rust: f64 = 0.0;
    let mut total_voronota: f64 = 0.0;

    for (i, (rust_val, voronota_val)) in
        rust_sasa_result.iter().zip(voronota_sas.iter()).enumerate()
    {
        let rust_f64 = f64::from(*rust_val);
        total_rust += rust_f64;
        total_voronota += voronota_val;
        let diff = (rust_f64 - voronota_val).abs();
        let rel_diff = if rust_f64 > 0.01 {
            diff / rust_f64 * 100.0
        } else {
            0.0
        };
        eprintln!(
            "Atom {}: rust-sasa={:7.3}, voronota={:7.3}, diff={:6.3} ({:5.1}%)",
            i, rust_val, voronota_val, diff, rel_diff
        );
    }

    eprintln!("\n=== Summary ===");
    eprintln!("Total rust-sasa: {:.2} Å²", total_rust);
    eprintln!("Total voronota:  {:.2} Å²", total_voronota);
    let rel_total = ((total_rust - total_voronota) / total_rust).abs() * 100.0;
    eprintln!("Relative diff:   {:.1}%", rel_total);

    // Both methods should give similar total SASA (within 20% due to algorithm differences)
    assert!(
        rel_total < 20.0,
        "Total SASA should be within 20%: rust-sasa={:.2}, voronota={:.2}",
        total_rust,
        total_voronota
    );
}
