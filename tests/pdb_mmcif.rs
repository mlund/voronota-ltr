//! Integration tests for PDB and mmCIF input parsing.

mod common;

use approx::assert_abs_diff_eq;
use common::{EPSILON, parse_reference_cells, require_test_file};
use voronota_ltr::FileComputeError;
use voronota_ltr::input::{ParseOptions, RadiiLookup, compute_tessellation_from_file, parse_file};
use voronota_ltr::{Results, compute_tessellation};

#[test]
fn pdb_1ctf_ball_count() {
    let Some(path) = require_test_file("assembly_1ctf.pdb1") else {
        return;
    };

    let options = ParseOptions::default();
    let radii = RadiiLookup::new();
    let balls = parse_file(&path, &options, &radii).expect("Failed to parse PDB file");

    // Default includes heteroatoms (matching C++ behavior)
    assert_eq!(balls.len(), 492, "Expected 492 balls from 1ctf.pdb");
}

#[test]
fn pdb_1ctf_contacts() {
    let Some(path) = require_test_file("assembly_1ctf.pdb1") else {
        return;
    };

    let options = ParseOptions::default();
    let radii = RadiiLookup::new();
    let balls = parse_file(&path, &options, &radii).expect("Failed to parse PDB file");

    let result = compute_tessellation(&balls, 1.4, None, None, false);

    assert_eq!(result.contacts.len(), 3078, "Expected 3078 contacts");
    assert_eq!(result.cells.len(), 492, "Expected 492 cells");

    assert_abs_diff_eq!(result.total_sas_area(), 4097.64, epsilon = EPSILON);
    assert_abs_diff_eq!(result.total_contact_area(), 10663.91, epsilon = EPSILON);
}

#[test]
fn mmcif_1ctf_ball_count() {
    let Some(path) = require_test_file("assembly_1ctf.cif") else {
        return;
    };

    let options = ParseOptions::default();
    let radii = RadiiLookup::new();
    let balls = parse_file(&path, &options, &radii).expect("Failed to parse mmCIF file");

    // mmCIF file has two chains (A, A-2) in model 1
    // Default includes heteroatoms (matching C++ behavior)
    assert_eq!(
        balls.len(),
        984,
        "Expected 984 balls from 1ctf.cif (both chains in model 1)"
    );
}

#[test]
fn mmcif_1ctf_contacts() {
    let Some(path) = require_test_file("assembly_1ctf.cif") else {
        return;
    };

    let options = ParseOptions::default();
    let radii = RadiiLookup::new();
    let balls = parse_file(&path, &options, &radii).expect("Failed to parse mmCIF file");

    let result = compute_tessellation(&balls, 1.4, None, None, false);

    assert_eq!(result.cells.len(), 984, "Expected 984 cells");
    assert_eq!(result.contacts.len(), 6354, "Expected 6354 contacts");
}

#[test]
fn pdb_assembly_matches_mmcif() {
    // When parsing PDB as assembly, both models are included
    // This should match mmCIF which has both chains in model 1
    let (Some(pdb_path), Some(cif_path)) = (
        require_test_file("assembly_1ctf.pdb1"),
        require_test_file("assembly_1ctf.cif"),
    ) else {
        return;
    };

    let radii = RadiiLookup::new();

    // Parse PDB as assembly (both models)
    let pdb_options = ParseOptions {
        as_assembly: true,
        ..Default::default()
    };
    let pdb_balls = parse_file(&pdb_path, &pdb_options, &radii).expect("Failed to parse PDB file");

    // Parse mmCIF (both chains are in model 1)
    let cif_options = ParseOptions::default();
    let cif_balls =
        parse_file(&cif_path, &cif_options, &radii).expect("Failed to parse mmCIF file");

    // Both should have same number of balls (atom order may differ between formats)
    assert_eq!(
        pdb_balls.len(),
        cif_balls.len(),
        "PDB (as assembly) and mmCIF should produce same number of balls"
    );
}

#[test]
fn exclude_heteroatoms() {
    let Some(path) = require_test_file("assembly_1ctf.pdb1") else {
        return;
    };

    let radii = RadiiLookup::new();

    // Default (includes heteroatoms)
    let default_options = ParseOptions::default();
    let balls_with = parse_file(&path, &default_options, &radii).expect("Failed to parse PDB file");

    // Excluding heteroatoms
    let exclude_options = ParseOptions {
        exclude_heteroatoms: true,
        ..Default::default()
    };
    let balls_without =
        parse_file(&path, &exclude_options, &radii).expect("Failed to parse PDB file");

    assert_eq!(balls_with.len(), 492, "Expected 492 with heteroatoms");
    assert_eq!(balls_without.len(), 487, "Expected 487 without heteroatoms");
}

#[test]
fn radii_lookup_integration() {
    let lookup = RadiiLookup::new();

    // Test backbone atoms
    assert_abs_diff_eq!(lookup.get_radius("ALA", "CA"), 1.90, epsilon = 0.001);
    assert_abs_diff_eq!(lookup.get_radius("ALA", "N"), 1.70, epsilon = 0.001);
    assert_abs_diff_eq!(lookup.get_radius("ALA", "C"), 1.75, epsilon = 0.001);
    assert_abs_diff_eq!(lookup.get_radius("ALA", "O"), 1.49, epsilon = 0.001);

    // Test sidechain atoms
    assert_abs_diff_eq!(lookup.get_radius("ALA", "CB"), 1.92, epsilon = 0.001);
    assert_abs_diff_eq!(lookup.get_radius("ARG", "NH1"), 1.62, epsilon = 0.001);

    // Test metal ions
    assert_abs_diff_eq!(lookup.get_radius("ZN", "ZN"), 0.74, epsilon = 0.001);
}

#[test]
fn pdb_1ctf_per_atom_comparison() {
    let Some(pdb_path) = require_test_file("assembly_1ctf.pdb1") else {
        return;
    };
    let Some(ref_path) = require_test_file("1ctf_pdb_cells_reference.tsv") else {
        return;
    };

    // Parse and compute
    let options = ParseOptions::default();
    let radii = RadiiLookup::new();
    let balls = parse_file(&pdb_path, &options, &radii).expect("Failed to parse PDB file");
    let result = compute_tessellation(&balls, 1.4, None, None, false);

    // Load reference data
    let reference = parse_reference_cells(&ref_path);

    assert_eq!(
        result.cells.len(),
        reference.len(),
        "Cell count mismatch with reference"
    );

    // Compare per-atom values
    let tolerance = 0.01;
    let mut max_sas_diff = 0.0_f64;
    let mut max_vol_diff = 0.0_f64;

    for (i, (cell, refcell)) in result.cells.iter().zip(reference.iter()).enumerate() {
        assert_eq!(
            cell.index, refcell.index,
            "Atom index mismatch at position {i}"
        );

        let sas_diff = (cell.sas_area - refcell.sas_area).abs();
        let vol_diff = (cell.volume - refcell.volume).abs();

        max_sas_diff = max_sas_diff.max(sas_diff);
        max_vol_diff = max_vol_diff.max(vol_diff);

        assert!(
            sas_diff < tolerance,
            "Atom {i}: SAS area mismatch: got {}, expected {}, diff {}",
            cell.sas_area,
            refcell.sas_area,
            sas_diff
        );
        assert!(
            vol_diff < tolerance,
            "Atom {i}: Volume mismatch: got {}, expected {}, diff {}",
            cell.volume,
            refcell.volume,
            vol_diff
        );
    }

    eprintln!(
        "Per-atom comparison passed: max SAS diff = {max_sas_diff:.6}, max vol diff = {max_vol_diff:.6}"
    );
}

#[test]
fn custom_radii_file() {
    let (Some(pdb_path), Some(radii_path)) = (
        require_test_file("assembly_1ctf.pdb1"),
        require_test_file("custom_radii.txt"),
    ) else {
        return;
    };

    // Load custom radii
    let mut radii = RadiiLookup::empty();
    let radii_content = std::fs::read_to_string(&radii_path).expect("Failed to read radii file");
    radii
        .load_from_text(&radii_content)
        .expect("Failed to parse radii file");

    // Parse as assembly (both models) to match C++ test
    let options = ParseOptions {
        as_assembly: true,
        ..Default::default()
    };
    let balls = parse_file(&pdb_path, &options, &radii).expect("Failed to parse PDB file");

    let result = compute_tessellation(&balls, 1.4, None, None, false);

    // C++ reference values from contacts_1ctf_pdb_as_assembly_with_heteroatoms_using_custom_radii_summary.txt
    assert_eq!(balls.len(), 984, "Expected 984 balls");
    assert_eq!(result.contacts.len(), 6354, "Expected 6354 contacts");
    assert_abs_diff_eq!(result.total_sas_area(), 7094.42, epsilon = EPSILON);
    assert_abs_diff_eq!(result.total_contact_area(), 21423.54, epsilon = EPSILON);
}

#[test]
fn inter_chain_contacts_only() {
    use voronota_ltr::input::{build_chain_grouping, parse_file_with_records};

    let Some(cif_path) = require_test_file("assembly_1ctf.cif") else {
        return;
    };

    let radii = RadiiLookup::new();
    let options = ParseOptions {
        exclude_heteroatoms: true,
        ..Default::default()
    };

    let parsed =
        parse_file_with_records(&cif_path, &options, &radii).expect("Failed to parse mmCIF file");

    let grouping = build_chain_grouping(&parsed.records);
    let result = compute_tessellation(&parsed.balls, 1.4, None, Some(&grouping), false);

    // C++ reference: contacts_1ctf_mmcif_assembly_inter_chain_mesh_summary.txt
    assert_eq!(parsed.balls.len(), 974, "Expected 974 balls");
    assert_eq!(
        result.contacts.len(),
        218,
        "Expected 218 inter-chain contacts"
    );
    assert_abs_diff_eq!(result.total_contact_area(), 513.032, epsilon = 0.001);
}

#[test]
fn inter_residue_contacts_only() {
    use voronota_ltr::input::{build_residue_grouping, parse_file_with_records};

    let Some(pdb_path) = require_test_file("assembly_1ctf.pdb1") else {
        return;
    };

    let radii = RadiiLookup::new();
    let options = ParseOptions::default();

    let parsed =
        parse_file_with_records(&pdb_path, &options, &radii).expect("Failed to parse PDB file");

    let grouping = build_residue_grouping(&parsed.records);
    let result = compute_tessellation(&parsed.balls, 1.4, None, Some(&grouping), false);

    // C++ reference: voronota-lt -i assembly_1ctf.pdb1 --compute-only-inter-residue-contacts
    assert_eq!(parsed.balls.len(), 492, "Expected 492 balls");
    assert_eq!(
        result.contacts.len(),
        2013,
        "Expected 2013 inter-residue contacts"
    );
    assert_abs_diff_eq!(result.total_contact_area(), 4486.49, epsilon = EPSILON);
}

/// Compare selection counts against MDTraj reference values.
/// Values obtained via: `mdtraj.load('tests/data/assembly_1ctf.cif').topology.select(...)`
#[test]
fn selection_counts_match_mdtraj() {
    use voronota_ltr::input::{Selection, parse_file_with_records};

    let Some(cif_path) = require_test_file("assembly_1ctf.cif") else {
        return;
    };

    let radii = RadiiLookup::new();
    let options = ParseOptions {
        exclude_heteroatoms: true,
        ..Default::default()
    };

    let parsed =
        parse_file_with_records(&cif_path, &options, &radii).expect("Failed to parse mmCIF file");

    let count = |sel_str: &str| -> usize {
        let sel = Selection::parse(sel_str).unwrap();
        parsed.records.iter().filter(|r| sel.matches(r)).count()
    };

    // MDTraj reference values (from assembly_1ctf.cif, excludes hydrogens)
    // Total atoms in MDTraj: 1108 (includes H), our parser: 974 (excludes H)
    let mdtraj_values = [
        ("protein", 974),
        ("backbone", 544),
        ("sidechain", 430),
        ("resname ALA", 150),
        ("resname ALA GLY", 198),
        ("hydrophobic", 490),
        ("aromatic", 22),
        ("acidic", 226),
        ("basic", 204),
        ("polar", 54),
        ("charged", 430),
    ];

    for (sel_str, expected) in mdtraj_values {
        let actual = count(sel_str);
        assert_eq!(
            actual, expected,
            "Selection '{}': expected {} atoms (MDTraj), got {}",
            sel_str, expected, actual
        );
    }
}

// Tests for compute_tessellation_from_file

#[test]
fn compute_from_file_basic() {
    let Some(path) = require_test_file("assembly_1ctf.pdb1") else {
        return;
    };

    let result =
        compute_tessellation_from_file(&path, 1.4, None, false, None).expect("Should succeed");

    assert_eq!(result.contacts.len(), 3078, "Expected 3078 contacts");
    assert_eq!(result.cells.len(), 492, "Expected 492 cells");
    assert_abs_diff_eq!(result.total_sas_area(), 4097.64, epsilon = EPSILON);
}

#[test]
fn compute_from_file_with_cell_vertices() {
    let Some(path) = require_test_file("assembly_1ctf.pdb1") else {
        return;
    };

    let result =
        compute_tessellation_from_file(&path, 1.4, None, true, None).expect("Should succeed");

    // With cell vertices, we get tessellation network data
    assert!(
        result.cell_vertices.is_some(),
        "Should have cell vertices when requested"
    );
    assert!(
        result.cell_edges.is_some(),
        "Should have cell edges when requested"
    );
    assert!(
        !result.cell_vertices.unwrap().is_empty(),
        "Cell vertices should not be empty"
    );
}

#[test]
fn compute_from_file_with_selections() {
    let Some(path) = require_test_file("assembly_1ctf.cif") else {
        return;
    };

    // Two chain selections for inter-chain contacts
    let result =
        compute_tessellation_from_file(&path, 1.4, None, false, Some(&["chain A", "chain A-2"]))
            .expect("Should succeed with valid selections");

    // Should have inter-chain contacts only
    assert!(
        result.contacts.len() < 6354,
        "Inter-chain contacts should be fewer than all contacts"
    );
}

#[test]
fn compute_from_file_too_few_selections() {
    let Some(path) = require_test_file("assembly_1ctf.pdb1") else {
        return;
    };

    let result = compute_tessellation_from_file(&path, 1.4, None, false, Some(&["protein"]));

    assert!(
        matches!(result, Err(FileComputeError::TooFewSelections)),
        "Should fail with single selection"
    );
}

#[test]
fn compute_from_file_xyzr_with_selections_fails() {
    use std::io::Write;

    // Create a temporary XYZR file
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let xyzr_path = dir.path().join("test.xyzr");
    let mut file = std::fs::File::create(&xyzr_path).expect("Failed to create file");
    writeln!(file, "0.0 0.0 0.0 1.5").expect("Failed to write");
    writeln!(file, "3.0 0.0 0.0 1.5").expect("Failed to write");

    let result =
        compute_tessellation_from_file(&xyzr_path, 1.4, None, false, Some(&["protein", "hetatm"]));

    assert!(
        matches!(result, Err(FileComputeError::SelectionRequiresRecords)),
        "Should fail when using selections with XYZR format"
    );
}
