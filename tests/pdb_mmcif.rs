//! Integration tests for PDB and mmCIF input parsing.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use voronota_ltr::input::{ParseOptions, RadiiLookup, parse_file};
use voronota_ltr::{Results, compute_tessellation};

/// Default tolerance for floating-point comparisons
const EPSILON: f64 = 0.01;

/// Get the test data directory path (repo-local tests/data/).
fn get_test_data_dir() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(manifest_dir).join("tests/data")
}

/// Helper macro for approximate equality assertions with tolerance.
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => {
        assert_approx_eq!($a, $b, EPSILON)
    };
    ($a:expr, $b:expr, $tol:expr) => {{
        let (a, b, tol) = ($a, $b, $tol);
        assert!(
            (a - b).abs() < tol,
            "assertion failed: |{} - {}| = {} >= {}",
            a,
            b,
            (a - b).abs(),
            tol
        );
    }};
}

fn test_file(name: &str) -> PathBuf {
    get_test_data_dir().join(name)
}

#[test]
fn pdb_1ctf_ball_count() {
    let path = test_file("assembly_1ctf.pdb1");
    if !path.exists() {
        eprintln!("Skipping test: {path:?} not found");
        return;
    }

    let options = ParseOptions::default();
    let radii = RadiiLookup::new();
    let balls = parse_file(&path, &options, &radii).expect("Failed to parse PDB file");

    // Default includes heteroatoms (matching C++ behavior)
    assert_eq!(balls.len(), 492, "Expected 492 balls from 1ctf.pdb");
}

#[test]
fn pdb_1ctf_contacts() {
    let path = test_file("assembly_1ctf.pdb1");
    if !path.exists() {
        eprintln!("Skipping test: {path:?} not found");
        return;
    }

    let options = ParseOptions::default();
    let radii = RadiiLookup::new();
    let balls = parse_file(&path, &options, &radii).expect("Failed to parse PDB file");

    let result = compute_tessellation(&balls, 1.4, None, None);

    assert_eq!(result.contacts.len(), 3078, "Expected 3078 contacts");
    assert_eq!(result.cells.len(), 492, "Expected 492 cells");

    assert_approx_eq!(result.total_sas_area(), 4097.64);
    assert_approx_eq!(result.total_contact_area(), 10663.91);
}

#[test]
fn mmcif_1ctf_ball_count() {
    let path = test_file("assembly_1ctf.cif");
    if !path.exists() {
        eprintln!("Skipping test: {path:?} not found");
        return;
    }

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
    let path = test_file("assembly_1ctf.cif");
    if !path.exists() {
        eprintln!("Skipping test: {path:?} not found");
        return;
    }

    let options = ParseOptions::default();
    let radii = RadiiLookup::new();
    let balls = parse_file(&path, &options, &radii).expect("Failed to parse mmCIF file");

    let result = compute_tessellation(&balls, 1.4, None, None);

    assert_eq!(result.cells.len(), 984, "Expected 984 cells");
    assert_eq!(result.contacts.len(), 6354, "Expected 6354 contacts");
}

#[test]
fn pdb_assembly_matches_mmcif() {
    // When parsing PDB as assembly, both models are included
    // This should match mmCIF which has both chains in model 1
    let pdb_path = test_file("assembly_1ctf.pdb1");
    let cif_path = test_file("assembly_1ctf.cif");

    if !pdb_path.exists() || !cif_path.exists() {
        eprintln!("Skipping test: test files not found");
        return;
    }

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
    let path = test_file("assembly_1ctf.pdb1");
    if !path.exists() {
        eprintln!("Skipping test: {path:?} not found");
        return;
    }

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
    assert_approx_eq!(lookup.get_radius("ALA", "CA"), 1.90, 0.001);
    assert_approx_eq!(lookup.get_radius("ALA", "N"), 1.70, 0.001);
    assert_approx_eq!(lookup.get_radius("ALA", "C"), 1.75, 0.001);
    assert_approx_eq!(lookup.get_radius("ALA", "O"), 1.49, 0.001);

    // Test sidechain atoms
    assert_approx_eq!(lookup.get_radius("ALA", "CB"), 1.92, 0.001);
    assert_approx_eq!(lookup.get_radius("ARG", "NH1"), 1.62, 0.001);

    // Test metal ions
    assert_approx_eq!(lookup.get_radius("ZN", "ZN"), 0.74, 0.001);
}

/// Reference data for per-atom comparison (index, sas_area, volume).
struct CellReference {
    index: usize,
    sas_area: f64,
    volume: f64,
}

/// Parse C++ reference file (TSV format from --print-cells).
fn parse_reference_cells(path: &Path) -> Vec<CellReference> {
    let file = File::open(path).expect("Failed to open reference file");
    let reader = BufReader::new(file);

    reader
        .lines()
        .skip(1) // Skip header
        .filter_map(|line| {
            let line = line.ok()?;
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() < 8 {
                return None;
            }
            Some(CellReference {
                index: parts[5].parse().ok()?,
                sas_area: parts[6].parse().ok()?,
                volume: parts[7].parse().ok()?,
            })
        })
        .collect()
}

#[test]
fn pdb_1ctf_per_atom_comparison() {
    let pdb_path = test_file("assembly_1ctf.pdb1");
    let ref_path = test_file("1ctf_pdb_cells_reference.tsv");

    if !pdb_path.exists() {
        eprintln!("Skipping test: {pdb_path:?} not found");
        return;
    }
    if !ref_path.exists() {
        eprintln!("Skipping test: {ref_path:?} not found");
        return;
    }

    // Parse and compute
    let options = ParseOptions::default();
    let radii = RadiiLookup::new();
    let balls = parse_file(&pdb_path, &options, &radii).expect("Failed to parse PDB file");
    let result = compute_tessellation(&balls, 1.4, None, None);

    // Load reference data
    let reference = parse_reference_cells(&ref_path);

    assert_eq!(
        result.cells.len(),
        reference.len(),
        "Cell count mismatch with reference"
    );

    // Compare per-atom values
    let tolerance = 0.01; // Allow small numerical differences
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
    let pdb_path = test_file("assembly_1ctf.pdb1");
    let radii_path = test_file("custom_radii.txt");

    if !pdb_path.exists() || !radii_path.exists() {
        eprintln!("Skipping test: test files not found");
        return;
    }

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

    let result = compute_tessellation(&balls, 1.4, None, None);

    // C++ reference values from contacts_1ctf_pdb_as_assembly_with_heteroatoms_using_custom_radii_summary.txt
    assert_eq!(balls.len(), 984, "Expected 984 balls");
    assert_eq!(result.contacts.len(), 6354, "Expected 6354 contacts");
    assert_approx_eq!(result.total_sas_area(), 7094.42);
    assert_approx_eq!(result.total_contact_area(), 21423.54);
}

#[test]
fn inter_chain_contacts_only() {
    use voronota_ltr::input::{build_chain_grouping, parse_file_with_records};

    let cif_path = test_file("assembly_1ctf.cif");
    if !cif_path.exists() {
        eprintln!("Skipping test: {cif_path:?} not found");
        return;
    }

    let radii = RadiiLookup::new();
    let options = ParseOptions {
        exclude_heteroatoms: true,
        ..Default::default()
    };

    let parsed =
        parse_file_with_records(&cif_path, &options, &radii).expect("Failed to parse mmCIF file");

    let grouping = build_chain_grouping(&parsed.records);
    let result = compute_tessellation(&parsed.balls, 1.4, None, Some(&grouping));

    // C++ reference: contacts_1ctf_mmcif_assembly_inter_chain_mesh_summary.txt
    // 974 balls, 218 inter-chain contacts, 513.032 contact area
    assert_eq!(parsed.balls.len(), 974, "Expected 974 balls");
    assert_eq!(
        result.contacts.len(),
        218,
        "Expected 218 inter-chain contacts"
    );
    assert_approx_eq!(result.total_contact_area(), 513.032, 0.001);
}

#[test]
fn inter_residue_contacts_only() {
    use voronota_ltr::input::{build_residue_grouping, parse_file_with_records};

    let pdb_path = test_file("assembly_1ctf.pdb1");
    if !pdb_path.exists() {
        eprintln!("Skipping test: {pdb_path:?} not found");
        return;
    }

    let radii = RadiiLookup::new();
    let options = ParseOptions::default();

    let parsed =
        parse_file_with_records(&pdb_path, &options, &radii).expect("Failed to parse PDB file");

    let grouping = build_residue_grouping(&parsed.records);
    let result = compute_tessellation(&parsed.balls, 1.4, None, Some(&grouping));

    // C++ reference: voronota-lt -i assembly_1ctf.pdb1 --compute-only-inter-residue-contacts
    // 492 balls, 2013 inter-residue contacts, 4486.49 contact area
    assert_eq!(parsed.balls.len(), 492, "Expected 492 balls");
    assert_eq!(
        result.contacts.len(),
        2013,
        "Expected 2013 inter-residue contacts"
    );
    assert_approx_eq!(result.total_contact_area(), 4486.49, 0.01);
}
