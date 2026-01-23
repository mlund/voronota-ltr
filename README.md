[![CI](https://github.com/mlund/voronota-ltr/actions/workflows/ci.yml/badge.svg)](https://github.com/mlund/voronota-ltr/actions/workflows/ci.yml)

# Voronota-LT in Rust

Experimental, native Rust port of [voronota-lt](https://github.com/kliment-olechnovic/voronota/tree/master/expansion_lt)
originally written in C++. Replaces the previous [bindings crate](https://github.com/mlund/voronota-rs).
Computes radical Voronoi tessellation of atomic balls constrained inside a solvent-accessible surface.
Outputs inter-atom contact areas, solvent accessible surface (SAS) areas, and volumes.

## Features

- [x] Pure, safe Rust
- [x] Basic radical tessellation (stateless)
- [x] Updateable tessellation for incremental updates (stateful)
- [x] Periodic boundaries
- [x] Per atom SASA and volume
- [x] Groupings to avoid internal contacts
- [x] Parallel processing using Rayon - see benchmarks below
- [x] PDB, mmCIF, and XYZR input formats with auto-detection
- [x] Unit-tests and benchmarks carried over from the C++ side
- [x] Python bindings via PyO3
- Based on Voronota-LT v1.1.479 (`f5ad92de4e9723ab767db3e5035c0e7532f31595`)

## Installation

The port can be used either as a library for other projects, or as a basic CLI tool:

```sh
cargo install voronota-ltr
```

## Rust API

```rust
use voronota_ltr::{Ball, Results, compute_tessellation};

let balls = vec![
    Ball::new(0.0, 0.0, 0.0, 1.5),
    Ball::new(3.0, 0.0, 0.0, 1.5),
    Ball::new(1.5, 2.5, 0.0, 1.5),
];

let result = compute_tessellation(&balls, 1.4, None, None, false);

// Per-ball SAS areas and volumes (indexed by ball)
// Returns None for atoms without contacts (lonely atoms)
let sas_areas: Vec<Option<f64>> = result.sas_areas();
let volumes: Vec<Option<f64>> = result.volumes();

// Total SAS area
let total_sas: f64 = result.total_sas_area();

// Detailed contact and cell data
for contact in &result.contacts {
    println!("Contact {}-{}: area={:.2}", contact.id_a, contact.id_b, contact.area);
}
```

### With periodic boundary conditions

```rust
use voronota_ltr::{Ball, PeriodicBox, compute_tessellation};

let balls = vec![Ball::new(0.0, 0.0, 0.0, 1.5)];
let pbox = PeriodicBox::from_corners((0.0, 0.0, 0.0), (10.0, 10.0, 10.0));

let result = compute_tessellation(&balls, 1.4, Some(&pbox), None, false);
```

### Updateable tessellation

For simulations where only a few spheres change position each step, `UpdateableTessellation`
provides efficient incremental updates:

```rust
use voronota_ltr::{Ball, UpdateableTessellation};

let mut balls = vec![
    Ball::new(0.0, 0.0, 0.0, 1.0),
    Ball::new(2.0, 0.0, 0.0, 1.0),
    Ball::new(4.0, 0.0, 0.0, 1.0),
];

// Create with backup support for undo
let mut tess = UpdateableTessellation::with_backup();
tess.init(&balls, 1.0, None);

// Move first sphere
balls[0].x += 0.1;
tess.update_with_changed(&balls, &[0]);  // Only recompute affected contacts

// Get results
let summary = tess.summary();
println!("Total contacts: {}", summary.contacts.len());

// Undo last update
tess.restore();
```

## Command Line Tool

Supports PDB, mmCIF, and XYZR input formats (auto-detected from extension or content):

```sh
# PDB / mmCIF / XYZR input
voronota-ltr structure.pdb --probe 1.4

# Save JSON output to file instead of stdout
voronota-ltr structure.pdb -o results.json

# Exclude heteroatoms (HETATM records)
voronota-ltr structure.pdb --exclude-heteroatoms

# Include hydrogen atoms (excluded by default)
voronota-ltr structure.pdb --include-hydrogens

# Custom radii file (format: residue atom radius per line)
voronota-ltr structure.pdb --radii-file custom_radii.txt

# With periodic boundary conditions
voronota-ltr structure.xyzr --periodic-box-corners 0 0 0 100 100 100
```

### Custom selections

Group atoms using VMD-like selection syntax to filter contacts.
Only inter-group contacts are computed.

```sh
# Protein-ligand interface (5IN3: galactose-1-phosphate uridylyltransferase)
voronota-ltr 5IN3.cif -s "protein" "resname G1P H2U"

# Homodimer interface
voronota-ltr 5IN3.cif -s "chain A" "chain B"
```

See the [Selection Language](#selection-language) section for full syntax.

### Loading JSON output in Python

If using the CLI, results can be loaded from JSON:

```python
import json

with open("results.json") as f:
    data = json.load(f)

# Per-ball data (indexed by ball, None for atoms without contacts)
sas_areas = data["sas_areas"]  # list of float or None
volumes = data["volumes"]  # list of float or None

# Totals
total_sasa = data["total_sas_area"]
total_volume = data["total_volume"]
total_contact_area = data["total_contact_area"]

# Contact details
for contact in data["contacts"]:
    print(f"Contact {contact['id_a']}-{contact['id_b']}: area={contact['area']:.2f}")
```

### PyMOL visualization

Generate a Python script to visualize contact surfaces in PyMOL:

```sh
voronota-ltr structure.pdb --inter-chain-only --pymol contacts.py
pymol structure.pdb contacts.py
```

This creates three CGO objects: `contacts_balls` (cyan spheres), `contacts_faces` (yellow contact surfaces), and `contacts_wireframe` (red boundary lines).

## Python Interface

Build and install locally using [maturin](https://www.maturin.rs/):

```sh
pip install maturin
maturin develop --features python          # Development build
maturin develop --features python --release  # Optimized build
```

This installs both the Python module and CLI binary.

Run tests:

```sh
python -m unittest discover -s tests -p "test_*.py"
```

Basic usage:

```python
import voronota_ltr

result = voronota_ltr.compute_tessellation(
    balls=[(0, 0, 0, 1.5), (3, 0, 0, 1.5), (1.5, 2.5, 0, 1.5)],
    probe=1.4,
)

print(f"Total SAS area: {result['total_sas_area']:.2f}")
print(f"Total volume: {result['total_volume']:.2f}")

for contact in result["contacts"]:
    print(f"Contact {contact['id_a']}-{contact['id_b']}: area={contact['area']:.2f}")
```

Input balls can be tuples, dicts, or NumPy arrays:

```python
import numpy as np

# Tuples
balls = [(0, 0, 0, 1.5), (3, 0, 0, 1.5)]

# Dicts
balls = [{"x": 0, "y": 0, "z": 0, "r": 1.5}, {"x": 3, "y": 0, "z": 0, "r": 1.5}]

# NumPy array (N x 4)
balls = np.array([[0, 0, 0, 1.5], [3, 0, 0, 1.5]])

result = voronota_ltr.compute_tessellation(balls=balls, probe=1.4)
```

With periodic boundaries:

```python
# Orthorhombic box from corner coordinates
result = voronota_ltr.compute_tessellation(
    balls=balls,
    probe=1.4,
    periodic_box={"corners": [(0, 0, 0), (50, 50, 50)]},
)

# Triclinic cell from lattice vectors
result = voronota_ltr.compute_tessellation(
    balls=balls,
    probe=1.4,
    periodic_box={"vectors": [(50, 0, 0), (0, 50, 0), (0, 0, 50)]},
)
```

With tessellation network output:

```python
result = voronota_ltr.compute_tessellation(
    balls=balls,
    probe=1.4,
    with_cell_vertices=True,
)

for vertex in result["cell_vertices"]:
    print(f"Vertex at ({vertex['x']:.2f}, {vertex['y']:.2f}, {vertex['z']:.2f})")
    print(f"  On SAS: {vertex['is_on_sas']}")
```

## Selection Language

The `-s/--selection` flag accepts VMD-like selection syntax for defining atom groups.
Only contacts between different groups are computed, which is useful for interface analysis.

### Syntax

```sh
voronota-ltr file.pdb -s "selection1" "selection2" ...
```

At least two selections are required. Only contacts between atoms in *different* groups are computed.

### Keywords

| Keyword | Description | Example |
|---------|-------------|---------|
| `chain` | Chain identifier(s) | `chain A B` |
| `resname` | Residue name(s) | `resname ALA GLY` |
| `resid` | Residue number(s) or range | `resid 1 to 100` or `resid 1:100` |
| `name` | Atom name(s) with glob patterns | `name CA` or `name C*` |
| `protein` | Standard amino acids | `protein` |
| `backbone` | Protein backbone (C, CA, N, O) | `backbone` |
| `sidechain` | Protein sidechains (non-backbone) | `sidechain` |
| `nucleic` | Nucleic acid residues | `nucleic` |
| `hetatm` | HETATM records | `hetatm` |
| `hydrophobic` | Hydrophobic residues (ALA, VAL, LEU, ILE, MET, PHE, TRP, PRO, GLY) | `hydrophobic` |
| `aromatic` | Aromatic residues (PHE, TYR, TRP, HIS) | `aromatic` |
| `acidic` | Acidic residues (ASP, GLU) | `acidic` |
| `basic` | Basic residues (ARG, LYS, HIS) | `basic` |
| `polar` | Polar residues (SER, THR, ASN, GLN) | `polar` |
| `charged` | Charged residues (ASP, GLU, ARG, LYS) | `charged` |
| `all` | All atoms | `all` |
| `none` | No atoms | `none` |

### Boolean operators

Combine selections with `and`, `or`, `not`, and parentheses:

```sh
# Sidechain-only dimer interface (exclude backbone)
voronota-ltr 5IN3.cif -s "chain A and sidechain" "chain B and sidechain"

# Protein-ligand contacts, excluding cryoprotectant (EDO)
voronota-ltr 5IN3.cif -s "protein and not resname EDO" "resname G1P H2U ZN"
```

### Glob patterns

Atom and residue names support glob wildcards:

- `*` matches any characters: `name C*` matches CA, CB, CG, etc.
- `?` matches single character: `name ?A` matches CA, NA, etc.
- `[abc]` matches character class: `name C[AG]` matches CA or CG

## Benchmarks

Run benchmarks with `cargo bench`.
Performance on Apple M4 processor (10 cores) - the speedup is relative to single threaded runs:

| Dataset        | Balls | C++ (OpenMP) | Rust (Rayon) | C++ Speedup | Rust Speedup |
|----------------|-------|--------------|--------------|-------------|--------------|
| `balls_cs_1x1` | 100   | 179 µs       | 79 µs        | 0.4x        | 0.8x         |
| `balls_2zsk`   | 3545  | 14 ms        | 12 ms        | 5.1x        | 5.9x         |
| `balls_3dlb`   | 9745  | 38 ms        | 30 ms        | 5.0x        | 5.9x         |

## License

MIT License

Copyright (c) 2026 Kliment Olechnovic and Mikael Lund

