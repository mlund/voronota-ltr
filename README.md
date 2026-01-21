# Rust Port of Voronota-LT

Unofficial Rust port of [voronota-lt](https://github.com/kliment-olechnovic/voronota/tree/master/expansion_lt)
originally written in C++.
Computes radical Voronoi tessellation of atomic balls constrained inside a solvent-accessible surface.
Outputs inter-atom contact areas, solvent accessible surface (SAS) areas, and volumes.

## Features

- [x] Basic radical tessellation (stateless)
- [x] Updateable tessellation for incremental updates
- [x] Periodic boundaries
- [x] Groupings to avoid internal contacts
- [x] Parallel processing using Rayon - see benchmarks below
- [x] Unit-tests and benchmarks carried over from the C++ side
- [x] Pure, safe Rust

## Installation

The port can be used either as a library for other projects, or as a basic CLI tool:

```sh
cargo install voronota-ltr
```

## Usage

### API

```rust
use voronota_ltr::{Ball, compute_tessellation};

let balls = vec![
    Ball::new(0.0, 0.0, 0.0, 1.5),
    Ball::new(3.0, 0.0, 0.0, 1.5),
    Ball::new(1.5, 2.5, 0.0, 1.5),
];

let result = compute_tessellation(&balls, 1.4, None, None);

for contact in &result.contacts {
    println!("Contact {}-{}: area={:.2}", contact.id_a, contact.id_b, contact.area);
}

for cell in &result.cells {
    println!("Cell {}: SAS={:.2}, vol={:.2}", cell.index, cell.sas_area, cell.volume);
}
```

#### With periodic boundary conditions

```rust
use voronota_ltr::{Ball, PeriodicBox, compute_tessellation};

let balls = vec![Ball::new(0.0, 0.0, 0.0, 1.5)];
let pbox = PeriodicBox::from_corners((0.0, 0.0, 0.0), (10.0, 10.0, 10.0));

let result = compute_tessellation(&balls, 1.4, Some(&pbox), None);
```

#### Updateable tessellation

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

### CLI

Input is a `.xyzr` file with whitespace-separated values (last 4 columns: x y z radius):

```sh
# Summary output
voronota-ltr -i atoms.xyzr --probe 1.4

# Print contacts table
voronota-ltr -i atoms.xyzr --probe 1.4 --print-contacts

# Print cells table
voronota-ltr -i atoms.xyzr --probe 1.4 --print-cells

# With periodic boundary conditions
voronota-ltr -i atoms.xyzr --probe 1.4 --periodic-box-corners 0 0 0 100 100 100
```

## Benchmarks

Run benchmarks with `cargo bench`.
Performance on Apple M4 processor (10 cores) - the speedup is relative to single threaded runs:

| Dataset        | Balls | C++ (OpenMP) | Rust (Rayon) | C++ Speedup | Rust Speedup |
|----------------|-------|--------------|--------------|-------------|--------------|
| `balls_cs_1x1` | 100   | 179 µs       | 79 µs        | 0.4x        | 0.8x         |
| `balls_2zsk`   | 3545  | 14 ms        | 12 ms        | 5.1x        | 5.9x         |
| `balls_3dlb`   | 9745  | 38 ms        | 30 ms        | 5.0x        | 5.9x         |

## License

MIT

