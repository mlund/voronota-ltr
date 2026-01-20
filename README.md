# Rust Port of VoronotaLT

Unofficial Rust port of [voronota-lt](https://github.com/kliment-olechnovic/voronota/tree/master/expansion_lt)
originally written in C++.
Computes radical Voronoi tessellation of atomic balls constrained inside a solvent-accessible surface.
Outputs inter-atom contact areas, solvent accessible surface (SAS) areas, and volumes.

## Installation

### Library

```toml
[dependencies]
voronotalt = "0.1"
```

### CLI

```sh
cargo install voronotalt
```

## Usage

### API

```rust
use voronotalt::{Ball, compute_tessellation};

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
use voronotalt::{Ball, PeriodicBox, compute_tessellation};

let balls = vec![Ball::new(0.0, 0.0, 0.0, 1.5)];
let pbox = PeriodicBox::from_corners((0.0, 0.0, 0.0), (10.0, 10.0, 10.0));

let result = compute_tessellation(&balls, 1.4, Some(&pbox), None);
```

### CLI

Input is a `.xyzr` file with whitespace-separated values (last 4 columns: x y z radius):

```sh
# Summary output
voronotalt -i atoms.xyzr --probe 1.4

# Print contacts table
voronotalt -i atoms.xyzr --probe 1.4 --print-contacts

# Print cells table
voronotalt -i atoms.xyzr --probe 1.4 --print-cells

# With periodic boundary conditions
voronotalt -i atoms.xyzr --probe 1.4 --periodic-box-corners 0 0 0 100 100 100

# From stdin
cat atoms.xyzr | voronotalt --probe 1.4
```

Run `voronotalt --help` for all options.

## Benchmarks

Performance comparison with C++ voronota-lt on 9745 balls (probe=1.4):

| Implementation | Single-threaded | Multi-threaded | Speedup |
|----------------|-----------------|----------------|---------|
| C++ voronota-lt (g++ -O3, OpenMP) | 187 ms | 42 ms | 4.5x |
| Rust voronotalt (release, Rayon) | 177 ms | 31 ms | 5.7x |

Run benchmarks with `cargo bench`.

## License

MIT

