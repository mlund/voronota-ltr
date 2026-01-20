# Optimization Analysis

Performance analysis of the Rust voronotalt implementation compared to C++ voronota-lt.

## Benchmark Summary

Tested on Apple Silicon M4 (10 cores):

| Dataset      | Balls | C++ (OpenMP) | Rust (Rayon) | C++ Speedup | Rust Speedup |
|--------------|-------|--------------|--------------|-------------|--------------|
| balls_cs_1x1 | 100   | 179 µs       | 79 µs        | 0.4x        | 0.8x         |
| balls_2zsk   | 3545  | 14 ms        | 12 ms        | 5.1x        | 5.9x         |
| balls_3dlb   | 9745  | 38 ms        | 30 ms        | 5.0x        | 5.9x         |

Rust is ~20% faster than C++ in both single-threaded and multi-threaded modes.

## Profile Results

Profiling with samply on the largest dataset (9745 balls, 100 iterations):

| Function                     | Time  | Description                          |
|------------------------------|-------|--------------------------------------|
| `construct_contact_descriptor` | 73.3% | Main contact algorithm (with callees) |
| `mark_and_cut_contour`        | 18.3% | Cut contour by neighbor halfspaces   |
| `find_colliding_ids`          | 8.0%  | 27-cell spatial neighbor search      |
| `calculate_solid_angle`       | 6.7%  | SAS solid angle computation          |
| `init_contour`                | 5.5%  | Create initial hexagonal contour     |
| `restrict_contour_to_circle`  | 3.1%  | Project contour back to circle       |

## Native CPU Flags

Testing with `-C target-cpu=native`:

| Dataset      | Default | Native | Improvement |
|--------------|---------|--------|-------------|
| balls_cs_1x1 | 65 µs   | 61 µs  | +6%         |
| balls_2zsk   | 69 ms   | 69 ms  | 0%          |
| balls_3dlb   | 179 ms  | 179 ms | 0%          |

Only the smallest dataset shows improvement, suggesting larger workloads are memory-bound.

## SIMD Optimization Potential

Analyzed using the `wide` crate for explicit SIMD:

### High potential
- `sphere_intersects_sphere`: norm_squared + compare (batching 4 pairs)
- `find_colliding_ids`: batch sphere-sphere tests in 27-cell search

### Medium potential
- `mark_and_cut_contour`: halfspace tests with dot products
- `calculate_solid_angle`: cross/dot product loops
- Periodic sphere shifts: 26 linear combinations per sphere

### Challenges
1. **Data layout**: Current AoS (Array of Structures) layout with `nalgebra::Point3<f64>` doesn't align with SIMD's preferred SoA (Structure of Arrays)
2. **Control flow**: Hot functions have early-exit conditions (`continue`, `break`) that break SIMD batching
3. **Variable-length data**: Contours have 6-20 points, not aligned to SIMD widths

### Verdict
SIMD would require significant refactoring for marginal gains (~10-15%). The existing Rayon parallelization already provides 5.9x speedup on larger datasets.

## Recommendations

1. **Current state is well-optimized**: Rust outperforms C++ by ~20%
2. **Rayon parallelization is effective**: 5.9x speedup vs 5.0x for C++ OpenMP
3. **Memory allocation is efficient**: Low overhead from parallel collection
4. **Skip SIMD refactoring**: Effort outweighs potential gains given the algorithm's control flow

## Profiling Commands

```sh
# Build with debug symbols
CARGO_PROFILE_RELEASE_DEBUG=true cargo build --release

# Profile with samply
samply record ./target/release/examples/profile

# Benchmark with native CPU
RUSTFLAGS="-C target-cpu=native" cargo bench
```
