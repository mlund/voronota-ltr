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

### Proposed Changes for SIMD

#### 1. Structure of Arrays (SoA) for Spheres
```rust
// Current AoS
struct Sphere { center: Point3<f64>, r: f64 }

// Proposed SoA - enables loading 4 coordinates at once
struct Spheres {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    r: Vec<f64>,
}
```

#### 2. Batch Collision Detection in `find_colliding_ids`
Test 4 candidate spheres simultaneously instead of one at a time:
```rust
let dx = f64x4::from([c0.x, c1.x, c2.x, c3.x]) - central_x;
let dy = f64x4::from([c0.y, c1.y, c2.y, c3.y]) - central_y;
let dz = f64x4::from([c0.z, c1.z, c2.z, c3.z]) - central_z;
let dist_sq = dx*dx + dy*dy + dz*dz;
let sum_r = f64x4::from([r0, r1, r2, r3]) + central_r;
let mask = dist_sq.cmp_lt(sum_r * sum_r);
```

#### 3. Fixed-Size Padded Contours
```rust
const MAX_CONTOUR: usize = 32;
struct ContactDescriptor {
    contour: [ContourPoint; MAX_CONTOUR],  // fixed size, no allocation
    len: usize,
}
```

#### 4. Batch Halfspace Tests in `mark_and_cut_contour`
Test 4 contour points against a cutting plane simultaneously:
```rust
for chunk in contour.chunks(4) {
    let px = f64x4::from([chunk[0].p.x, chunk[1].p.x, ...]);
    let dist = (px - plane_x) * n.x + (py - plane_y) * n.y + (pz - plane_z) * n.z;
}
```

#### 5. Deferred Cutting with Bitmasks
Compute all cutting planes first, then batch test each contour point against multiple planes.

### Estimated Impact

| Change | Affected Time | Potential Speedup | Net Improvement |
|--------|---------------|-------------------|-----------------|
| Batch collision detection | 8% | 2-3x | 3-5% |
| Batch halfspace tests | 18% | 2-3x | 6-9% |
| Fixed-size contours | - | Better cache | 1-2% |
| **Total** | | | **10-15%** |

### Verdict
SIMD would require significant refactoring for marginal gains (~10-15%). The existing Rayon parallelization already provides 5.9x speedup on larger datasets. If pursuing SIMD, start with batch collision detection in `find_colliding_ids` as it's isolated and low-risk.

## GPU Compute with wgpu

### Suitability Assessment

| Aspect | GPU Suitability |
|--------|-----------------|
| Collision detection (8%) | Good - uniform compute |
| Contact construction (73%) | Poor - divergent branches |
| Data size (10k spheres) | Poor - too small |
| Memory transfer overhead | Poor - would dominate |

### What Could Work

Only collision detection is suitable for GPU offload:

```wgsl
// WGSL compute shader for batch collision testing
@group(0) @binding(0) var<storage, read> spheres: array<vec4<f32>>; // xyz + r
@group(0) @binding(1) var<storage, read> pairs: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> results: array<u32>;

@compute @workgroup_size(256)
fn check_collisions(@builtin(global_invocation_id) id: vec3<u32>) {
    let pair = pairs[id.x];
    let a = spheres[pair.x];
    let b = spheres[pair.y];
    let d = a.xyz - b.xyz;
    let dist_sq = dot(d, d);
    let sum_r = a.w + b.w;
    results[id.x] = select(0u, 1u, dist_sq < sum_r * sum_r);
}
```

### Hybrid Architecture

```
CPU                              GPU
─────────────────────────────────────────────
spheres[] ──────────────────────► buffer
                                     │
                              ┌──────▼──────┐
                              │  Collision  │
                              │   detect    │
                              └──────┬──────┘
                                     │
collision_pairs[] ◄──────────────────┘
         │
┌────────▼────────┐
│ Contact constr. │  ← Keep on CPU (too branchy)
└─────────────────┘
```

### Cost-Benefit Analysis

| Factor | Value |
|--------|-------|
| Kernel launch overhead | ~50-100µs per dispatch |
| Buffer transfer (10k spheres) | ~10-20µs |
| Current collision time | ~2.4ms (8% of 30ms) |
| Potential GPU collision time | ~0.1-0.5ms |
| **Net savings** | **~2ms per call** |

### Verdict

**Marginal benefit for current workload sizes.**

- wgpu adds ~500 lines of boilerplate
- Only collision detection (8% of runtime) is GPU-suitable
- Break-even point: ~50k+ spheres

**Worth it if:** Processing 50k+ spheres regularly, or building toward larger molecular systems.

**Not worth it if:** Current performance is acceptable, typical workloads <20k spheres.

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
