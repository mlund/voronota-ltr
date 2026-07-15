# Cell measures in 0.7

Voronota-LTR stores detailed cells sparsely but exposes solvent-accessible surface areas and
volumes densely, with one entry per input ball. Before 0.7, an absent sparse cell produced `None`.
That value could mean either that the weighted cell was geometrically empty or that it had not
been computed. It also forced callers to guess whether a detached full sphere should replace the
missing value.

Version 0.7 makes the distinction explicit:

| Rust state | Meaning | Numerical interpretation |
|---|---|---:|
| `CellMeasure::Computed(value)` | The cell was computed. A detached ball has its full-sphere measure. | `value` |
| `CellMeasure::Empty` | The weighted cell is geometrically empty. | `0.0` |
| `CellMeasure::NotComputed` | Contact filtering prevented a complete cell calculation. | unavailable |

## Rust migration

Code written for 0.6 commonly used a fallback:

```rust,ignore
let volumes: Vec<Option<f64>> = result.volumes();
let volume = volumes[i].unwrap_or(full_sphere_volume);
```

In 0.7, handle the geometry explicitly:

```rust
use voronota_ltr::{CellMeasure, Results};
# fn consume(_: f64) {}
# let balls = [voronota_ltr::Ball::new(0.0, 0.0, 0.0, 1.0)];
# let result = voronota_ltr::compute_tessellation(&balls, 1.4, None, None, false);

for measure in result.volumes() {
    match measure {
        CellMeasure::Computed(volume) => consume(volume),
        CellMeasure::Empty => consume(0.0),
        CellMeasure::NotComputed => return Err("cell volume was not computed"),
    }
}
# Ok::<(), &'static str>(())
```

Do not replace `Empty` with a free-sphere value. Detached balls are already returned as
`Computed(full_sphere_measure)`.

`TessellationResult` now owns private completeness state, so downstream crates should obtain it
from `compute_tessellation` or `UpdateableTessellation::summary` rather than construct it with a
struct literal.

Passing `groups` filters the contact network and makes cell measures incomplete. The dense Rust
API therefore returns `NotComputed` for the result. The sparse `cells` and the total methods remain
available, but their values are partial for grouped calculations.

## Python migration

Python results retain the existing sparse `cells` list and add three dense lists:

```python
states = result["cell_states"]
sas_areas = result["sas_areas"]
volumes = result["volumes"]
```

Each list has `result["num_balls"]` entries. `"computed"` carries a float, `"empty"` carries
`0.0`, and `"not_computed"` carries `None`. Use `cell_states` when a computed numerical zero must
be distinguished from an empty cell.

## Serde and CLI JSON

Serde-serialized `TessellationResult` values preserve whether their cell data is incomplete and
can still deserialize pre-0.7 values. The command-line JSON format is retained for compatibility;
its dense arrays use `null` whenever no cell record is included and do not encode why. Use the
Rust or Python API for explicit cell classification.
