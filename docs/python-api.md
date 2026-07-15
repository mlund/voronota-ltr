# Python API

The `voronota_ltr` extension uses dictionaries and lists so it can be used without additional
Python packages. NumPy is optional and is only needed when passing an array.

```python
import voronota_ltr

print(voronota_ltr.__version__)
```

## `compute_tessellation`

```python
compute_tessellation(
    balls,
    probe,
    periodic_box=None,
    groups=None,
    with_cell_vertices=False,
)
```

`balls` may be a list of `(x, y, z, radius)` tuples, a list of four-element lists, a list of
`{"x", "y", "z", "r"}` dictionaries, or a NumPy array with shape `(N, 4)`.

`periodic_box` accepts either `{"corners": [(x1, y1, z1), (x2, y2, z2)]}` or
`{"vectors": [a, b, c]}`. `groups` must contain one integer group identifier per ball and
restricts contacts to different groups; a length mismatch raises `ValueError`.

The returned dictionary contains:

| Key | Type | Description |
|---|---|---|
| `num_balls` | `int` | Number of input balls. |
| `contacts` | `list[dict]` | Contact indices, area, arc length, and centrality. |
| `cells` | `list[dict]` | Sparse computed-cell records with index, SAS area, and volume. |
| `cell_states` | `list[str]` | Dense `computed`, `empty`, or `not_computed` state. |
| `sas_areas` | `list[float \| None]` | Dense SAS areas; empty is `0.0`, unavailable is `None`. |
| `volumes` | `list[float \| None]` | Dense volumes; empty is `0.0`, unavailable is `None`. |
| `total_sas_area` | `float` | Sum over sparse computed cells. |
| `total_volume` | `float` | Sum over sparse computed cells. |
| `total_contact_area` | `float` | Sum of reported contact areas. |
| `cell_vertices` | `list[dict]` | Present when `with_cell_vertices=True`. |
| `cell_edges` | `list[dict]` | Present when `with_cell_vertices=True`. |

Grouped calculations intentionally return `not_computed` dense measures because filtering removes
planes needed to determine complete cells. Their total SAS area and volume are partial sums.

## `compute_tessellation_from_file`

```python
compute_tessellation_from_file(
    path,
    probe,
    periodic_box=None,
    with_cell_vertices=False,
    group_selections=None,
)
```

This reads PDB, mmCIF, or XYZR input and returns the same dictionary as `compute_tessellation`.
`group_selections` accepts at least two VMD-like selection strings and reports inter-group contacts.

## `compute_solvent_spheres`

```python
compute_solvent_spheres(
    balls,
    probe,
    volume_probe=None,
    subdivision_depth=None,
    periodic_box=None,
)
```

The result is a list of dictionaries containing `x`, `y`, `z`, `radius`, `weight`, and
`parent_index`. `subdivision_depth` ranges from 0 to 4 and defaults to 2.
