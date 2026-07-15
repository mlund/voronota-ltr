# Changelog

## 0.7.0 - Unreleased

### Changed

- Breaking Rust API: `Results::sas_areas()` and `Results::volumes()` now return
  `Vec<CellMeasure>` instead of `Vec<Option<f64>>`.
- Python tessellation results now include dense `cell_states`, `sas_areas`, and `volumes` lists.
- The Python module exposes its package version as `voronota_ltr.__version__`.
- Implementations of the public `Results` trait must classify their per-ball measures explicitly.
- `TessellationResult` carries private completeness state and can no longer be constructed with an
  external struct literal; obtain it from the compute APIs.

### Fixed

- Hidden or contained balls are no longer mistaken for detached balls with full-sphere measures.
- Missing-cell full-sphere fallbacks can no longer inflate molecular volumes as the probe radius
  grows.
- Geometrically empty cells are represented as zero rather than an ambiguous missing value.
- Group-filtered cell measures report `NotComputed`; this state survives serde round trips.
- Python list input works without NumPy installed. NumPy remains optional unless arrays are used.

### Compatibility

- The CLI JSON schema remains unchanged: its `sas_areas` and `volumes` arrays still contain a
  float or `null`. Rust and Python callers should use the explicit cell states when the distinction
  between an empty and an unavailable cell matters.
- Serde can read older `TessellationResult` data that lacks the new completeness marker.

See [Cell measures in 0.7](docs/cell-measures.md) for migration examples.
