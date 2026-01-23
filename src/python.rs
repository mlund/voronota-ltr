// Copyright (c) 2026 Kliment Olechnovic and Mikael Lund
// Part of the voronota-ltr project, licensed under the MIT License.
// SPDX-License-Identifier: MIT

//! Python bindings for voronota-ltr using `PyO3`.

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::{Ball, PeriodicBox, Results, compute_tessellation as compute_tessellation_rs};

/// Extract a required key from a Python dict.
fn extract_key<'py, T: FromPyObject<'py>>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<T> {
    dict.get_item(key)?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(format!("missing '{key}'")))?
        .extract()
}

/// Extract ball from a 4-element sequence (tuple or list).
fn ball_from_sequence(seq: &Bound<'_, PyAny>, type_name: &str) -> PyResult<Ball> {
    let len = seq.len()?;
    if len != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "ball {type_name} must have 4 elements (x, y, z, r)"
        )));
    }
    Ok(Ball::new(
        seq.get_item(0)?.extract()?,
        seq.get_item(1)?.extract()?,
        seq.get_item(2)?.extract()?,
        seq.get_item(3)?.extract()?,
    ))
}

/// Parse a single ball from tuple `(x, y, z, r)`, list `[x, y, z, r]`, or dict `{x, y, z, r}`.
fn parse_single_ball(obj: &Bound<'_, PyAny>) -> PyResult<Ball> {
    // Dict checked first since it's the most explicit format
    if let Ok(dict) = obj.downcast::<PyDict>() {
        return Ok(Ball::new(
            extract_key(dict, "x")?,
            extract_key(dict, "y")?,
            extract_key(dict, "z")?,
            extract_key(dict, "r")?,
        ));
    }

    // Tuple and list share the same indexing interface
    if obj.downcast::<PyTuple>().is_ok() {
        return ball_from_sequence(obj, "tuple");
    }
    if obj.downcast::<PyList>().is_ok() {
        return ball_from_sequence(obj, "list");
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "ball must be tuple (x,y,z,r), list [x,y,z,r], or dict {x,y,z,r}",
    ))
}

/// Parse balls from list of tuples/dicts or numpy array (N x 4).
fn parse_balls(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Ball>> {
    // Try numpy array first (N x 4)
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<f64>>() {
        let shape = arr.shape();
        if shape[1] != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "numpy array must have shape (N, 4) for (x, y, z, r)",
            ));
        }
        return Ok(arr
            .as_array()
            .rows()
            .into_iter()
            .map(|row| Ball::new(row[0], row[1], row[2], row[3]))
            .collect());
    }

    // Otherwise iterate as list
    let list = obj.downcast::<PyList>()?;
    list.iter().map(|item| parse_single_ball(&item)).collect()
}

/// Parse periodic box from dict with either "corners" or "vectors" key.
fn parse_periodic_box(obj: &Bound<'_, PyAny>) -> PyResult<PeriodicBox> {
    let dict = obj.downcast::<PyDict>()?;

    // Option 1: {"corners": [(x1,y1,z1), (x2,y2,z2)]}
    if let Some(corners) = dict.get_item("corners")? {
        let c: Vec<(f64, f64, f64)> = corners.extract()?;
        if c.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "corners must have exactly 2 points",
            ));
        }
        return Ok(PeriodicBox::from_corners(c[0], c[1]));
    }

    // Option 2: {"vectors": [(ax,ay,az), (bx,by,bz), (cx,cy,cz)]}
    if let Some(vectors) = dict.get_item("vectors")? {
        let v: Vec<(f64, f64, f64)> = vectors.extract()?;
        if v.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "vectors must have exactly 3 direction vectors",
            ));
        }
        return Ok(PeriodicBox::from_vectors(v[0], v[1], v[2]));
    }

    Err(pyo3::exceptions::PyValueError::new_err(
        "periodic_box must have 'corners' or 'vectors' key",
    ))
}

/// Convert contacts to list of dicts.
fn contacts_to_list<'py>(
    py: Python<'py>,
    contacts: &[crate::Contact],
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for c in contacts {
        let dict = PyDict::new(py);
        dict.set_item("id_a", c.id_a)?;
        dict.set_item("id_b", c.id_b)?;
        dict.set_item("area", c.area)?;
        dict.set_item("arc_length", c.arc_length)?;
        list.append(dict)?;
    }
    Ok(list)
}

/// Convert cells to list of dicts.
fn cells_to_list<'py>(py: Python<'py>, cells: &[crate::Cell]) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for c in cells {
        let dict = PyDict::new(py);
        dict.set_item("index", c.index)?;
        dict.set_item("sas_area", c.sas_area)?;
        dict.set_item("volume", c.volume)?;
        list.append(dict)?;
    }
    Ok(list)
}

/// Convert cell vertices to list of dicts.
#[allow(clippy::cast_possible_wrap)] // Array indices won't overflow i64
fn vertices_to_list<'py>(
    py: Python<'py>,
    vertices: &[crate::CellVertex],
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for v in vertices {
        let dict = PyDict::new(py);
        let indices = PyList::new(py, v.ball_indices.iter().map(|&i| i.map(|x| x as i64)))?;
        dict.set_item("ball_indices", indices)?;
        dict.set_item("x", v.x)?;
        dict.set_item("y", v.y)?;
        dict.set_item("z", v.z)?;
        dict.set_item("is_on_sas", v.is_on_sas())?;
        list.append(dict)?;
    }
    Ok(list)
}

/// Convert cell edges to list of dicts.
#[allow(clippy::cast_possible_wrap)] // Array indices won't overflow i64
fn edges_to_list<'py>(py: Python<'py>, edges: &[crate::CellEdge]) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for e in edges {
        let dict = PyDict::new(py);
        let indices = PyList::new(py, e.ball_indices.iter().map(|&i| i.map(|x| x as i64)))?;
        dict.set_item("ball_indices", indices)?;
        dict.set_item("length", e.length)?;
        dict.set_item("is_on_sas", e.is_on_sas())?;
        list.append(dict)?;
    }
    Ok(list)
}

/// Convert `TessellationResult` to Python dict.
fn result_to_dict<'py>(
    py: Python<'py>,
    result: &crate::TessellationResult,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("num_balls", result.num_balls)?;
    dict.set_item("contacts", contacts_to_list(py, &result.contacts)?)?;
    dict.set_item("cells", cells_to_list(py, &result.cells)?)?;
    dict.set_item("total_sas_area", result.total_sas_area())?;
    dict.set_item("total_volume", result.total_volume())?;
    dict.set_item("total_contact_area", result.total_contact_area())?;

    // Optional vertex/edge output
    if let Some(vertices) = &result.cell_vertices {
        dict.set_item("cell_vertices", vertices_to_list(py, vertices)?)?;
    }
    if let Some(edges) = &result.cell_edges {
        dict.set_item("cell_edges", edges_to_list(py, edges)?)?;
    }

    Ok(dict)
}

/// Compute radical Voronoi tessellation of spheres.
///
/// # Arguments
///
/// * `balls` - Input spheres as:
///   - List of tuples: `[(x, y, z, r), ...]`
///   - List of dicts: `[{"x": 0, "y": 0, "z": 0, "r": 1.5}, ...]`
///   - `NumPy` array: `np.array([[x, y, z, r], ...])`
/// * `probe` - Probe radius for solvent-accessible surface (typically 1.4 for water)
/// * `periodic_box` - Optional periodic boundary conditions as dict:
///   - `{"corners": [(x1,y1,z1), (x2,y2,z2)]}` for orthorhombic box
///   - `{"vectors": [(ax,ay,az), (bx,by,bz), (cx,cy,cz)]}` for triclinic cell
/// * `groups` - Optional list of group IDs for filtering inter-group contacts
/// * `with_cell_vertices` - If True, include tessellation vertices and edges in output
///
/// # Returns
///
/// Dict containing:
/// * `num_balls` - Number of input spheres
/// * `contacts` - List of contact dicts with `id_a`, `id_b`, `area`, `arc_length`
/// * `cells` - List of cell dicts with `index`, `sas_area`, `volume`
/// * `total_sas_area` - Total solvent-accessible surface area
/// * `total_volume` - Total volume
/// * `total_contact_area` - Total contact area
/// * `cell_vertices` - (optional) List of vertex dicts if `with_cell_vertices=True`
/// * `cell_edges` - (optional) List of edge dicts if `with_cell_vertices=True`
#[pyfunction]
#[pyo3(signature = (balls, probe, periodic_box=None, groups=None, with_cell_vertices=false))]
#[allow(clippy::needless_pass_by_value)] // PyO3 requires owned values for extraction
fn compute_tessellation<'py>(
    py: Python<'py>,
    balls: &Bound<'_, PyAny>,
    probe: f64,
    periodic_box: Option<&Bound<'_, PyAny>>,
    groups: Option<Vec<i32>>,
    with_cell_vertices: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let balls = parse_balls(balls)?;
    let pbox = periodic_box.map(parse_periodic_box).transpose()?;

    // Release GIL during computation
    let result = py.allow_threads(|| {
        compute_tessellation_rs(
            &balls,
            probe,
            pbox.as_ref(),
            groups.as_deref(),
            with_cell_vertices,
        )
    });

    result_to_dict(py, &result)
}

/// Python module definition.
#[pymodule]
fn voronota_ltr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_tessellation, m)?)?;
    Ok(())
}
