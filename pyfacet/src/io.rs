use crate::pyndarray::NdArrayD;
use facet_core::ndarray::{Data, NdArray};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict, wrap_pyfunction};

use std::{
    collections::HashMap,
    io::{BufRead, BufReader},
};

/// The column names in `labels` will be treated as row labels instead of data points.
///
/// Column anmes in `meta` will be treated as metadata and ignored.
///
/// Returns a dict with the loaded data.
///
/// Data and labels are loaded separately, but corresponding rows maintain their indices.
///
/// ```python
/// # obtain the label, row pairs after loading the dataset
/// for label, row in zip(dataset["labels"], dataset["data"].iter_cols()):
///     print(label, row)
/// ```
#[pyfunction(labels = "[].to_vec()", meta = "[].to_vec()", ignore_error = "false")]
pub fn load_csv<'a, 'py>(
    py: Python<'py>,
    fname: &'a str,
    labels: Vec<&'a str>,
    meta: Vec<&'a str>,
    ignore_error: bool,
) -> PyResult<&'py PyDict> {
    let mut column_indices: HashMap<String, usize> = HashMap::new();
    let mut label_columns = Vec::new();
    let mut meta_columns = Vec::new();
    let mut columns: Vec<String> = Vec::new();
    let mut data = Data::new();

    let f = std::fs::OpenOptions::new()
        .read(true)
        .create(false)
        .open(fname)
        .map_err(|err| {
            PyValueError::new_err(format!(
                "Failed to open file [{}] for reading  {:?}",
                fname, err
            ))
        })?;

    let mut lines = BufReader::new(f).lines();

    // TODO: allow no header
    match lines.next() {
        Some(Ok(head)) => {
            for (i, item) in head.split(',').enumerate() {
                columns.push(item.to_string());
                column_indices.insert(item.to_string(), i);
                if labels.iter().any(|l| *l == item) {
                    label_columns.push(i);
                }
                if meta.iter().any(|l| *l == item) {
                    meta_columns.push(i);
                }
            }
        }
        _ => {
            todo!("no data read")
        }
    }

    let mut labels = Vec::new();
    let mut rows = 0;
    while let Some(Ok(line)) = lines.next() {
        rows += 1;
        let mut skipit = label_columns.iter();
        let mut skipind = skipit.next();
        let mut rowlabels = Vec::new();
        for (i, item) in line.split(',').enumerate() {
            if Some(&i) == skipind {
                skipind = skipit.next();
                rowlabels.push(item.to_string());
            } else {
                match item.parse() {
                    Ok(d) => data.push(d),
                    Err(err) if !ignore_error => {
                        return Err(PyValueError::new_err(format!(
                            "Failed to parse data point in row: {}, col: {} item: {} error: {}",
                            rows, i, item, err
                        )));
                    }
                    _ => data.push(f32::NAN),
                }
            }
        }
        labels.push(rowlabels)
    }

    let data = NdArrayD {
        inner: NdArray::new_with_values(vec![rows as u32, (data.len() / rows) as u32], data)
            .map_err(|err| {
                PyValueError::new_err(format!("failed to build nd array of data {}", err))
            })?,
    };

    let res = PyDict::new(py);
    match label_columns.len() {
        1 => {
            res.set_item("labels", labels.into_iter().flatten().collect::<Vec<_>>())?;
        }
        _ => {
            res.set_item("labels", labels)?;
        }
    }
    res.set_item("columns", columns)?;
    res.set_item("data", Py::new(py, data)?)?;

    Ok(res)
}

pub fn setup_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_csv, m)?)?;
    Ok(())
}
