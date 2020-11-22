use std::{borrow::Cow, convert::TryInto};

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Shape {
    Scalar,
    Vector(u64),
    Matrix(u32, u32),
    /// Over 3 dimensions
    Tensor(Box<[u32]>),
}

impl Shape {
    pub fn last(&self) -> Option<u32> {
        match self {
            Shape::Scalar => None,
            Shape::Matrix(_, n) => Some(*n),
            Shape::Vector(n) => Some(*n).map(|n| n.try_into().unwrap()),
            Shape::Tensor(ref s) => s.last().cloned(),
        }
    }

    /// Return the last two dimensions. Effectively the shape of the matrices contained in the
    /// shape.
    pub fn last_two(&self) -> Option<[u32; 2]> {
        match self {
            Shape::Scalar => None,
            Shape::Vector(_) => None,
            Shape::Matrix(n, m) => Some([*n, *m]),
            Shape::Tensor(shp) => {
                let len = shp.len();
                debug_assert!(len >= 3);
                let [n, m] = [shp[len - 2], shp[len - 1]];
                Some([n, m])
            }
        }
    }

    /// Total number of elements in an array this shape spans.
    ///
    /// For NdArrays this is equal to the `len` of its value array
    pub fn span(&self) -> usize {
        match self {
            Shape::Scalar => 1,
            Shape::Vector(n) => *n as usize,
            Shape::Matrix(n, m) => *n as usize * *m as usize,
            Shape::Tensor(vals) => vals.iter().map(|x| *x as usize).product(),
        }
    }

    /// Number of columns spanned by this shape.
    pub fn col_span(&self) -> usize {
        match self {
            Shape::Scalar => 1,
            Shape::Vector(n) => *n as usize,
            Shape::Matrix(n, _) => *n as usize,
            Shape::Tensor(shp) => shp[..shp.len() - 2].iter().map(|x| *x as usize).product(),
        }
    }

    /// Number of elements the last `i` dimensions of this shape spans
    ///
    /// If the total number of dimensions is less than `i` then all are counted
    pub fn last_span(&self, i: u32) -> usize {
        if i == 0 {
            return 0;
        }
        match self {
            Shape::Scalar => 1,
            Shape::Vector(i) => *i as usize,
            Shape::Matrix(n, m) => {
                if i <= 1 {
                    *m as usize
                } else {
                    *n as usize * *m as usize
                }
            }
            Shape::Tensor(shp) => shp[shp.len() - (i as usize).min(shp.len())..]
                .iter()
                .map(|x| *x as usize)
                .product(),
        }
    }

    pub fn as_array(&self) -> Cow<Box<[u32]>> {
        match self {
            Shape::Scalar => Cow::Owned([].into()),
            Shape::Vector(n) => Cow::Owned([*n as u32].into()),
            Shape::Matrix(n, m) => Cow::Owned([*n, *m].into()),
            Shape::Tensor(t) => Cow::Borrowed(t),
        }
    }
}

impl From<Box<[u32]>> for Shape {
    fn from(shape: Box<[u32]>) -> Self {
        Shape::from(shape.as_ref())
    }
}

impl From<Vec<u32>> for Shape {
    fn from(shape: Vec<u32>) -> Self {
        Shape::from(shape.as_slice())
    }
}

impl From<u32> for Shape {
    fn from(shape: u32) -> Self {
        From::from(shape as u64)
    }
}

impl From<u64> for Shape {
    fn from(shape: u64) -> Self {
        match shape {
            0 => Shape::Scalar,
            _ => Shape::Vector(shape),
        }
    }
}

impl From<[u32; 2]> for Shape {
    fn from([n, m]: [u32; 2]) -> Self {
        Shape::Matrix(n, m)
    }
}

impl<'a> From<&'a [u32]> for Shape {
    fn from(shape: &'a [u32]) -> Self {
        match shape.len() {
            0 | 1 if shape[0] == 0 => Shape::Scalar,
            1 => Shape::Vector(shape[0] as u64),
            2 => Shape::Matrix(shape[0], shape[1]),
            _ => Shape::Tensor(shape.into()),
        }
    }
}
