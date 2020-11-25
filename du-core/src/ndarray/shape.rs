use std::ops::{Index, IndexMut};

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Shape {
    /// this value is meaningless, just helps us convert into slice
    Scalar([u32; 1]),
    Vector([u32; 1]),
    Matrix([u32; 2]),
    /// Over 3 dimensions
    Tensor(Box<[u32]>),
}

impl Shape {
    pub fn last(&self) -> Option<u32> {
        match self {
            Shape::Scalar(_) => None,
            Shape::Matrix([_, n]) => Some(*n),
            Shape::Vector(n) => Some(n[0]),
            Shape::Tensor(s) => s.last().cloned(),
        }
    }

    /// Return the last two dimensions. Effectively the shape of the matrices contained in the
    /// shape.
    pub fn last_two(&self) -> Option<[u32; 2]> {
        match self {
            Shape::Scalar(_) => None,
            Shape::Vector(_) => None,
            Shape::Matrix(s) => Some(s.clone()),
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
            Shape::Scalar(_) => 1,
            Shape::Vector([n]) => *n as usize,
            Shape::Matrix([n, m]) => *n as usize * *m as usize,
            Shape::Tensor(vals) => vals.iter().map(|x| *x as usize).product(),
        }
    }

    /// Number of columns spanned by this shape.
    pub fn col_span(&self) -> usize {
        match self {
            Shape::Scalar(_) => 1,
            Shape::Vector([n]) => *n as usize,
            Shape::Matrix([n, _]) => *n as usize,
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
            Shape::Scalar(_) => 1,
            Shape::Vector([i]) => *i as usize,
            Shape::Matrix([n, m]) => {
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

    pub fn as_slice(&self) -> &[u32] {
        match &self {
            Shape::Scalar(s) => &s[0..0], // empty slice
            Shape::Vector(s) => s,
            Shape::Matrix(s) => s,
            Shape::Tensor(s) => s,
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
        match shape {
            0 => Shape::Scalar([0]),
            _ => Shape::Vector([shape]),
        }
    }
}

impl From<[u32; 2]> for Shape {
    fn from([n, m]: [u32; 2]) -> Self {
        Shape::Matrix([n, m])
    }
}

impl<'a> From<&'a [u32]> for Shape {
    fn from(shape: &'a [u32]) -> Self {
        match shape.len() {
            0 => Shape::Scalar([0]),
            1 if shape[0] == 0 => Shape::Scalar([0]),
            1 => Shape::Vector([shape[0]]),
            2 => Shape::Matrix([shape[0], shape[1]]),
            _ => Shape::Tensor(shape.into()),
        }
    }
}

/// Vector with the stride of each element of each dimension
///
/// 0 long shapes will return [1], they span a single item
pub fn stride_vec(width: usize, shp: &[u32]) -> Vec<usize> {
    let len = shp.len();

    if len == 0 {
        return vec![1];
    }

    let mut res = Vec::with_capacity(len);
    for i in 0..len - 1 {
        res.push(shp[i + 1..].iter().map(|x| *x as usize * width).product());
    }
    res.push(width); // stride of the last dimension is always 1
    res
}

impl Index<usize> for Shape {
    type Output = u32;

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            Shape::Scalar(s) | Shape::Vector(s) => &s[0],
            Shape::Matrix(s) => &s[index],
            Shape::Tensor(s) => &s[index],
        }
    }
}

impl IndexMut<usize> for Shape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self {
            Shape::Scalar(s) | Shape::Vector(s) => &mut s[0],
            Shape::Matrix(s) => &mut s[index],
            Shape::Tensor(s) => &mut s[index],
        }
    }
}
