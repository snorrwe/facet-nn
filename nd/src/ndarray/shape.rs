#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Shape {
    Scalar,
    Vector(u32),
    Matrix(u32, u32),
    /// Over 3 dimensions
    Nd(Box<[u32]>),
}

impl Shape {
    pub fn last(&self) -> Option<u32> {
        match self {
            Shape::Scalar => None,
            Shape::Matrix(_, n) | Shape::Vector(n) => Some(*n),
            Shape::Nd(ref s) => s.last().cloned(),
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
            Shape::Nd(vals) => vals.iter().map(|x| *x as usize).product(),
        }
    }
}

impl From<Box<[u32]>> for Shape {
    fn from(shape: Box<[u32]>) -> Self {
        match shape.len() {
            0 | 1 if shape[0] == 0 => Shape::Scalar,
            1 => Shape::Vector(shape[0]),
            2 => Shape::Matrix(shape[0], shape[1]),
            _ => Shape::Nd(shape),
        }
    }
}

impl From<Vec<u32>> for Shape {
    fn from(shape: Vec<u32>) -> Self {
        match shape.len() {
            0 | 1 if shape[0] == 0 => Shape::Scalar,
            1 => Shape::Vector(shape[0]),
            2 => Shape::Matrix(shape[0], shape[1]),
            _ => Shape::Nd(shape.into()),
        }
    }
}
