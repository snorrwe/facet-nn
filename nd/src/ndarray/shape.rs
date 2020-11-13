#[derive(Debug, Clone)]
pub enum NdArrayShape {
    Scalar,
    Vector(u32),
    Matrix(u32, u32),
    Nd(Box<[u32]>),
}

impl NdArrayShape {
    pub fn last(&self) -> Option<u32> {
        match self {
            NdArrayShape::Scalar => None,
            NdArrayShape::Matrix(_, n) | NdArrayShape::Vector(n) => Some(*n),
            NdArrayShape::Nd(ref s) => s.last().cloned(),
        }
    }
}
