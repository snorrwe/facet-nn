use std::mem::MaybeUninit;

#[derive(thiserror::Error, Debug)]
pub enum NdArrayError {
    #[error("DimensionMismatch error, expected: {expected}, actual: {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

#[derive(Debug)]
pub struct NdArray<T> {
    dims: Box<[u32]>,
    values: Box<[T]>,
}

unsafe impl<T> Send for NdArray<T> {}

impl<T> NdArray<T>
where
    T: Copy,
{
    pub fn new(dims: Box<[u32]>) -> Self {
        let len: usize = dims.iter().map(|x| *x as usize).product();
        let values = (0..len)
            .map(|_| unsafe { MaybeUninit::uninit().assume_init() })
            .collect::<Vec<_>>();
        Self {
            dims,
            values: values.into_boxed_slice(),
        }
    }
}

impl<T> NdArray<T>
where
    T: Default,
{
    pub fn new_default(dims: Box<[u32]>) -> Self {
        let len: usize = dims.iter().map(|x| *x as usize).product();
        let values = (0..len).map(|_| Default::default()).collect::<Vec<_>>();
        Self {
            dims,
            values: values.into_boxed_slice(),
        }
    }
}

impl<T> NdArray<T> {
    /// If invalid returns false and leaves this instance unchanged
    pub fn set_slice(&mut self, values: Box<[T]>) -> Result<&mut Self, NdArrayError> {
        if values.len() != self.values.len() {
            return Err(NdArrayError::DimensionMismatch {
                expected: self.values.len(),
                actual: values.len(),
            });
        }

        self.values = values;

        Ok(self)
    }

    pub fn dims(&self) -> &[u32] {
        &self.dims
    }

    pub fn as_slice(&self) -> &[T] {
        &self.values
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.values
    }

    /// Returns `None` on invalid index
    pub fn get(&self, index: &[u32]) -> Option<&T> {
        let i = get_index(1, self.dims(), index)?;
        self.values.get(i)
    }

    pub fn get_mut(&mut self, index: &[u32]) -> Option<&mut T> {
        let i = get_index(1, self.dims(), index)?;
        self.values.get_mut(i)
    }

    /// Index must be N-1 long. Will return a vector
    ///
    /// E.g for a 1D array the column is the array itself
    /// for a 2D array, with dims (2, 5) the column is a 5D vector
    ///
    /// Returns `None` on invalid index.
    ///
    /// ```
    /// use nd::ndarray::NdArray;
    ///
    /// let mut arr = NdArray::<i32>::new([4, 2, 3].into());
    /// arr.as_mut_slice()
    ///     .iter_mut()
    ///     .enumerate()
    ///     .for_each(|(i, x)| *x = i as i32);
    ///
    /// let col = arr.get_column(&[1, 1]).unwrap();
    ///
    /// assert_eq!(col, &[9, 10, 11]);
    /// ```
    pub fn get_column(&self, index: &[u32]) -> Option<&[T]> {
        match self.dims.len() {
            0 => None,
            1 => Some(self.as_slice()),
            _ => {
                let w = *self.dims.last().unwrap() as usize;
                let i = get_index(
                    w,
                    &self.dims[..self.dims.len() - 1], // skip the last dim
                    index,
                )?;
                let i = i * w;
                Some(&self.as_slice()[i..i + w])
            }
        }
    }

    pub fn get_column_mut(&mut self, index: &[u32]) -> Option<&mut [T]> {
        match self.dims.len() {
            0 => None,
            1 => Some(self.as_mut_slice()),
            _ => {
                let w = *self.dims.last().unwrap() as usize;
                let i = get_index(
                    w,
                    &self.dims[..self.dims.len() - 1], // skip the last dim
                    index,
                )?;
                let i = i * w;
                Some(&mut self.as_mut_slice()[i..i + w])
            }
        }
    }

    pub fn iter_cols(&self) -> impl Iterator<Item = &[T]> {
        ColumnIterator::new(
            &self.values,
            self.dims.last().cloned().unwrap_or(0) as usize,
        )
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct ColumnIterator<'a, T> {
    arr: &'a [T],
    column_size: usize,
    current: usize,
}

impl<'a, T> ColumnIterator<'a, T> {
    pub fn new(arr: &'a [T], column_size: usize) -> Self {
        Self {
            arr,
            column_size,
            current: 0,
        }
    }
}

impl<'a, T> Iterator for ColumnIterator<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        let tail = self.current + self.column_size;
        if tail <= self.arr.len() {
            let res = &self.arr[self.current..tail];
            self.current = tail;
            Some(res)
        } else {
            None
        }
    }
}

/// fold the index
#[inline]
fn get_index(width: usize, dims: &[u32], index: &[u32]) -> Option<usize> {
    // they should be equal to avoid confusion, but this is enough to work
    let mut it = dims.iter().rev();
    let mut a = *it.next()? as usize;

    let mut res = *index.last()? as usize;
    let iit = index.iter().rev();

    for (dim, ind) in it.zip(iit).skip(1) {
        a *= *dim as usize;
        res += a * *ind as usize;
    }
    Some(res * width)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nd_index() {
        let i = get_index(1, &[2, 4, 8], &[1, 3, 5]).unwrap();

        assert_eq!(i, 5 + 2 * 8 + 1 * 4 * 8);
    }

    #[test]
    fn get_column() {
        let mut arr = NdArray::<i32>::new([4, 2, 3].into());
        arr.as_mut_slice()
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = i as i32);

        let col = arr.get_column(&[1, 1]).unwrap();

        assert_eq!(col, &[9, 10, 11]);
    }

    #[test]
    fn test_slice_frees_correctly() {
        let mut arr = NdArray::new([5, 5].into());

        arr.set_slice(vec![69u32; 25].into_boxed_slice()).unwrap();

        for val in arr.as_slice() {
            assert_eq!(*val, 69);
        }
    }

    #[test]
    fn test_iter_cols() {
        let mut arr = NdArray::new([5, 8].into());
        arr.set_slice((0..40).collect::<Vec<_>>().into_boxed_slice())
            .unwrap();

        let mut count = 0;
        for (i, col) in arr.iter_cols().enumerate() {
            count = i + 1;
            assert_eq!(col.len(), 8, "index {}", i);
            let i = i * 8;
            assert_eq!(
                col,
                (i..i + 8).collect::<Vec<_>>().as_slice(),
                "index {}",
                i
            );
        }
        assert_eq!(count, 5);
    }
}
