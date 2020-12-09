/// Dense iterator in the innermost dimensions of an NdArray
///
pub struct ColumnIter<'a, T> {
    arr: &'a [T],
    column_size: usize,
    current: usize,
}

impl<'a, T> ColumnIter<'a, T> {
    pub fn new(arr: &'a [T], column_size: usize) -> Self {
        Self {
            arr,
            column_size,
            current: 0,
        }
    }
}

impl<'a, T> Iterator for ColumnIter<'a, T> {
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

pub struct ColumnIterMut<'a, T> {
    arr: &'a mut [T],
    column_size: usize,
    current: usize,
}

impl<'a, T> ColumnIterMut<'a, T> {
    pub fn new(arr: &'a mut [T], column_size: usize) -> Self {
        Self {
            arr,
            column_size,
            current: 0,
        }
    }
}

impl<'a, T> Iterator for ColumnIterMut<'a, T> {
    type Item = &'a mut [T];

    fn next(&mut self) -> Option<Self::Item> {
        let tail = self.current + self.column_size;
        if tail <= self.arr.len() {
            let res = unsafe { &mut *(&mut self.arr[self.current..tail] as *mut _) };
            self.current = tail;
            Some(res)
        } else {
            None
        }
    }
}
