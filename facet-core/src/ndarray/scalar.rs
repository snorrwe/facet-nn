use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use super::{Data, NdArray};

impl<T> Add<T> for NdArray<T>
where
    T: Add<T, Output = T> + Copy,
{
    type Output = Self;

    fn add(self, b: T) -> Self::Output {
        let values: Data<T> = self.values.iter().map(move |a| *a + b).collect();
        Self::new_with_values(self.shape.clone(), values).unwrap()
    }
}

impl<T> AddAssign<T> for NdArray<T>
where
    T: AddAssign<T> + Copy,
{
    fn add_assign(&mut self, rhs: T) {
        for a in self.values.iter_mut() {
            *a += rhs;
        }
    }
}

impl<T> Sub<T> for NdArray<T>
where
    T: Sub<T, Output = T> + Copy,
{
    type Output = Self;

    fn sub(self, b: T) -> Self::Output {
        let values: Data<T> = self.values.iter().map(move |a| *a - b).collect();
        Self::new_with_values(self.shape.clone(), values).unwrap()
    }
}

impl<T> SubAssign<T> for NdArray<T>
where
    T: SubAssign<T> + Copy,
{
    fn sub_assign(&mut self, rhs: T) {
        for a in self.values.iter_mut() {
            *a -= rhs;
        }
    }
}

impl<T> Mul<T> for NdArray<T>
where
    T: Mul<T, Output = T> + Copy,
{
    type Output = Self;

    fn mul(self, b: T) -> Self::Output {
        let values: Data<T> = self.values.iter().map(move |a| *a * b).collect();
        Self::new_with_values(self.shape.clone(), values).unwrap()
    }
}

impl<T> MulAssign<T> for NdArray<T>
where
    T: MulAssign<T> + Copy,
{
    fn mul_assign(&mut self, rhs: T) {
        for a in self.values.iter_mut() {
            *a *= rhs;
        }
    }
}

impl<T> Div<T> for NdArray<T>
where
    T: Div<T, Output = T> + Copy,
{
    type Output = Self;

    fn div(self, b: T) -> Self::Output {
        let values: Data<T> = self.values.iter().map(move |a| *a / b).collect();
        Self::new_with_values(self.shape, values).unwrap()
    }
}

impl<T> DivAssign<T> for NdArray<T>
where
    T: DivAssign<T> + Copy,
{
    fn div_assign(&mut self, rhs: T) {
        for a in self.values.iter_mut() {
            *a /= rhs;
        }
    }
}
