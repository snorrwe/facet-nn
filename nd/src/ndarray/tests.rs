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

#[test]
fn test_vector_inner() {
    let mut a = NdArray::new([8].into());
    a.set_slice(vec![69; 8].into()).unwrap();

    let mut b = NdArray::new([8].into());
    b.set_slice(vec![69; 8].into()).unwrap();

    let c = a.inner(&b);

    assert_eq!(c, Some((69i32).pow(2) * 8));
}

#[test]
fn test_mat_mat_inner() {
    let mut a = NdArray::new([3, 3].into());
    a.set_slice(vec![42; 9].into()).unwrap();

    let mut b = NdArray::new([3, 3].into());
    b.set_slice(vec![69; 9].into()).unwrap();

    let c = a.inner(&b).unwrap();

    assert_eq!(c, ((42 * 69) * 9));
}

#[test]
fn test_nd_nd_inner() {
    let mut a = NdArray::new([2, 3, 2].into());
    a.set_slice(vec![42; 12].into()).unwrap();

    let mut b = NdArray::new([3, 2, 2].into());
    b.set_slice(vec![69; 12].into()).unwrap();

    let c = a.inner(&b).unwrap();

    assert_eq!(c, ((42 * 69) * 12));
}

#[test]
fn test_vector_matrix_mul() {
    // transpose matrix, displacing the 3d homogeneous vector by 5,5,5
    #[rustfmt::skip]
    fn mat() -> Box<[i32]> {
        [1, 0, 0, 0, 
         0, 1, 0, 0,
         0, 0, 1, 0,
         5, 5, 5, 1].into()
    }

    let a = NdArray::new_with_values([4].into(), [1, 2, 3, 1].into()).unwrap();
    let b = NdArray::new_with_values([4, 4].into(), mat()).unwrap();

    let c = a.matmul(&b).expect("matmul");

    assert_eq!(c.shape, Shape::Vector(4));
    assert_eq!(c.as_slice(), &[6, 7, 8, 1]);
}

#[test]
fn test_vector_matrix_mul_w_broadcasting() {
    // transpose matrix, displacing the 3d homogeneous vector by 5,5,5
    #[rustfmt::skip]
    fn mat() -> Box<[i32]> {
        [1, 0, 0, 0, 
         0, 1, 0, 0,
         0, 0, 1, 0,
         5, 5, 5, 1].into()
    }
    let mat = mat();

    let a = NdArray::new_with_values([4].into(), [1, 2, 3, 1].into()).unwrap();
    let b = NdArray::new_with_values(
        [4, 4, 4].into(),
        (0..4)
            .flat_map(|_| mat.iter().cloned())
            .collect::<Vec<_>>()
            .into(),
    )
    .unwrap();

    let c = a.matmul(&b).expect("matmul");

    println!("{:?}", c);
    assert_eq!(c.shape, Shape::Matrix(4, 4));
    for col in c.iter_cols() {
        assert_eq!(col, &[6, 7, 8, 1]);
    }
}

#[test]
fn test_matrix_vector_mul() {
    // transpose matrix, displacing the 3d homogeneous vector by 5,5,5
    #[rustfmt::skip]
    fn mat() -> Box<[i32]> {
        [1, 0, 0, 5, 
         0, 1, 0, 5,
         0, 0, 1, 5,
         0, 0, 0, 1].into()
    }

    let a = NdArray::new_with_values([4, 4].into(), mat()).unwrap();
    let b = NdArray::new_with_values([4].into(), [1, 2, 3, 1].into()).unwrap();

    let c = a.matmul(&b).expect("matmul");

    assert_eq!(c.shape, Shape::Vector(4));
    assert_eq!(c.as_slice(), &[6, 7, 8, 1]);
}

#[test]
fn test_mat_mat_mul() {
    let a = NdArray::new_with_values([2, 3].into(), [1, 2, -1, 2, 0, 1].into()).unwrap();
    let b = NdArray::new_with_values([3, 2].into(), [3, 1, 0, -1, -2, 3].into()).unwrap();

    let c = a.matmul(&b).expect("matmul");

    assert_eq!(c.shape, Shape::Matrix(2, 2));
    assert_eq!(c.as_slice(), &[5, -4, 4, 5]);
}

#[test]
fn test_mat_mat_mul_many() {
    let a = NdArray::new_with_values([2, 3].into(), [1, 2, -1, 2, 0, 1].into()).unwrap();

    // 2 times the matrix from above
    let b = NdArray::new_with_values(
        [2, 3, 2].into(),
        [3, 1, 0, -1, -2, 3, 3, 1, 0, -1, -2, 3].into(),
    )
    .unwrap();

    let c = a.matmul(&b).expect("matmul");

    assert_eq!(c.shape, Shape::Tensor([2, 2, 2].into()));
    assert_eq!(c.as_slice(), &[5, -4, 4, 5, 5, -4, 4, 5]);
}
