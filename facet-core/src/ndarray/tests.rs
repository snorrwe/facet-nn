use super::*;

#[test]
fn nd_index() {
    let i = get_index(&[2, 4, 8], &[4 * 8, 8, 1], &[1, 3, 5]).unwrap();
    assert_eq!(i, 5 + 3 * 8 + 4 * 8);
}

#[test]
fn get_column() {
    let mut arr = NdArray::<i32>::new(&[4, 2, 3][..]);
    arr.as_mut_slice()
        .iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x = i as i32);

    let col = arr.get_row(&[1, 1]).unwrap();

    assert_eq!(col, &[9, 10, 11]);
}

#[test]
fn test_slice_frees_correctly() {
    let mut arr = NdArray::new([5, 5]);

    arr.set_slice(vec![69u32; 25].into()).unwrap();

    for val in arr.as_slice() {
        assert_eq!(*val, 69);
    }
}

#[test]
fn test_iter_rows() {
    let mut arr = NdArray::new([5, 8]);
    arr.set_slice((0..40).collect::<Data<_>>()).unwrap();

    let mut count = 0;
    for (i, col) in arr.iter_rows().enumerate() {
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
    let a = NdArray::new_vector(vec![69; 8]);
    let b = NdArray::new_vector(vec![69; 8]);

    let c = a.inner(&b);

    assert_eq!(c, Some((69i32).pow(2) * 8));
}

#[test]
fn test_mat_mat_inner() {
    let mut a = NdArray::new([3, 3]);
    a.set_slice(vec![42; 9].into()).unwrap();

    let mut b = NdArray::new([3, 3]);
    b.set_slice(vec![69; 9].into()).unwrap();

    let c = a.inner(&b).unwrap();

    assert_eq!(c, ((42 * 69) * 9));
}

#[test]
fn test_nd_nd_inner() {
    let mut a = NdArray::new(&[2, 3, 2][..]);
    a.set_slice(vec![42; 12].into()).unwrap();

    let mut b = NdArray::new(&[3, 2, 2][..]);
    b.set_slice(vec![69; 12].into()).unwrap();

    let c = a.inner(&b).unwrap();

    assert_eq!(c, ((42 * 69) * 12));
}

#[test]
fn test_vector_matrix_mul() {
    // transpose matrix, displacing the 3d homogeneous vector by 5,5,5
    #[rustfmt::skip]
    fn mat() -> Data<i32> {
        [1, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1, 0,
         5, 5, 5, 1].into()
    }

    let a = NdArray::new_with_values(&[4][..], Data::from_slice(&[1, 2, 3, 1][..])).unwrap();
    let b = NdArray::new_with_values(&[4, 4][..], mat()).unwrap();

    let mut c = NdArray::new(0);
    a.matmul(&b, &mut c).expect("matmul");

    assert_eq!(c.shape, Shape::Vector([4]));
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

    let a = NdArray::new_with_values(&[4][..], Data::from_slice(&[1, 2, 3, 1])).unwrap();
    let b = NdArray::new_with_values(
        &[4, 4, 4][..],
        (0..4)
            .flat_map(|_| mat.iter().cloned())
            .collect::<Data<_>>(),
    )
    .unwrap();

    let mut c = NdArray::new(0);
    a.matmul(&b, &mut c).expect("matmul");

    println!("{:?}", c);
    assert_eq!(c.shape, Shape::Matrix([4, 4]));
    for col in c.iter_rows() {
        assert_eq!(col, &[6, 7, 8, 1]);
    }
}

#[test]
fn test_matrix_vector_mul() {
    // transpose matrix, displacing the 3d homogeneous vector by 5,5,5
    #[rustfmt::skip]
    fn mat() -> Data<i32> {
        [1, 0, 0, 5,
         0, 1, 0, 5,
         0, 0, 1, 5,
         0, 0, 0, 1].into()
    }

    let a = NdArray::new_with_values(&[4, 4][..], mat()).unwrap();
    let b = NdArray::new_with_values(4u32, Data::from_slice(&[1, 2, 3, 1])).unwrap();

    let mut c = NdArray::new(0);
    a.matmul(&b, &mut c).expect("matmul");

    assert_eq!(c.shape, Shape::Vector([4]));
    assert_eq!(c.as_slice(), &[6, 7, 8, 1]);
}

#[test]
fn test_mat_mat_mul() {
    let a = NdArray::new_with_values([2, 3], Data::from_slice(&[1, 2, -1, 2, 0, 1])).unwrap();
    let b = NdArray::new_with_values([3, 2], Data::from_slice(&[3, 1, 0, -1, -2, 3])).unwrap();

    let mut c = NdArray::new(0);
    a.matmul(&b, &mut c).expect("matmul");

    assert_eq!(c.shape, Shape::Matrix([2, 2]));
    assert_eq!(c.as_slice(), &[5, -4, 4, 5]);
}

#[test]
fn test_mat_mat_mul_f64() {
    let a = NdArray::new_with_values([2, 3], Data::from_slice(&[1.0, 2.0, -1.0, 2.0, 0.0, 1.0]))
        .unwrap();
    let b = NdArray::new_with_values([3, 2], Data::from_slice(&[3.0, 1.0, 0.0, -1.0, -2.0, 3.0]))
        .unwrap();

    let mut c = NdArray::new(0);
    a.matmul(&b, &mut c).expect("matmul");
    let mut d = NdArray::new(0);
    a.matmul_f32(&b, &mut d).expect("matmul f64");

    assert_eq!(c.shape, Shape::Matrix([2, 2]));
    assert_eq!(d.shape, Shape::Matrix([2, 2]));
    assert_eq!(c.as_slice(), &[5.0, -4.0, 4.0, 5.0]);
    assert_eq!(c.as_slice(), d.as_slice());
}

/// regression test
#[test]
fn test_mat_mat_mul_f64_multi_call() {
    let a = NdArray::new_with_values([2, 3], Data::from_slice(&[1.0, 2.0, -1.0, 2.0, 0.0, 1.0]))
        .unwrap();
    let b = NdArray::new_with_values([3, 2], Data::from_slice(&[3.0, 1.0, 0.0, -1.0, -2.0, 3.0]))
        .unwrap();

    let mut c = NdArray::new(0);
    a.matmul_f32(&b, &mut c).expect("matmul");
    a.matmul_f32(&b, &mut c).expect("matmul");
    a.matmul_f32(&b, &mut c).expect("matmul");
    a.matmul_f32(&b, &mut c).expect("matmul");
    a.matmul_f32(&b, &mut c).expect("matmul");

    assert_eq!(c.shape, Shape::Matrix([2, 2]));
    assert_eq!(c.as_slice(), &[5.0, -4.0, 4.0, 5.0]);
}

#[test]
fn test_mat_mat_mul_many() {
    let a = NdArray::new_with_values([2, 3], Data::from_slice(&[1, 2, -1, 2, 0, 1])).unwrap();

    // 2 times the matrix from above
    let b = NdArray::new_with_values(
        &[2, 3, 2][..],
        Data::from_slice(&[3, 1, 0, -1, -2, 3, 3, 1, 0, -1, -2, 3]),
    )
    .unwrap();

    let mut c = NdArray::new(0);
    a.matmul(&b, &mut c).expect("matmul");

    assert_eq!(c.shape, Shape::Tensor(SmallVec::from_slice(&[2, 2, 2])));
    assert_eq!(c.as_slice(), &[5, -4, 4, 5, 5, -4, 4, 5]);
}

#[test]
fn test_mat_transpose() {
    let a = NdArray::new_with_values(&[2, 3][..], Data::from_slice(&[1, 2, 3, 4, 5, 6])).unwrap();

    println!("{}", a.to_string());
    let b = a.transpose();
    println!("{}", b.to_string());

    assert_eq!(b.shape, Shape::Matrix([3, 2]));
    assert_eq!(b.as_slice(), &[1, 4, 2, 5, 3, 6], "{}", b.to_string());
}

#[test]
fn test_tensor_transpose() {
    let a = NdArray::new_with_values(
        &[4, 2, 3][..],
        smallvec::smallvec![
            1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6,
        ],
    )
    .unwrap();

    println!("{}", a.to_string());
    let b = a.transpose();

    assert_eq!(b.shape, Shape::Tensor(SmallVec::from_slice(&[4, 3, 2])));
    assert_eq!(
        b.as_slice(),
        &[1, 4, 2, 5, 3, 6, 1, 4, 2, 5, 3, 6, 1, 4, 2, 5, 3, 6, 1, 4, 2, 5, 3, 6],
        "\n{}",
        b.to_string()
    );
}

#[test]
fn test_horizontal_flip_even_rows() {
    let inp = [1, 2, 3, 4, 5, 6, 7, 8];
    let mut out = [0; 8];

    matrix::flip_mat_horizontal([2, 4], &inp, &mut out);
    assert_eq!(&out, &[5, 6, 7, 8, 1, 2, 3, 4][..]);
}

#[test]
fn test_horizontal_flip_odd_rows() {
    let inp = [1, 2, 3, 4, 5, 6];
    let mut out = [0; 6];

    matrix::flip_mat_horizontal([3, 2], &inp, &mut out);

    assert_eq!(&out, &[5, 6, 1, 2, 3, 4][..]);
}

#[test]
fn test_vertical_flip_even_rows() {
    let inp = [1, 2, 3, 4, 5, 6, 7, 8];
    let mut out = [0; 8];

    matrix::flip_mat_vertical([2, 4], &inp, &mut out);
    assert_eq!(&out, &[4, 3, 2, 1, 8, 7, 6, 5][..]);
}

#[test]
fn test_vertical_flip_odd_rows() {
    let inp = [1, 2, 3, 4, 5, 6];
    let mut out = [0; 6];

    matrix::flip_mat_vertical([3, 2], &inp, &mut out);

    assert_eq!(&out, &[2, 1, 4, 3, 6, 5][..]);
}

#[test]
fn test_vertical_flip_tensor() {
    let inp = NdArray::new_with_values(
        &[3, 2, 2][..],
        vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4].into(),
    )
    .unwrap();

    let out = inp.flip_mat_vertical().unwrap();
    assert_eq!(out.as_slice(), &[2, 1, 4, 3, 2, 1, 4, 3, 2, 1, 4, 3][..]);

    //sanity check
    let inp = [1, 2, 3, 4];
    let mut out = [0; 4];
    matrix::flip_mat_vertical([2, 2], &inp, &mut out);
    assert_eq!(&out, &[2, 1, 4, 3][..]);
}

#[test]
fn test_col_iter_returns_one_scalar() {
    let a = NdArray::new_scalar(42.69f32);

    let mut c = a.iter_rows();

    let s = c.next().expect("expected one");

    assert_eq!(s[0], 42.69);
    assert!(matches!(c.next(), None));
}

#[test]
fn test_iter_rows_returns_all_items_in_vec() {
    let a = NdArray::new_vector(vec![1, 2, 3, 4, 5, 6, 7, 8]);

    let count = a
        .iter_rows()
        .map(|actual| {
            assert_eq!(actual.len(), 8);
            assert_eq!(actual, &[1, 2, 3, 4, 5, 6, 7, 8]);
        })
        .count();

    assert_eq!(count, 1);
}
