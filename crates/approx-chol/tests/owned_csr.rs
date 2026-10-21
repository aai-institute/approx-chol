use approx_chol::{factorize, CsrRef, OwnedCsr};

#[test]
fn owned_csr_routes_factor_alike() {
    let (rp, ci, v) = (
        [0usize, 2, 5, 7],
        [0usize, 1, 0, 1, 2, 1, 2],
        [1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0],
    );
    let owned: OwnedCsr = OwnedCsr::try_from_usize(&rp, &ci, &v, 3).expect("shape");
    let by_input = factorize(&owned).expect("factor via From<&OwnedCsr> for Input");
    let by_ref = factorize(owned.as_csr_ref()).expect("factor via as_csr_ref");
    let wide = CsrRef::new(&rp, &ci, &v, 3).expect("usize view");
    let narrowed = wide.to_owned_u32().expect("narrow");
    let by_narrow = factorize(&narrowed).expect("factor narrowed");
    let b = [1.0, 0.0, -1.0];
    let x = |f: &approx_chol::Factor| f.solve(&b).expect("solve");
    assert_eq!(x(&by_input), x(&by_ref));
    assert_eq!(x(&by_input), x(&by_narrow));
}
