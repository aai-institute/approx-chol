use approx_chol::{factorize, CsrError, CsrRef, Error};

struct PanicIntoCsr;

impl<'a> From<PanicIntoCsr> for CsrRef<'a, f64, u32> {
    fn from(_: PanicIntoCsr) -> Self {
        panic!("boom during conversion");
    }
}

#[test]
fn factorize_catches_panicking_conversion() {
    let err =
        factorize::<f64, u32, _>(PanicIntoCsr).expect_err("panicking conversion must map to error");
    assert!(matches!(
        err,
        Error::InvalidCsr(CsrError::InputConversionPanicked)
    ));
}
