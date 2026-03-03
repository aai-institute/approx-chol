pub trait ErrOrPanic<E> {
    fn err_or_panic(self, context: &str) -> E;
}

impl<T, E: core::fmt::Debug> ErrOrPanic<E> for Result<T, E> {
    fn err_or_panic(self, context: &str) -> E {
        match self {
            Ok(_) => panic!("{context}"),
            Err(err) => err,
        }
    }
}
