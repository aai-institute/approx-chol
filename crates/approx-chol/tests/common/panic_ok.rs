pub trait OrPanic<T> {
    fn or_panic(self, context: &str) -> T;
}

impl<T, E: core::fmt::Debug> OrPanic<T> for Result<T, E> {
    fn or_panic(self, context: &str) -> T {
        match self {
            Ok(value) => value,
            Err(err) => panic!("{context}: {err:?}"),
        }
    }
}

impl<T> OrPanic<T> for Option<T> {
    fn or_panic(self, context: &str) -> T {
        match self {
            Some(value) => value,
            None => panic!("{context}"),
        }
    }
}
