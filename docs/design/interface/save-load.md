# Save and load interface

The contract is `../contracts/save-load.md`; `ApproxInverse` is that of `interface/build.md`.

## Decision

```rust
#[cfg(feature = "serde")]
impl<T: Real> serde::Serialize for ApproxInverse<T> {}
#[cfg(feature = "serde")]
impl<'de, T: Real> serde::Deserialize<'de> for ApproxInverse<T> {}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub version: u32,
    pub scalar: Scalar,
}
impl Header {
    pub fn of<T: Real>() -> Header;
    pub fn expect<T: Real>(self) -> Result<(), HeaderMismatch>;
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scalar {
    F32,
    F64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeaderMismatch {
    pub found: Header,
    pub expected: Header,
}
```

- The saved form is the pair `(Header, B)`, an array in a self-describing format: `version`, a
  `u32`, and `scalar`, `"f32"` or `"f64"`, then $B$. postcard writes the same bytes as the flat
  `(version, scalar, B)`. The configuration is not in the form.
- `Header::of::<T>()` is the current header, format version `0x4143_000E` and `T`'s scalar; it
  replaces `FACTOR_FORMAT_VERSION`. `expect::<T>` is the one check: `HeaderMismatch` unless the
  header equals `of::<T>()`; its `Display` names what differs, found and expected. A load reads
  the `Header` first and checks it before it reads $B$.
- A rejection of the header or of an invariant of $B$ goes through `D::Error::custom`, with that
  message; a decoding error is the format's own. A format that drops custom messages, such as
  postcard, keeps only the error. The same $B$ bit for bit needs a format that round-trips floats
  exactly: serde_json does with its `float_roundtrip` feature only.

Python:

```python
class ApproxInverse(LinearOperator):
    def __reduce__(self) -> tuple[object, ...]: ...
```

An `ApproxInverse` pickles as its saved form, encoded with postcard, and its held budget. A load
reads the `Header` itself, checks it with `expect`, and raises `ValueError` with the
`HeaderMismatch` message; a form that then fails to decode or validate raises `ValueError` as
invalid.

## Why

- **A pair, the header first:** a struct is a map in a self-describing format, so header-first
  reading needs a sequence there; a positional format, such as postcard, decodes the rest by
  position, and a mismatch checked first fails as itself, not as garbage further on.
- **`Header` a public type:** Rust and the Python binding read the same two fields; one type
  owns them and their check, where a bare version constant left the scalar to each reader.
- **No error type for the load:** serde returns the format's error, so a typed error would reach
  the caller only as its message; `HeaderMismatch` serves a caller that reads the header itself.
- **No configuration in the form:** the solve does not read it; a caller who rebuilds records it
  apart, as `../contracts/build.md` states.
- **postcard for pickle:** its wire format is specified and stable since 1.0, and floats take
  their width, 8 bytes per `f64`; MessagePack and CBOR add a byte per float. A pickle crosses
  releases inside joblib's files, so the encoding is part of the form.
- **Python reads the header itself:** postcard drops custom messages, and a version or scalar
  mismatch, the failure a caller can act on, is the one a message must name.
- **`__reduce__`, not the default:** the pickle then holds only approx-chol's versioned form,
  not scipy's object state, which is not versioned.
- **The budget in the pickle:** it is what the Python operator holds; another process applies at
  the same budget, all cores when it was `None`.
- **`ValueError` in Python:** a saved form that does not load is bad input, as `InputError` and
  `BuildError` are `ValueError`s.
