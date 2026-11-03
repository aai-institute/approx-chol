# Save and load interface

The contract is `../contracts/save-load.md`; `ApproxInverse` is that of `interface/build.md`.

## Decision

```rust
#[cfg(feature = "serde")]
impl<T: Real> serde::Serialize for ApproxInverse<T> {}
#[cfg(feature = "serde")]
impl<'de, T: Real> serde::Deserialize<'de> for ApproxInverse<T> {}
```

- The saved form is a tuple, an array in a self-describing format: `version`, a `u32`; `scalar`,
  `"f32"` or `"f64"`; then $B$. A load reads the first two and rejects a form whose version is
  not the current one, or whose scalar is not `T`, before it reads $B$.
- A rejection of the version, the scalar or an invariant of $B$ goes through `D::Error::custom`,
  with a message naming what was found and what was expected; a decoding error is the format's
  own. A format that drops custom messages, such as postcard, keeps only the error.

Python:

```python
class ApproxInverse(LinearOperator):
    def __reduce__(self) -> tuple[object, ...]: ...
```

An `ApproxInverse` pickles as its saved form, encoded with postcard, and its held budget. A load
reads the version and the scalar itself and raises `ValueError` naming what was found and what
was expected; a form that then fails to decode or validate raises `ValueError` as invalid.

## Why

- **A tuple, `version` and `scalar` first:** a struct is a map in a self-describing format, so
  header-first reading needs the tuple there; a positional format, such as postcard, decodes the
  rest by position, and a mismatch checked first fails as itself, not as garbage further on.
- **No error type of its own:** serde returns the format's error, so a typed error would reach
  the caller only as its message.
- **postcard for pickle:** its wire format is specified and stable since 1.0, and floats take
  their width, 8 bytes per `f64`; MessagePack and CBOR add a byte per float. A pickle crosses
  releases inside joblib's files, so the encoding is part of the form.
- **Python reads the header itself:** postcard drops custom messages, and a version or scalar
  mismatch, the failure a caller can act on, is the one a message must name.
- **`__reduce__`, not the default:** scipy's `LinearOperator` changed its pickled state between
  releases; rebuilding through the constructor does not depend on it.
- **The budget in the pickle:** it is what the Python operator holds; another process applies at
  the same budget, all cores when it was `None`.
- **`ValueError` in Python:** a saved form that does not load is bad input, as `InputError` and
  `BuildError` are `ValueError`s.
