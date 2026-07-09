#[path = "common/panic_err.rs"]
mod panic_err;
#[path = "common/panic_ok.rs"]
mod panic_ok;

use approx_chol::{
    factorize_with, Config, ConfigError, CsrRef, Error, InputClass, OnDeficit, Scaling,
};
use panic_err::ErrOrPanic;
use panic_ok::OrPanic;

// Path Laplacian (sign-free, previously-valid input).
fn laplacian() -> CsrRef<'static, f64, u32> {
    static RP: [u32; 5] = [0, 2, 5, 8, 10];
    static CI: [u32; 10] = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    static V: [f64; 10] = [1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
    CsrRef::new(&RP, &CI, &V, 4).or_panic("valid csr")
}

#[test]
fn default_assumes_auto_and_errors_on_deficit() {
    let c = Config::default();
    assert_eq!(c.assume, InputClass::Auto);
    assert_eq!(c.scaling.on_deficit, OnDeficit::Error);
    assert!(c.scaling.budget > 0 && c.scaling.slack > 0.0);
}

#[test]
fn functional_update_idiom_is_preserved() {
    let c = Config {
        assume: InputClass::HMatrix,
        ..Default::default()
    };
    assert_eq!(c.assume, InputClass::HMatrix);
    assert_eq!(c.seed, 0);
    assert_eq!(c.scaling, Scaling::default());
}

#[test]
fn every_class_hint_is_constructible() {
    for class in [
        InputClass::Auto,
        InputClass::Laplacian,
        InputClass::Sddm,
        InputClass::Sdd,
        InputClass::HMatrix,
    ] {
        let c = Config {
            assume: class,
            ..Default::default()
        };
        assert_eq!(c.assume, class);
    }
}

#[test]
fn default_config_factors_previously_valid_input() {
    factorize_with(laplacian(), Config::default()).or_panic("default config must factor");
}

#[test]
fn scaling_rejects_invalid_params() {
    for (scaling, expected) in [
        (
            Scaling {
                budget: 0,
                ..Default::default()
            },
            ConfigError::ScalingBudgetMustBePositive,
        ),
        (
            Scaling {
                slack: 0.0,
                ..Default::default()
            },
            ConfigError::ScalingSlackMustBeFinitePositive,
        ),
        (
            Scaling {
                slack: f64::NAN,
                ..Default::default()
            },
            ConfigError::ScalingSlackMustBeFinitePositive,
        ),
    ] {
        let config = Config {
            scaling,
            ..Default::default()
        };
        let err = factorize_with(laplacian(), config).err_or_panic("invalid scaling must error");
        assert_eq!(err, Error::InvalidConfig(expected));
    }
}
