use approx_chol::{Config, SchurMode};

use crate::common::bipartite::Bipartite;
use crate::common::frustrated::solves;
use crate::common::sparse::clique_tree;

#[test]
fn an_approximate_complement_sampled_as_the_factor_is_same() {
    let g = Bipartite::frustrated();
    let config = Config {
        sampling: clique_tree(2),
        exact_max_dim: 0,
        complement: SchurMode::Same,
        ..Config::default()
    };
    let approximate = |copies| Config {
        complement: SchurMode::Approximate {
            sampling: clique_tree(copies),
        },
        ..config
    };
    let same = solves(&g, config);
    assert_eq!(solves(&g, approximate(2)), same);
    assert_ne!(solves(&g, approximate(1)), same);
}

#[cfg(feature = "serde")]
mod serde {
    use super::*;

    #[test]
    fn a_config_round_trips() {
        let config = Config {
            seed: 3,
            sampling: clique_tree(u32::MAX),
            exact_max_dim: 0,
            complement: SchurMode::Approximate {
                sampling: clique_tree(2),
            },
        };
        let json = serde_json::to_string(&config).unwrap();
        assert_eq!(serde_json::from_str::<Config>(&json).unwrap(), config);
        let bytes = postcard::to_stdvec(&config).unwrap();
        assert_eq!(postcard::from_bytes::<Config>(&bytes).unwrap(), config);
    }

    #[test]
    fn zero_copies_are_rejected() {
        let json = serde_json::to_value(Config::default()).unwrap();
        let zero = json.to_string().replace(r#""copies":1"#, r#""copies":0"#);
        assert_ne!(zero, json.to_string(), "the default states its copies");
        assert!(serde_json::from_str::<Config>(&zero).is_err());
    }

    /// A field this version does not read, `split_merge` above all, is rejected beside the ones it does.
    #[test]
    fn a_config_stating_split_merge_is_rejected() {
        let config = Config {
            complement: SchurMode::Approximate {
                sampling: clique_tree(2),
            },
            ..Config::default()
        };
        let json = serde_json::to_value(config).unwrap();
        assert_eq!(
            serde_json::from_value::<Config>(json.clone()).unwrap(),
            config
        );
        for pointer in ["", "/complement/Approximate", "/sampling/CliqueTree"] {
            let mut stated = json.clone();
            let object = stated.pointer_mut(pointer).and_then(|v| v.as_object_mut());
            object
                .expect("an object")
                .insert("split_merge".into(), 2.into());
            assert!(
                serde_json::from_value::<Config>(stated).is_err(),
                "{pointer:?}"
            );
        }
    }
}
