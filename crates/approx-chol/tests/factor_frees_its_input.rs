//! Its own binary: the bench's allocator counts every live byte, so nothing else may run beside it.

use std::num::NonZeroUsize;

use approx_chol::input::Side;
use approx_chol::{ApproxInverse, Complement, Config, ScalingPolicy, Sgdd, Threads};

#[path = "integration/common/bipartite.rs"]
#[allow(dead_code, reason = "only its `groups`")]
mod bipartite;
#[path = "integration/common/disjoint.rs"]
mod disjoint;
#[path = "integration/common/lcg.rs"]
mod lcg;
#[path = "../benches/common/peak.rs"]
mod peak;

use bipartite::Bipartite;
use disjoint::Disjoint;
use peak::{live, peak_while};

/// Bytes the reduction holds, bytes the factor holds, and the most live past the reduction's start while factoring.
fn bytes<R>(
    reduce: impl FnOnce() -> R,
    factor: impl FnOnce(R) -> ApproxInverse<f64>,
) -> [usize; 3] {
    let start = live();
    let reduced = reduce();
    let held = live() - start;
    let (factored, past_held) = peak_while(|| factor(reduced));
    let kept = live() - start;
    drop(factored);
    [held, kept, held + past_held]
}

/// 400 blocks each way: on 1 and 4 threads the peaks sit within 4% of the larger of reduction and factor; inputs held to the end put them 44% (complement) and 67% (sgdd) past it.
#[test]
fn factoring_many_blocks_never_holds_the_reduction_beside_the_factor() {
    let gram = Disjoint::new(&[500; 400], |_| false);
    let bipartite = Bipartite::groups(8, &[(120, 60, false); 400]);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let (config, policy) = (Config::default(), ScalingPolicy::default());
    for threads in [Threads::Count(NonZeroUsize::MIN), Threads::All] {
        let sgdd = pool.install(|| {
            bytes(
                || Sgdd::reduce(gram.view(), policy, threads).unwrap(),
                |reduced| reduced.factor(config, threads),
            )
        });
        let complement = pool.install(|| {
            bytes(
                || Complement::reduce_onto(bipartite.view(), Side::Right, policy, threads).unwrap(),
                |reduced| reduced.factor(config, threads),
            )
        });
        for (name, [held, kept, peak]) in [("sgdd", sgdd), ("complement", complement)] {
            let bound = held.max(kept) + held.min(kept) / 4;
            assert!(
                peak < bound,
                "{name} at {threads:?}: peak {peak} past {bound} (reduction {held}, factor {kept})"
            );
        }
    }
}
