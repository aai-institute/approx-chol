//! Each Gram of #281, as `FactorWidth2` and as its CSR twin: the peak bytes its reduction and its build hold, on the calling thread and on all of a 12-thread pool.

#[path = "common/csr.rs"]
#[allow(dead_code, reason = "the Grams take only `gram` and `csr`")]
mod csr;
#[path = "../tests/integration/common/disjoint.rs"]
mod disjoint;
#[path = "common/gram.rs"]
mod gram;
#[path = "../tests/integration/common/lcg.rs"]
mod lcg;
#[path = "common/peak.rs"]
mod peak;

use std::num::NonZeroUsize;

use approx_chol::input::Input;
use approx_chol::{Config, ScalingPolicy, Sgdd, Threads};
use peak::peak_while;

fn main() {
    let pool = (rayon::ThreadPoolBuilder::new().num_threads(12).build()).expect("a pool");
    let one = Threads::Count(NonZeroUsize::MIN);
    println!("case\tn\tcolumns\tnnz_A\tinput\tthreads\treduce_peak_bytes\tbuild_peak_bytes");
    for (name, sizes) in gram::cases() {
        let (gram, csr) = gram::with_csr(&sizes);
        let inputs: [(&str, Input<'_, f64, u32>); 2] =
            [("fw2", gram.view().into()), ("csr", csr.csr().into())];
        for (input, view) in inputs {
            for (t, threads) in [("t1", one), ("t12", Threads::All)] {
                let reduce = || Sgdd::reduce(view, ScalingPolicy::default(), threads);
                let (_, reduced) = pool.install(|| peak_while(reduce));
                let build = || {
                    reduce()
                        .expect("reduces")
                        .factor(Config::default(), threads)
                };
                let (_, built) = pool.install(|| peak_while(build));
                let (n, columns, nnz) = (gram.n, gram.a.len(), csr.nnz());
                println!("{name}\t{n}\t{columns}\t{nnz}\t{input}\t{t}\t{reduced}\t{built}");
            }
        }
    }
}
