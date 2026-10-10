use crate::factor::clique_tree::PivotColumn;
use crate::Real;

/// A step's neighbors follow the previous step's, so the solve's running `usize` offset finds them at any size.
#[derive(Clone, Copy, Debug)]
struct StepHeader<T> {
    vertex: u32,
    /// Below `n`, so `u32` always holds it and the header stays 16 bytes.
    len: u32,
    /// `D⁻¹` premultiplied by the retained fraction, so no pass rebuilds it.
    pivot_scale: T,
}

/// The eliminated columns, appended in pivot order.
#[derive(Clone, Debug)]
pub(super) struct Steps<T> {
    headers: Vec<StepHeader<T>>,
    neighbors: Vec<u32>,
    coefficients: Vec<T>,
}

impl<T: Real> Steps<T> {
    pub(super) fn with_capacity(n: usize) -> Self {
        Self {
            headers: Vec::with_capacity(n),
            neighbors: Vec::new(),
            coefficients: Vec::new(),
        }
    }

    /// The one site deriving the remainder's share.
    pub(super) fn push(&mut self, vertex: usize, pivot: T, column: &PivotColumn<T>) {
        let start = self.neighbors.len();
        let retained = match column.shares() {
            None => T::one(),
            Some((shares, remainder)) => {
                let mut left = T::one();
                for &(neighbor, share) in shares {
                    self.neighbors.push(neighbor);
                    self.coefficients.push(share);
                    left = left - share;
                }
                self.neighbors.push(remainder);
                self.coefficients.push(left);
                left
            }
        };
        // Admitted input reaches this only past ~2^1074 range in one star (#168); `one` drops the scale.
        debug_assert!(
            (T::one() / pivot).is_finite(),
            "pivot has no finite inverse"
        );
        let inverse = match T::one() / pivot {
            inverse if inverse.is_finite() => inverse,
            _ => T::one(),
        };
        self.headers.push(StepHeader {
            vertex: vertex as u32,
            len: (self.neighbors.len() - start) as u32,
            pivot_scale: retained * inverse,
        });
    }

    /// A factor cannot exist without naming the vertex its steps left behind.
    pub(super) fn finish(self, uneliminated: usize) -> ApproximateCholesky<T> {
        ApproximateCholesky {
            steps: self,
            uneliminated: uneliminated as u32,
        }
    }
}

/// `LDLᵀ` with sampled columns: a serial recurrence, so shares are composed against the original pivot.
#[derive(Clone, Debug)]
pub(in crate::factor::cholesky) struct ApproximateCholesky<T> {
    steps: Steps<T>,
    /// The one vertex no step eliminates, so nothing divides its entry by a pivot.
    uneliminated: u32,
}

impl<T: Real> ApproximateCholesky<T> {
    /// Forward, zero the uneliminated entry, backward; the gather dispatches `[]`, `[_]`, lanes by row length.
    pub(in crate::factor::cholesky) fn substitute(&self, y: &mut [T]) {
        let Steps {
            headers,
            neighbors,
            coefficients,
        } = &self.steps;
        let mut end = 0;
        for header in headers {
            let span = end..end + header.len as usize;
            end = span.end;
            let pivot = y[header.vertex as usize];
            match coefficients[span.clone()] {
                [c] => {
                    debug_assert!(c == T::one(), "a lone coefficient takes the whole pivot");
                    let j = neighbors[span.start] as usize;
                    y[j] = y[j] + pivot;
                }
                _ => {
                    for (&j, &c) in neighbors[span.clone()].iter().zip(&coefficients[span]) {
                        y[j as usize] = y[j as usize] + c * pivot;
                    }
                }
            }
            y[header.vertex as usize] = pivot * header.pivot_scale;
        }
        y[self.uneliminated as usize] = T::zero();
        for header in headers.iter().rev() {
            let span = end - header.len as usize..end;
            end = span.start;
            let v = header.vertex as usize;
            match coefficients[span.clone()] {
                [] => {}
                [c] => {
                    debug_assert!(c == T::one(), "a lone coefficient takes the whole pivot");
                    y[v] = y[v] + y[neighbors[span.start] as usize];
                }
                [.., retained] => {
                    let mut lanes = [T::zero(); LANES];
                    lanes[0] = retained * y[v];
                    let mut js = neighbors[span.clone()].chunks_exact(LANES);
                    let mut cs = coefficients[span].chunks_exact(LANES);
                    for (js, cs) in js.by_ref().zip(cs.by_ref()) {
                        for lane in 0..LANES {
                            lanes[lane] = lanes[lane] + cs[lane] * y[js[lane] as usize];
                        }
                    }
                    let mut total = lanes.iter().fold(T::zero(), |sum, &lane| sum + lane);
                    for (&j, &c) in js.remainder().iter().zip(cs.remainder()) {
                        total = total + c * y[j as usize];
                    }
                    y[v] = total;
                }
            }
        }
    }
}

/// The split reassociates the row's sum, so this is part of a factor's solve output.
const LANES: usize = 4;
