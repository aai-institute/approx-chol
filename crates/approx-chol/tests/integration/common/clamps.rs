use approx_chol::{ApproxInverse, Real, ScalingOutcome};

/// A clamped component's vertices, ascending, with its clamp.
#[derive(Clone, Debug, PartialEq)]
pub struct Clamped {
    pub vertices: Vec<usize>,
    pub violation: f64,
    pub scaling: ScalingOutcome,
}

/// Each clamped component in component order, as `components` lists it.
pub trait Clamps {
    fn clamps(&self) -> Vec<Clamped>;
}

impl<T: Real> Clamps for ApproxInverse<T> {
    fn clamps(&self) -> Vec<Clamped> {
        (self.components())
            .filter_map(|component| {
                let clamp = component.clamp()?;
                let mut vertices: Vec<usize> = component.vertices().collect();
                vertices.sort_unstable();
                Some(Clamped {
                    vertices,
                    violation: clamp.violation,
                    scaling: clamp.scaling,
                })
            })
            .collect()
    }
}
