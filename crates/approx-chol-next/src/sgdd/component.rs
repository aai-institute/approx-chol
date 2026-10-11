//! A reduced component: `M` and where it sits in `A`.

use std::sync::Arc;

use super::{
    within_series_floor, Balance, Congruence, Frame, Grounding, OffDiagonal, ReduceError,
    ScalingReport,
};
use crate::storage::{diagonal_in_range, in_range, narrow, widen, CrossBlock, UpperAdjacency};
use crate::Real;

/// One component of `A`, reduced: `M = F · A_c · F` over its vertices, eliminated block first.
#[derive(Clone, Debug)]
pub struct Component<T = f64> {
    frame: Arc<Frame<T>>,
    m: Sdd<T>,
    report: Option<ScalingReport>,
}

impl<T: Real> Component<T> {
    /// Panics unless `M` has one row per vertex of the frame.
    pub(super) fn new(frame: Frame<T>, m: Sdd<T>, report: Option<ScalingReport>) -> Self {
        assert_eq!(m.n(), frame.vertices.len(), "one row per vertex");
        Self {
            frame: Arc::new(frame),
            m,
            report,
        }
    }

    /// Only the degrees are checked: the shape proves the signature, dominance and floating.
    pub(super) fn signless(
        vertices: Box<[u32]>,
        cross: CrossBlock<T>,
    ) -> Result<Self, ReduceError> {
        within_series_floor::<T>(&vertices)?;
        if let Some(i) = cross
            .abs_row_sums()
            .iter()
            .position(|&d| !diagonal_in_range(d))
        {
            let vertex = vertices[i];
            return Err(ReduceError::OutOfRange { vertex });
        }
        let vertex = vertices[0];
        let congruence = Congruence::negate_from(cross.n_eliminated());
        let off_diagonal = (congruence.transform(OffDiagonal::Bipartite(Arc::new(cross))))
            .ok_or(ReduceError::Unscalable { vertex })?;
        let m = Sdd::new(off_diagonal, Balance::Balanced(Grounding::ungrounded()));
        Ok(Self::new(Frame::new(vertices, congruence), m, None))
    }

    /// The component's vertices in `A`'s numbering, in local order.
    pub fn vertices(&self) -> &[u32] {
        self.frame.vertices()
    }

    /// Present iff this component's scaling was clamped beyond the policy's tolerance.
    pub fn scaling_report(&self) -> Option<ScalingReport> {
        self.report
    }

    /// Shared with every factor of this component.
    pub(crate) fn frame(&self) -> &Arc<Frame<T>> {
        &self.frame
    }

    /// Whether `A_c` is proven singular, so a solve returns `A_c⁺b`.
    pub fn floats(&self) -> bool {
        matches!(&self.m.balance, Balance::Balanced(grounding) if !grounding.is_grounded())
    }

    /// `M`: its off-diagonal and its balance.
    pub(crate) fn m(&self) -> (&OffDiagonal<T>, &Balance<T>) {
        (&self.m.off_diagonal, &self.m.balance)
    }
}

/// One component of a graph Laplacian's view: its edges prove the signature and dominance, and its ground `g` the grounding.
pub(super) struct ProvenLaplacian<T> {
    vertices: Box<[u32]>,
    adjacency: UpperAdjacency<T>,
    /// The edges' weights summed per vertex, read only for the range `A`'s diagonal must clear.
    diagonal: Vec<T>,
    ground: Option<Vec<T>>,
}

impl<T: Real> ProvenLaplacian<T> {
    /// Panics unless one diagonal, and one ground when given, per vertex.
    pub(super) fn new(
        vertices: Box<[u32]>,
        adjacency: UpperAdjacency<T>,
        diagonal: Vec<T>,
        ground: Option<Vec<T>>,
    ) -> Self {
        assert_eq!(vertices.len(), adjacency.n(), "one vertex per row");
        assert_eq!(diagonal.len(), adjacency.n(), "one diagonal per row");
        assert!(
            (ground.as_ref()).is_none_or(|g| g.len() == adjacency.n()),
            "one ground per row"
        );
        Self {
            vertices,
            adjacency,
            diagonal,
            ground,
        }
    }

    /// The lone vertex, when it has nothing to solve: it has no edge, so only a ground makes it a component.
    pub(super) fn dead(&self) -> Option<u32> {
        let ground = |i: usize| (self.ground.as_ref()).map_or(T::zero(), |g| g[i]);
        let dead = self.vertices.len() == 1 && self.diagonal[0] + ground(0) == T::zero();
        dead.then_some(self.vertices[0])
    }

    /// Only ranges are checked: `A`'s diagonal `d + g`, summed in `f64`, and each ground like any entry.
    pub(super) fn into_component(self) -> Result<Component<T>, ReduceError> {
        let Self {
            vertices,
            adjacency,
            diagonal,
            ground,
        } = self;
        within_series_floor::<T>(&vertices)?;
        let out_of_range = |i: usize| match &ground {
            Some(g) => {
                !in_range(g[i]) || !diagonal_in_range(narrow::<T>(widen(diagonal[i]) + widen(g[i])))
            }
            None => !diagonal_in_range(diagonal[i]),
        };
        if let Some(i) = (0..vertices.len()).find(|&i| out_of_range(i)) {
            let vertex = vertices[i];
            return Err(ReduceError::OutOfRange { vertex });
        }
        let balance = Balance::Balanced(ground.map_or_else(Grounding::ungrounded, Grounding::of));
        if !balance.ground_in_range() {
            let vertex = vertices[0];
            return Err(ReduceError::OutOfRange { vertex });
        }
        let m = Sdd::new(OffDiagonal::General(adjacency), balance);
        Ok(Component::new(
            Frame::new(vertices, Congruence::identity()),
            m,
            None,
        ))
    }
}

/// `M`, weakly dominant once scaled and clamped: its off-diagonal, and whether a signature folded it.
#[derive(Clone, Debug)]
pub(super) struct Sdd<T> {
    off_diagonal: OffDiagonal<T>,
    balance: Balance<T>,
}

impl<T: Real> Sdd<T> {
    /// Panics unless one surplus per row, and a balanced `M` folded to a Z-matrix.
    pub(super) fn new(off_diagonal: OffDiagonal<T>, balance: Balance<T>) -> Self {
        assert!(
            balance.grounding().fits(off_diagonal.n()),
            "one surplus per row"
        );
        if let Balance::Balanced(_) = balance {
            assert!(off_diagonal.nonpositive(), "a folded M is a Z-matrix");
        }
        Self {
            off_diagonal,
            balance,
        }
    }

    fn n(&self) -> usize {
        self.off_diagonal.n()
    }
}
