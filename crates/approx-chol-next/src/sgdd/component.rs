//! A reduced component, whose grounding only proof or a surviving surplus may change.

use std::sync::Arc;

use super::{Balance, Frame, Grounding, OffDiagonal, ReduceError, ScalingReport};
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

    /// A proven alias anchors one vertex instead of the surplus rounding left.
    pub(super) fn float(&mut self) {
        *self.balanced_grounding() = Grounding::ungrounded();
    }

    /// An unproven component keeps a surplus: one that vanishes in `T` is `Unscalable`, never a float.
    pub(super) fn ground(&mut self, surplus: Vec<T>) -> Result<(), ReduceError> {
        assert_eq!(surplus.len(), self.m.n(), "one surplus per row");
        let grounding = Grounding::of(surplus);
        if !grounding.is_grounded() {
            let vertex = self.vertices()[0];
            return Err(ReduceError::Unscalable { vertex });
        }
        *self.balanced_grounding() = grounding;
        Ok(())
    }

    /// Panics unless folded: only a Z-matrix's surplus is its row sums `M·1`.
    fn balanced_grounding(&mut self) -> &mut Grounding<T> {
        match &mut self.m.balance {
            Balance::Balanced(grounding) => grounding,
            Balance::Frustrated(_) => panic!("a frustrated component is definite"),
        }
    }

    /// `M`: its off-diagonal and its balance.
    pub(crate) fn m(&self) -> (&OffDiagonal<T>, &Balance<T>) {
        (&self.m.off_diagonal, &self.m.balance)
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
