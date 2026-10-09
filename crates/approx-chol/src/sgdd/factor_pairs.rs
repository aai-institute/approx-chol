//! A view's components, each converted to SDDM form.

use num_traits::PrimInt;

use super::bipartite::Columns;
use super::cross_tab::{BipartiteComponent, CrossTab};
use super::{ScalingPolicy as ScalingConfig, ScalingReport, SgddError};

mod grounding;
pub(crate) mod sddm;
use grounding::{scaled_groundings, ScaledForest};
use sddm::NotScalable;
pub(crate) use sddm::{CoordinateMap, Grounding, LocalComponent, MatrixForm, SddmMatrix};

/// How a component's block converts: a plain pair is a Laplacian by construction.
#[derive(Clone, Copy)]
enum ComponentClass {
    KnownLaplacian,
    General(Grounding),
}

/// A component's vertices in `A`'s numbering, eliminated block first, and its converted matrix.
#[derive(Clone, Debug)]
pub(crate) struct LocalDomain {
    pub(crate) globals: Vec<u32>,
    pub(crate) component: LocalComponent,
}

/// One view's `C`.
pub(crate) fn cross_tab<I: PrimInt>(
    columns: Columns<'_, I>,
    n_rows: usize,
    n_cols: usize,
) -> CrossTab {
    CrossTab::build(columns, n_rows, n_cols)
}

/// Dead singletons produce no subdomain; a `None` diagonal states a Graph, a Laplacian by type.
pub(crate) fn split_into_subdomains<I: PrimInt>(
    columns: Columns<'_, I>,
    full_ct: CrossTab,
    diagonal: Option<(&[f64], &[f64])>,
    config: &ScalingConfig,
) -> Result<(Vec<LocalDomain>, Option<ScalingReport>), SgddError> {
    debug_assert!(diagonal.is_none_or(|(row_diag, col_diag)| {
        (row_diag.len(), col_diag.len()) == (full_ct.n_rows(), full_ct.n_cols())
    }));
    let known_laplacian = diagonal.is_none();
    let n_rows_full = full_ct.n_rows();
    let components: Vec<(BipartiteComponent, ComponentClass)> = if known_laplacian {
        (full_ct.bipartite_connected_components().into_iter())
            .map(|comp| (comp, ComponentClass::KnownLaplacian))
            .collect()
    } else {
        let mut forest = ScaledForest::grow(&columns, &full_ct.c);
        (forest.components().into_iter())
            .map(|(root, comp)| (comp, ComponentClass::General(forest.grounding(root))))
            .collect()
    };

    let full_ct_n_cols = full_ct.n_cols();
    let cross_tabs: Vec<CrossTab> = if components.len() == 1 {
        vec![full_ct]
    } else {
        let mut row_remap = vec![u32::MAX; full_ct.n_rows()];
        let mut col_remap = vec![u32::MAX; full_ct.n_cols()];
        components
            .iter()
            .map(|(comp, _)| full_ct.extract_component(comp, &mut row_remap, &mut col_remap))
            .collect()
    };

    let mut levels = (!known_laplacian).then(|| vec![None; n_rows_full + full_ct_n_cols]);
    let mut converted = Vec::with_capacity(components.len());
    let mut report: Option<ScalingReport> = None;
    for ((comp, class), comp_ct) in components.iter().zip(cross_tabs) {
        let comp_diag: Option<Vec<f64>> = diagonal.map(|(row_diag, col_diag)| {
            comp.rows
                .iter()
                .map(|&i| row_diag[i])
                .chain(comp.cols.iter().map(|&i| col_diag[i]))
                .collect()
        });
        // A Graph's weights cannot cancel, so its only dead component is an edgeless singleton.
        let dead = match &comp_diag {
            Some(comp_diag) => comp_diag.iter().all(|&v| v == 0.0),
            None => comp.rows.len() + comp.cols.len() == 1,
        };
        if dead {
            continue;
        }
        let comp_globals: Vec<u32> = comp
            .rows
            .iter()
            .map(|&i| i as u32)
            .chain(comp.cols.iter().map(|&i| (n_rows_full + i) as u32))
            .collect();
        let vertex = comp_globals[0];
        let (comp_ct, comp_diag, comp_globals) =
            sddm::orient_for_elimination(comp_ct, comp_diag, comp_globals);
        let swapped = comp_ct.n_rows() != comp.rows.len();
        let converted_component = match (*class, comp_diag) {
            (ComponentClass::KnownLaplacian, _) => {
                Ok((sddm::convert_known_laplacian(comp_ct), None))
            }
            (ComponentClass::General(_), Some(comp_diag)) => {
                sddm::convert_general(comp_ct, comp_diag, config)
            }
            (ComponentClass::General(_), None) => unreachable!("a Gram supplies its diagonal"),
        };
        let (component, uncertified) =
            converted_component.map_err(|NotScalable| SgddError::Unscalable { vertex })?;
        if let Some(uncertified) = uncertified {
            if report.is_none_or(|worst| uncertified.violation > worst.violation) {
                report = Some(ScalingReport {
                    iterations: uncertified.iterations,
                    violation: uncertified.violation,
                });
            }
        }
        if let (Some(levels), MatrixForm::Laplacian) = (levels.as_mut(), component.form) {
            let n_eliminated = component.matrix.n_eliminated();
            let (row_offset, col_offset) = if swapped {
                (n_eliminated, 0)
            } else {
                (0, n_eliminated)
            };
            let factor = |local| {
                Some((
                    converted.len(),
                    component.coordinates.factor(local, n_eliminated),
                ))
            };
            for (p, &i) in comp.rows.iter().enumerate() {
                levels[i] = factor(row_offset + p);
            }
            for (q, &j) in comp.cols.iter().enumerate() {
                levels[n_rows_full + j] = factor(col_offset + q);
            }
        }
        converted.push((*class, comp_globals, component));
    }

    // A folded component's dropped surplus is `fᵀGf` exactly, so only those read the observations.
    let scaled =
        levels.map(|levels| scaled_groundings(&columns, &levels, n_rows_full, converted.len()));
    let mut domains = Vec::with_capacity(converted.len());
    for (k, (class, comp_globals, mut component)) in converted.into_iter().enumerate() {
        if let ComponentClass::General(forest) = class {
            let grounding = match (component.form, &scaled) {
                (MatrixForm::Laplacian, Some(scaled)) => scaled[k],
                _ => forest,
            };
            if grounding == Grounding::Floating {
                component.matrix.float();
            }
        }
        domains.push(LocalDomain {
            globals: comp_globals,
            component,
        });
    }
    Ok((domains, report))
}
