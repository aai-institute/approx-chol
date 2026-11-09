/// `‖Ax − b‖ / ‖b‖` from `Ax`.
pub fn relative_residual(ax: &[f64], b: &[f64]) -> f64 {
    let norm = |v: &mut dyn Iterator<Item = f64>| v.map(|v| v * v).sum::<f64>().sqrt();
    norm(&mut ax.iter().zip(b).map(|(ax, b)| ax - b)) / norm(&mut b.iter().copied())
}

/// PCG on `apply` preconditioned by `precondition`, from zero; the iterations to `‖Ax − b‖ ≤ tolerance ‖b‖`, if within `max`: the recurrence's `r` stops it, the true residual confirms.
pub fn pcg_iterations(
    apply: impl Fn(&[f64]) -> Vec<f64>,
    precondition: impl Fn(&[f64]) -> Vec<f64>,
    b: &[f64],
    tolerance: f64,
    max: usize,
) -> Option<usize> {
    let dot = |u: &[f64], v: &[f64]| u.iter().zip(v).map(|(u, v)| u * v).sum::<f64>();
    let bound = tolerance * dot(b, b).sqrt();
    let (mut x, mut r) = (vec![0.0; b.len()], b.to_vec());
    let mut z = precondition(&r);
    let (mut p, mut rz) = (z.clone(), dot(&r, &z));
    for iteration in 1..=max {
        let q = apply(&p);
        let alpha = rz / dot(&p, &q);
        (x.iter_mut().zip(&p)).for_each(|(x, p)| *x += alpha * p);
        (r.iter_mut().zip(&q)).for_each(|(r, q)| *r -= alpha * q);
        if dot(&r, &r).sqrt() <= bound && relative_residual(&apply(&x), b) <= tolerance {
            return Some(iteration);
        }
        z = precondition(&r);
        let next = dot(&r, &z);
        (p.iter_mut().zip(&z)).for_each(|(p, z)| *p = z + next / rz * *p);
        rz = next;
    }
    None
}
