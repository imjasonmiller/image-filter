pub fn gaussian_kernel_1d(sigma: f64) -> Vec<f64> {
    if sigma <= 0.0 {
        panic!("sigma should be > 0")
    }

    let radius = sigma.ceil() as i64 * 3;
    let mut kernel = vec![0.0; (radius * 2 + 1) as usize];

    for (i, x) in (-radius..=radius).enumerate() {
        kernel[i] = f64::exp(-(x.pow(2) as f64) / (2.0 * sigma.powi(2)));
    }

    // Normalize the kernel values to have a sum of 1
    let sum = kernel.iter().sum::<f64>();

    kernel.iter().map(|n| n / sum).collect()
}

pub fn gaussian_kernel_2d(sigma: f64) -> Vec<Vec<f64>> {
    if sigma <= 0.0 {
        panic!("sigma should be > 0")
    }

    let radius = sigma.ceil() as isize * 3;
    let mut kernel = vec![vec![0.0; (radius * 2 + 1) as usize]; (radius * 2 + 1) as usize];

    for (i, x) in (-radius..=radius).enumerate() {
        for (j, y) in (-radius..=radius).enumerate() {
            kernel[j][i] = f64::exp(-(x.pow(2) + y.pow(2)) as f64 / (2.0 * sigma.powi(2)));
        }
    }

    // Normalize the kernel values to have a sum of 1
    let sum = kernel.iter().flat_map(IntoIterator::into_iter).sum::<f64>();

    kernel
        .iter()
        .map(|x| x.iter().map(|n| n / sum).collect())
        .collect()
}

