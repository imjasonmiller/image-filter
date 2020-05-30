pub fn gaussian_kernel_1d(sigma: f64) -> Vec<Vec<f64>> {
    if sigma <= 0.0 {
        panic!("sigma should be > 0")
    }

    let radius = sigma.ceil() as i64 * 3;
    let mut kernel = vec![vec![0.0; (radius * 2 + 1) as usize]];

    for (i, x) in (-radius..=radius).enumerate() {
        kernel[0][i] = f64::exp(-(x.pow(2) as f64) / (2.0 * sigma.powi(2)));
    }

    // Normalize the kernel values to have a sum of 1
    let sum = kernel[0].iter().sum::<f64>();

    kernel
        .iter()
        .map(|x| x.iter().map(|n| n / sum).collect())
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[should_panic(expected = "sigma should be > 0")]
    fn invalid_zero_gaussian_kernel_1d() {
        gaussian_kernel_1d(0.0);
    }

    #[test]
    #[should_panic(expected = "sigma should be > 0")]
    fn invalid_negative_gaussian_kernel_1d() {
        gaussian_kernel_1d(-1.0);
    }

    #[test]
    #[should_panic(expected = "sigma should be > 0")]
    fn invalid_zero_gaussian_kernel_2d() {
        gaussian_kernel_2d(0.0);
    }

    #[test]
    #[should_panic(expected = "sigma should be > 0")]
    fn invalid_negative_gaussian_kernel_2d() {
        gaussian_kernel_2d(-1.0);
    }

    #[test]
    fn valid_kernel_1d() {
        let expect: [[f64; 7]; 1] = [[
            0.00081721, 0.02804152, 0.23392642, 0.47442967, 0.23392642, 0.02804152, 0.00081721,
        ]];

        let result = gaussian_kernel_1d(0.84089642);

        for i in 0..result.len() {
            assert_relative_eq!(expect[0][i], result[0][i], epsilon = 1e-8f64);
        }
    }

    #[test]
    fn valid_kernel_2d() {
        let expect: [[f64; 7]; 7] = [
            [
                0.00000066, 0.00002291, 0.00019116, 0.00038771, 0.00019116, 0.00002291, 0.00000066,
            ],
            [
                0.00002291, 0.00078632, 0.00655965, 0.01330372, 0.00655965, 0.00078632, 0.00002291,
            ],
            [
                0.00019116, 0.00655965, 0.05472157, 0.11098163, 0.05472157, 0.00655965, 0.00019116,
            ],
            [
                0.00038771, 0.01330372, 0.11098163, 0.22508351, 0.11098163, 0.01330372, 0.00038771,
            ],
            [
                0.00019116, 0.00655965, 0.05472157, 0.11098163, 0.05472157, 0.00655965, 0.00019116,
            ],
            [
                0.00002291, 0.00078632, 0.00655965, 0.01330372, 0.00655965, 0.00078632, 0.00002291,
            ],
            [
                0.00000066, 0.00002291, 0.00019116, 0.00038771, 0.00019116, 0.00002291, 0.00000066,
            ],
        ];

        let result = gaussian_kernel_2d(0.84089642);

        for i in 0..result.len() {
            for j in 0..result.len() {
                assert_relative_eq!(expect[i][j], result[i][j], epsilon = 1e-8f64);
            }
        }
    }
}
