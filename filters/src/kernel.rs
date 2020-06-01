use ndarray::prelude::*;
use ndarray::Array;
use std::iter::FromIterator;

pub fn gaussian_kernel_1d(sigma: f64) -> (Array2<f64>, Array2<f64>) {
    if sigma <= 0.0 {
        panic!("sigma should be > 0.0")
    }

    // Generate a 1×N Gaussian kernel
    let radius = sigma.ceil() as i64 * 3;
    let kernel = Array::from_iter(
        (-radius..=radius).map(|x| (-(x.pow(2) as f64) / (2.0 * sigma.powi(2))).exp()),
    );

    // Return normalized kernels
    let kernel_x = (&kernel / kernel.sum())
        .into_shape((1, radius as usize * 2 + 1))
        .unwrap();

    let kernel_y = kernel_x.clone().reversed_axes();

    (kernel_x, kernel_y)
}

pub fn gaussian_kernel_2d(sigma: f64) -> Array2<f64> {
    if sigma <= 0.0 {
        panic!("sigma should be > 0.0")
    }

    // Generate an N×N Gaussian kernel
    let radius = sigma.ceil() as i64 * 3;
    let kernel = Array::from_shape_fn(
        (radius as usize * 2 + 1, radius as usize * 2 + 1),
        |(i, j)| {
            let i = (i as i64 - radius) as f64;
            let j = (j as i64 - radius) as f64;

            (-(j.powi(2) + i.powi(2)) / (2.0 * sigma.powi(2))).exp()
        },
    );

    // Return normalized kernel
    &kernel / kernel.sum()
}

pub fn sobel2d() -> (Array2<f64>, Array2<f64>) {
    #[rustfmt::skip]
    let kernel_x = array![
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0],
    ];

    #[rustfmt::skip]
    let kernel_y = array![
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ];

    (kernel_x, kernel_y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[should_panic(expected = "sigma should be > 0.0")]
    fn invalid_zero_gaussian_kernel_1d() {
        gaussian_kernel_1d(0.0);
    }

    #[test]
    #[should_panic(expected = "sigma should be > 0.0")]
    fn invalid_negative_gaussian_kernel_1d() {
        gaussian_kernel_1d(-1.0);
    }

    #[test]
    #[should_panic(expected = "sigma should be > 0.0")]
    fn invalid_zero_gaussian_kernel_2d() {
        gaussian_kernel_2d(0.0);
    }

    #[test]
    #[should_panic(expected = "sigma should be > 0.0")]
    fn invalid_negative_gaussian_kernel_2d() {
        gaussian_kernel_2d(-1.0);
    }

    #[test]
    fn valid_gaussian_1d() {
        let expect: [[f64; 7]; 1] = [[
            0.00081721, 0.02804152, 0.23392642, 0.47442967, 0.23392642, 0.02804152, 0.00081721,
        ]];

        for ((i, j), result) in gaussian_kernel_1d(0.84089642).0.indexed_iter() {
            assert_relative_eq!(expect[i][j], result, epsilon = 1e-8f64);
        }
    }

    #[test]
    fn valid_gaussian_2d() {
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

        for ((i, j), result) in gaussian_kernel_2d(0.84089642).indexed_iter() {
            assert_relative_eq!(expect[i][j], result, epsilon = 1e-8f64);
        }
    }

    #[test]
    fn valid_sobel2d() {
        #[rustfmt::skip]
        let expect_x: [[f64; 3]; 3] = [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0]
        ];

        #[rustfmt::skip]
        let expect_y: [[f64; 3]; 3] = [
            [1.0, 2.0, 1.0],
            [0.0, 0.0, 0.0],
            [-1.0, -2.0, -1.0],
        ];

        let (sobel2d_x, sobel2d_y) = sobel2d();

        for ((i, j), result) in sobel2d_x.indexed_iter() {
            assert_relative_eq!(expect_x[i][j], result);
        }

        for ((i, j), result) in sobel2d_y.indexed_iter() {
            assert_relative_eq!(expect_y[i][j], result);
        }
    }
}

