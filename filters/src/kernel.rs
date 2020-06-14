use ndarray::prelude::*;
use ndarray::Array;
use std::iter::FromIterator;

pub fn box_blur_kernel_1d(radius: usize) -> (Array2<f32>, Array2<f32>) {
    let kernel = Array::<f32, _>::ones((1, radius * 2 + 1).into_shape());

    // Return normalized kernels
    let kernel_x = &kernel / kernel.sum();
    let kernel_y = kernel_x.clone().reversed_axes();

    (kernel_x, kernel_y)
}

pub fn box_blur_kernel_2d(radius: usize) -> Array2<f32> {
    let kernel = Array::<f32, _>::ones((radius * 2 + 1, radius * 2 + 1).into_shape());

    // Return normalized kernel
    &kernel / kernel.sum()
}

pub fn gaussian_blur_kernel_1d(sigma: f32) -> (Array2<f32>, Array2<f32>) {
    assert!(sigma > 0.0, "--sigma should be > 0.0");

    // Generate a 1×N Gaussian kernel
    let radius = sigma.ceil() as i32 * 3;
    let kernel = Array::from_iter(
        (-radius..=radius).map(|x| (-(x.pow(2) as f32) / (2.0 * sigma.powi(2))).exp()),
    );

    // Return normalized kernels
    let kernel_x = (&kernel / kernel.sum())
        .into_shape((1, radius as usize * 2 + 1))
        .unwrap();

    let kernel_y = kernel_x.clone().reversed_axes();

    (kernel_x, kernel_y)
}

pub fn gaussian_blur_kernel_2d(sigma: f32) -> Array2<f32> {
    assert!(sigma > 0.0, "--sigma should be > 0.0");

    // Generate an N×N Gaussian kernel
    let radius = sigma.ceil() as i32 * 3;
    let kernel = Array::from_shape_fn(
        (radius as usize * 2 + 1, radius as usize * 2 + 1),
        |(i, j)| {
            let i = (i as i32 - radius) as f32;
            let j = (j as i32 - radius) as f32;

            (-(j.powi(2) + i.powi(2)) / (2.0 * sigma.powi(2))).exp()
        },
    );

    // Return normalized kernel
    &kernel / kernel.sum()
}

pub fn sobel_2d() -> (Array2<f32>, Array2<f32>) {
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
        gaussian_blur_kernel_1d(0.0);
    }

    #[test]
    #[should_panic(expected = "sigma should be > 0.0")]
    fn invalid_negative_gaussian_kernel_1d() {
        gaussian_blur_kernel_1d(-1.0);
    }

    #[test]
    #[should_panic(expected = "sigma should be > 0.0")]
    fn invalid_zero_gaussian_kernel_2d() {
        gaussian_blur_kernel_2d(0.0);
    }

    #[test]
    #[should_panic(expected = "sigma should be > 0.0")]
    fn invalid_negative_gaussian_kernel_2d() {
        gaussian_blur_kernel_2d(-1.0);
    }

    #[test]
    fn valid_gaussian_1d() {
        let expect: [[f32; 7]; 1] = [[
            0.00081721, 0.02804152, 0.23392642, 0.47442967, 0.23392642, 0.02804152, 0.00081721,
        ]];

        for ((i, j), result) in gaussian_blur_kernel_1d(0.84089642).0.indexed_iter() {
            assert_relative_eq!(expect[i][j], result, epsilon = 1e-8f32);
        }
    }

    #[test]
    fn valid_gaussian_2d() {
        let expect: [[f32; 7]; 7] = [
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

        for ((i, j), result) in gaussian_blur_kernel_2d(0.84089642).indexed_iter() {
            assert_relative_eq!(expect[i][j], result, epsilon = 1e-8f32);
        }
    }

    #[test]
    fn valid_box_blur_1d() {
        for radius in 1..10 {
            let (kernel_x, kernel_y) = box_blur_kernel_1d(radius);

            let expect = kernel_x.sum() / kernel_x.len() as f32;

            for result in kernel_x.iter() {
                assert_relative_eq!(expect, result);
            }

            let expect = kernel_y.sum() / kernel_y.len() as f32;

            for result in kernel_y.iter() {
                assert_relative_eq!(expect, result);
            }
        }
    }

    #[test]
    fn valid_box_blur_2d() {
        for radius in 1..10 {
            let kernel = box_blur_kernel_2d(radius);

            let expect = kernel.sum() / kernel.len() as f32;

            for result in kernel.iter() {
                assert_relative_eq!(expect, result);
            }
        }
    }

    #[test]
    fn valid_sobel2d() {
        #[rustfmt::skip]
        let expect_x: [[f32; 3]; 3] = [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0]
        ];

        #[rustfmt::skip]
        let expect_y: [[f32; 3]; 3] = [
            [1.0, 2.0, 1.0],
            [0.0, 0.0, 0.0],
            [-1.0, -2.0, -1.0],
        ];

        let (sobel2d_x, sobel2d_y) = sobel_2d();

        for ((i, j), result) in sobel2d_x.indexed_iter() {
            assert_relative_eq!(expect_x[i][j], result);
        }

        for ((i, j), result) in sobel2d_y.indexed_iter() {
            assert_relative_eq!(expect_y[i][j], result);
        }
    }
}

