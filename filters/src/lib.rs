use ndarray::prelude::*;
use rayon::prelude::*;

mod kernel;

#[derive(Debug, PartialEq, Default)]
pub struct Image<'a, T>
where
    T: Sync + Send + Copy + Into<f64>,
{
    pub buf_read: &'a mut [T],
    pub buf_write: &'a mut [T],
    pub width: u32,
    pub height: u32,
    pub channels: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Weight(f64);

impl Into<u8> for Weight {
    fn into(self) -> u8 {
        self.0.min(255.0).max(0.0) as u8
    }
}

impl std::ops::AddAssign for Weight {
    fn add_assign(&mut self, other: Self) {
        *self = Self(self.0 + other.0);
    }
}

pub fn box_blur_1d<T>(img: &mut Image<T>, radius: usize)
where
    T: Sync + Send + Copy + Into<f64>,
    Weight: Into<T>,
{
    let (kernel_x, kernel_y) = kernel::box_blur_kernel_1d(radius);

    // Blur along the x-axis
    convolve(img, &kernel_x);

    // Use the previous buffer as source for the second pass
    img.buf_read.copy_from_slice(img.buf_write);

    // Blur along the y-axis
    convolve(img, &kernel_y)
}

pub fn box_blur_2d<T>(img: &mut Image<T>, radius: usize)
where
    T: Sync + Send + Copy + Into<f64>,
    Weight: Into<T>,
{
    let kernel = kernel::box_blur_kernel_2d(radius);

    convolve(img, &kernel);
}

pub fn box_blur_1d_gpu<T>(img: &mut Image<T>, radius: usize)
where
    T: Sync + Send + Copy + Into<f64>,
    Weight: Into<T>,
{
}

pub fn gaussian_blur_1d<T>(img: &mut Image<T>, sigma: f64)
where
    T: Sync + Send + Copy + Into<f64>,
    Weight: Into<T>,
{
    let (kernel_x, kernel_y) = kernel::gaussian_blur_kernel_1d(sigma);

    // Blur along the x-axis
    convolve(img, &kernel_x);

    // Use the previous buffer as source for the second pass
    img.buf_read.copy_from_slice(img.buf_write);

    // Blur along the y-axis
    convolve(img, &kernel_y)
}

pub fn gaussian_blur_2d<T>(img: &mut Image<T>, sigma: f64)
where
    T: Sync + Send + Copy + Into<f64>,
    Weight: Into<T>,
{
    let kernel = kernel::gaussian_blur_kernel_2d(sigma);

    convolve(img, &kernel);
}

pub fn sobel2d<T>(img: &mut Image<T>, sigma: Option<f64>)
where
    T: Sync + Send + Copy + Into<f64>,
    Weight: Into<T>,
{
    // Apply Gaussian blur if -s / --sigma is passed
    if let Some(sigma) = sigma {
        gaussian_blur_1d(img, sigma);

        // Write the result to the read buffer for the second pass
        img.buf_read.copy_from_slice(img.buf_write);
    }

    let (kernel_x, kernel_y) = kernel::sobel_2d();

    // Change color to Luma
    // See: https://www.wikiwand.com/en/Grayscale#/Luma_coding_in_video_systems
    img.buf_read.par_chunks_mut(img.channels).for_each(|p| {
        #[rustfmt::skip]
        let y = Weight(
            0.299 * p[0].into() + // R
            0.587 * p[1].into() + // G
            0.114 * p[2].into()   // B
        );

        p[0] = y.into();
        p[1] = y.into();
        p[2] = y.into();
    });

    // Find the gradient along the x-axis
    convolve(img, &kernel_x);

    // Create an extra buffer, as one is required for each gradient
    let mut tmp = img.buf_read.to_vec();

    // Find the gradient along the y-axis
    convolve(
        &mut Image {
            buf_read: &mut img.buf_read,
            buf_write: &mut tmp,
            ..*img
        },
        &kernel_y,
    );

    let Image { channels, .. } = *img;

    // Apply Pythagorean theorem to both buffers for the gradient magnitude
    img.buf_write
        .par_chunks_mut(channels)
        .zip(tmp.par_chunks(channels))
        .for_each(|(gx, gy)| {
            for c in 0..channels {
                gx[c] = Weight(((gx[c].into()).powi(2) + (gy[c].into()).powi(2)).sqrt()).into();
            }
        });
}

pub fn convolve<T>(img: &mut Image<T>, kernel: &Array2<f64>)
where
    T: Sync + Send + Copy + Into<f64>,
    Weight: Into<T>,
{
    let rows_half = kernel.nrows() as isize / 2;
    let cols_half = kernel.ncols() as isize / 2;

    let Image {
        ref buf_read,
        width,
        height,
        channels,
        ..
    } = *img;

    img.buf_write
        // Each thread processes one row of pixels
        .par_chunks_exact_mut(width as usize * channels)
        .enumerate()
        .for_each(|(y, pixels)| {
            // Instantiate weighted sum outside for loop for perf
            let mut weighted_sum = vec![Weight(0.0); channels];

            for (x, pixel) in pixels.chunks_exact_mut(channels).enumerate() {
                // Clear weights
                for weight in weighted_sum.iter_mut() {
                    *weight = Weight(0.0);
                }

                for ((i, j), element) in kernel.indexed_iter() {
                    // Clamp kernel to image bounds
                    let edge_x = (x as isize + (j as isize - cols_half))
                        .min(width as isize - 1)
                        .max(0) as usize;
                    let edge_y = (y as isize + (i as isize - rows_half))
                        .min(height as isize - 1)
                        .max(0) as usize;

                    // Get respective pixel x- and y-coordinate
                    let p_x = edge_x * channels;
                    let p_y = edge_y * channels * width as usize;

                    // Get respective pixel as a slice of all channels
                    let pixel = buf_read.get((p_x + p_y)..(p_x + p_y) + channels).unwrap();

                    for (weight, &channel) in weighted_sum.iter_mut().zip(pixel) {
                        *weight += Weight(channel.into() * element);
                    }
                }

                for (channel, &weight) in pixel.iter_mut().zip(&weighted_sum) {
                    *channel = weight.into();
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;

    #[test]
    fn weightedelement_into_u8() {
        for expect in 0..=u8::MAX {
            let result: u8 = Weight(expect as f64).into();
            assert_eq!(expect, result);
        }
    }

    #[test]
    fn weightedelement_clamp_min() {
        let values: Vec<f64> = vec![-0.0, -0.25, -0.5, -1.0, -1.5, -2.0, -100.0 - 1000.0];

        for value in values.into_iter() {
            assert_eq!(0u8, Weight(value).into());
        }
    }

    #[test]
    fn weightedelement_clamp_max() {
        let values: Vec<f64> = vec![255.0, 255.25, 255.5, 256.0, 300.0, 2000.0, 5000.0];

        for value in values.into_iter() {
            assert_eq!(255u8, Weight(value).into());
        }
    }

    #[test]
    fn weightedelement_add_assign() {
        for n in (0..=255).combinations(2) {
            let expect = (n[0] + n[1]) as f64;

            let mut result = Weight(n[0] as f64);
            result += Weight(n[1] as f64);

            assert_eq!(expect, result.0);
        }
    }

    use image::{flat::SampleLayout, ImageBuffer, Rgb};

    #[test]
    fn text_box_blur_1d_rgb() {
        // Create a 3×1 image
        #[rustfmt::skip]
        let pixels: ImageBuffer<Rgb<u8>, _> =
            ImageBuffer::from_raw(3, 1, vec![
                0, 0, 0,        // R
                255, 255, 255,  // G
                0, 0, 0         // B
            ]).unwrap();

        let SampleLayout {
            width,
            height,
            channels,
            ..
        } = pixels.sample_layout();

        let mut actual = Image {
            buf_read: &mut pixels.clone(),
            buf_write: &mut pixels.clone(),
            width,
            height,
            channels: channels as usize,
        };

        box_blur_1d(&mut actual, 1);

        #[rustfmt::skip]
        assert_eq!(actual.buf_write, [
            85, 85, 85, // R
            85, 85, 85, // G
            85, 85, 85  // B
        ]);
    }

    #[test]
    fn test_image_default() {
        let actual = Image::<u8>::default();
        let expect = Image {
            buf_read: &mut [],
            buf_write: &mut [],
            width: 0,
            height: 0,
            channels: 0,
        };

        assert_eq!(actual, expect);
    }
}
