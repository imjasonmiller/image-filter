use ndarray::prelude::*;
use rayon::prelude::*;

mod kernel;

pub fn gaussian_1d(img: &[u8], buf: &mut [u8], width: u32, height: u32, sigma: f64) {
    let (kernel_x, kernel_y) = kernel::gaussian_kernel_1d(sigma);

    // Blur along the x-axis
    convolve(img, buf, width, height, 3, &kernel_x);

    // Blur along the y-axis
    convolve(&buf.to_owned(), buf, width, height, 3, &kernel_y);
}

pub fn gaussian_2d(img: &[u8], buf: &mut [u8], width: u32, height: u32, sigma: f64) {
    let kernel = kernel::gaussian_kernel_2d(sigma);

    convolve(img, buf, width, height, 3, &kernel);
}

pub fn sobel2d(img: &[u8], buf: &mut [u8], width: u32, height: u32) {
    let (kernel_x, kernel_y) = kernel::sobel2d();

    // Find the gradient along the x-axis
    convolve(img, buf, width, height, 3, &kernel_x);

    // Create an extra buffer, as one is required for each gradient
    let mut tmp = img.to_vec();

    // Find the gradient along the y-axis
    convolve(img, &mut tmp[..], width, height, 3, &kernel_y);

    // Apply Pythagorean theorem to both buffers for the gradient magnitude
    buf.par_chunks_mut(3)
        .zip(tmp.par_chunks(3))
        .for_each(|(gx, gy)| {
            for c in 0..3 {
                gx[c] = (f64::from(gx[c]).powi(2) + f64::from(gy[c]).powi(2)).sqrt() as u8
            }
        });
}

#[derive(Debug, Clone, Copy)]
pub struct WeightedElement(f64);

impl Into<u8> for WeightedElement {
    fn into(self) -> u8 {
        self.0.min(255.0).max(0.0) as u8
    }
}

impl std::ops::AddAssign for WeightedElement {
    fn add_assign(&mut self, other: Self) {
        *self = Self(self.0 + other.0);
    }
}

pub fn convolve<T>(
    img_src: &[T],
    img_buf: &mut [T],
    width: u32,
    height: u32,
    channel_count: u8,
    kernel: &Array2<f64>,
) where
    T: Sync + Send + Copy + Into<f64>,
    WeightedElement: Into<T>,
{
    let rows_radius = kernel.nrows() as i32 / 2;
    let cols_radius = kernel.ncols() as i32 / 2;

    img_buf
        .par_chunks_exact_mut(channel_count as usize)
        .enumerate()
        .for_each(|(i, pixel)| {
            let x = i as i32 % width as i32;
            let y = i as i32 / width as i32;

            let mut weights = vec![WeightedElement(0.0); channel_count as usize];

            for (i, kernel_y) in (y - rows_radius..=y + rows_radius).enumerate() {
                for (j, kernel_x) in (x - cols_radius..=x + cols_radius).enumerate() {
                    // Clamp kernel edges to image bounds
                    let edge_x = kernel_x.min(width as i32 - 1).max(0) as usize;
                    let edge_y = kernel_y.min(height as i32 - 1).max(0) as usize;

                    // The pixel x- and y coordinate
                    let p_x = edge_x * channel_count as usize;
                    let p_y = edge_y * channel_count as usize * width as usize;

                    // Range of pixel channel indices
                    let channels = (p_x + p_y)..(p_x + p_y) + channel_count as usize;

                    let pixel = img_src.get(channels).unwrap();

                    for (weight, &channel) in weights.iter_mut().zip(pixel) {
                        *weight += WeightedElement(channel.into() * kernel[[i, j]]);
                    }
                }
            }

            for c in 0..channel_count as usize {
                pixel[c] = weights[c].into();
            }
        });
}

