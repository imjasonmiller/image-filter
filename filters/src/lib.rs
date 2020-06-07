use ndarray::prelude::*;
use rayon::prelude::*;

mod kernel;

pub struct Image<'a, T>
where
    T: Sync + Send + Copy + Into<f64>,
{
    pub source: &'a mut [T],
    pub buffer: &'a mut [T],
    pub width: u32,
    pub height: u32,
    pub channel_count: u8,
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

pub fn box_blur_1d<T>(img: &mut Image<T>, radius: usize)
where
    T: Sync + Send + Copy + Into<f64>,
    WeightedElement: Into<T>,
{
    let (kernel_x, kernel_y) = kernel::box_blur_kernel_1d(radius);

    // Blur along the x-axis
    convolve(img, &kernel_x);

    // Use the previous buffer as source for the second pass
    img.source.copy_from_slice(img.buffer);

    // Blur along the y-axis
    convolve(img, &kernel_y)
}

pub fn box_blur_2d<T>(img: &mut Image<T>, radius: usize)
where
    T: Sync + Send + Copy + Into<f64>,
    WeightedElement: Into<T>,
{
    let kernel = kernel::box_blur_kernel_2d(radius);

    convolve(img, &kernel);
}

pub fn gaussian_blur_1d<T>(img: &mut Image<T>, sigma: f64)
where
    T: Sync + Send + Copy + Into<f64>,
    WeightedElement: Into<T>,
{
    let (kernel_x, kernel_y) = kernel::gaussian_blur_kernel_1d(sigma);

    // Blur along the x-axis
    convolve(img, &kernel_x);

    // Use the previous buffer as source for the second pass
    img.source.copy_from_slice(img.buffer);

    // Blur along the y-axis
    convolve(img, &kernel_y)
}

pub fn gaussian_blur_2d<T>(img: &mut Image<T>, sigma: f64)
where
    T: Sync + Send + Copy + Into<f64>,
    WeightedElement: Into<T>,
{
    let kernel = kernel::gaussian_blur_kernel_2d(sigma);

    convolve(img, &kernel);
}

pub fn sobel2d<T>(img: &mut Image<T>, sigma: Option<f64>)
where
    T: Sync + Send + Copy + Into<f64>,
    WeightedElement: Into<T>,
{
    if let Some(sigma) = sigma {
        gaussian_blur_1d(img, sigma);

        // Use the previous buffer as source for the second pass
        img.source.copy_from_slice(img.buffer);
    }

    let (kernel_x, kernel_y) = kernel::sobel2d();

    // Change color to Luma
    // See: https://www.wikiwand.com/en/Grayscale#/Luma_coding_in_video_systems
    img.source.par_chunks_mut(3).for_each(|p| {
        let y = WeightedElement(0.299 * p[0].into() + 0.587 * p[1].into() + 0.114 * p[2].into());
        p[0] = y.into();
        p[1] = y.into();
        p[2] = y.into();
    });

    // Find the gradient along the x-axis
    convolve(img, &kernel_x);

    // Create an extra buffer, as one is required for each gradient
    let mut tmp = img.source.to_vec();

    // Find the gradient along the y-axis
    convolve(
        &mut Image {
            source: &mut img.source,
            // buffer: &mut tmp[..],
            buffer: &mut tmp,
            ..*img
        },
        &kernel_y,
    );

    // Apply Pythagorean theorem to both buffers for the gradient magnitude
    img.buffer
        .par_chunks_mut(3)
        .zip(tmp.par_chunks(3))
        .for_each(|(gx, gy)| {
            for c in 0..3 {
                gx[c] = WeightedElement(((gx[c].into()).powi(2) + (gy[c].into()).powi(2)).sqrt())
                    .into();
            }
        });
}

pub fn convolve<T>(img: &mut Image<T>, kernel: &Array2<f64>)
where
    T: Sync + Send + Copy + Into<f64>,
    WeightedElement: Into<T>,
{
    let rows_half = kernel.nrows() as i32 / 2;
    let cols_half = kernel.ncols() as i32 / 2;

    let Image {
        ref source,
        width,
        height,
        channel_count,
        ..
    } = *img;

    img.buffer
        .par_chunks_exact_mut(channel_count as usize)
        .enumerate()
        .for_each(|(i, pixel)| {
            // Pixel x- and y-coordinate
            let x = i as i32 % width as i32;
            let y = i as i32 / width as i32;

            let mut weights = vec![WeightedElement(0.0); channel_count as usize];

            for (i, kernel_y) in (y - rows_half..=y + rows_half).enumerate() {
                for (j, kernel_x) in (x - cols_half..=x + cols_half).enumerate() {
                    // Clamp kernel edges to image bounds
                    let edge_x = kernel_x.min(width as i32 - 1).max(0) as usize;
                    let edge_y = kernel_y.min(height as i32 - 1).max(0) as usize;

                    // Kernel x- and y coordinate
                    let p_x = edge_x * channel_count as usize;
                    let p_y = edge_y * channel_count as usize * width as usize;

                    // Range of pixel channel indices
                    let channels = (p_x + p_y)..(p_x + p_y) + channel_count as usize;

                    let pixel = source.get(channels).unwrap();

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

