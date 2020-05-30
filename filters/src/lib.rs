use rayon::prelude::*;

mod kernel;

fn transpose_2d(mat_a: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let (rows, cols) = (mat_a.len(), mat_a[0].len());
    let mut mat_b = vec![vec![0.0; rows]; cols];

    for j in 0..rows {
        for i in 0..cols {
            mat_b[i][j] = mat_a[j][i]
        }
    }

    mat_b
}

pub fn gaussian_1d(img: &[u8], buf: &mut [u8], width: u32, height: u32, sigma: f64) {
    let kernel = kernel::gaussian_kernel_1d(sigma);

    convolve(img, buf, width, height, 3, &kernel);
    convolve(
        &buf.to_owned(),
        buf,
        width,
        height,
        3,
        &transpose_2d(kernel),
    );
}

pub fn gaussian_2d(img: &[u8], buf: &mut [u8], width: u32, height: u32, sigma: f64) {
    let kernel = kernel::gaussian_kernel_2d(sigma);
    convolve(img, buf, width, height, 3, &kernel);
}

#[derive(Debug, Clone, Copy)]
pub struct WeightedElement(f64);

// FIXME: Possible perf improvement by removing minimum bound
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
    kernel: &Vec<Vec<f64>>,
) where
    T: Sync + Send + Copy + Into<f64>,
    WeightedElement: Into<T>,
{
    let kernel_cols = kernel[0].len() as i32 / 2;
    let kernel_rows = kernel.len() as i32 / 2;

    img_buf
        .par_chunks_mut(channel_count as usize)
        .enumerate()
        .for_each(|(i, pixel)| {
            let x = i as i32 % width as i32;
            let y = i as i32 / width as i32;

            let mut weights = vec![WeightedElement(0.0); channel_count as usize];

            for (i, kernel_y) in (y - kernel_rows..=y + kernel_rows).enumerate() {
                for (j, kernel_x) in (x - kernel_cols..=x + kernel_cols).enumerate() {
                    let edge_x = kernel_x.min(width as i32 - 1).max(0) as usize;
                    let edge_y = kernel_y.min(height as i32 - 1).max(0) as usize;

                    let start = edge_y * width as usize * channel_count as usize
                        + edge_x * channel_count as usize;
                    let value = img_src.get(start..start + channel_count as usize).unwrap();

                    for c in 0..channel_count as usize {
                        weights[c] += WeightedElement(value[c].into() * kernel[i][j]);
                    }
                }
            }

            for c in 0..channel_count as usize {
                pixel[c] = weights[c].into();
            }
        });
}

