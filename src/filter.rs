use image::RgbImage;

mod kernel;

pub fn gaussian_1d(img: &RgbImage, crop_x: u32, crop_y: u32, buf: &mut RgbImage, sigma: f64) {
    let kernel = kernel::gaussian_kernel_1d(sigma);
    let radius = (kernel.len() - 1) / 2;

    let (w, h) = img.dimensions();

    // Convolve in two passes
    for x in 0..w {
        for y in 0..h {
            let mut sum: [f64; 3] = [0.0; 3];

            for (i, u) in ((x as i32 - radius as i32)..=(x as i32 + radius as i32)).enumerate() {
                // Clamp
                let uc = (u.min((w - 1) as i32)).max(0) as usize;

                let p = img.get_pixel(uc as u32, y as u32);

                sum[0] += p[0] as f64 * kernel[i];
                sum[1] += p[1] as f64 * kernel[i];
                sum[2] += p[2] as f64 * kernel[i];
            }

            let o_pixel = buf.get_pixel_mut(x + crop_x, y + crop_y);
            *o_pixel = image::Rgb([sum[0] as u8, sum[1] as u8, sum[2] as u8]);
        }
    }

    for x in 0..w {
        for y in 0..h {
            let mut sum: [f64; 3] = [0.0; 3];

            for (i, v) in ((y as i32 - radius as i32)..=(y as i32 + radius as i32)).enumerate() {
                // Clamp
                let vc = (v.min((h - 1) as i32)).max(0) as usize;

                let p = buf.get_pixel(x as u32 + crop_x, vc as u32 + crop_y);

                sum[0] += p[0] as f64 * kernel[i];
                sum[1] += p[1] as f64 * kernel[i];
                sum[2] += p[2] as f64 * kernel[i];
            }

            let o_pixel = buf.get_pixel_mut(x + crop_x, y + crop_y);
            *o_pixel = image::Rgb([sum[0] as u8, sum[1] as u8, sum[2] as u8]);
        }
    }
}

pub fn gaussian_2d(img: &RgbImage, crop_x: u32, crop_y: u32, buf: &mut RgbImage, sigma: f64) {
    let kernel = kernel::gaussian_kernel_2d(sigma);
    let radius = (kernel.len() - 1) / 2;

    let (w, h) = img.dimensions();

    for x in 0..w {
        for y in 0..h {
            // Convolve
            let mut sum: [f64; 3] = [0.0; 3];

            for (j, v) in ((y as i32 - radius as i32)..=(y as i32 + radius as i32)).enumerate() {
                for (i, u) in ((x as i32 - radius as i32)..=(x as i32 + radius as i32)).enumerate()
                {
                    // Clamp
                    let uc = (u.min((w - 1) as i32)).max(0) as usize;
                    let vc = (v.min((h - 1) as i32)).max(0) as usize;

                    let p = img.get_pixel(uc as u32, vc as u32);

                    sum[0] += p[0] as f64 * kernel[j][i];
                    sum[1] += p[1] as f64 * kernel[j][i];
                    sum[2] += p[2] as f64 * kernel[j][i];
                }
            }

            let o_pixel = buf.get_pixel_mut(x + crop_x, y + crop_y);

            *o_pixel = image::Rgb([sum[0] as u8, sum[1] as u8, sum[2] as u8]);
        }
    }
}

