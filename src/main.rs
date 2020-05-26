use clap::{crate_authors, crate_version, AppSettings::SubcommandRequiredElseHelp, Clap};
use image::{imageops, DynamicImage, ImageBuffer, RgbImage};
use std::path::{Path, PathBuf};

#[derive(Clap)]
#[clap(setting = SubcommandRequiredElseHelp, version = crate_version!(), author = crate_authors!())]
struct Opts {
    #[clap(subcommand)]
    filter: Filter,
    #[clap(short, long, parse(from_os_str))]
    input: PathBuf,
    #[clap(short, long, parse(from_os_str), default_value = "output.jpg")]
    output: PathBuf,
    #[clap(short, long, arg_enum, default_value = "crop")]
    edges: Edges,
    #[clap(short)]
    x: Option<u32>,
    #[clap(short)]
    y: Option<u32>,
    #[clap(short, long)]
    width: Option<u32>,
    #[clap(short, long)]
    height: Option<u32>,
}

#[derive(Clap)]
enum Edges {
    Clamp,
    Crop,
}

#[derive(Clap)]
enum Filter {
    #[clap(name = "gaussian1d")]
    Gaussian1D(Gaussian),
    #[clap(name = "gaussian2d")]
    Gaussian2D(Gaussian),
}

#[derive(Clap, Debug)]
struct Gaussian {
    #[clap(short, long, default_value = "0.84089642")]
    sigma: f64,
}

fn gaussian_kernel_1d(sigma: f64) -> Vec<f64> {
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

fn gaussian_kernel_2d(sigma: f64) -> Vec<Vec<f64>> {
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

fn gaussian_1d(img: &RgbImage, crop_x: u32, crop_y: u32, buf: &mut RgbImage, sigma: f64) {
    let kernel = gaussian_kernel_1d(sigma);
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

fn gaussian_2d(img: &RgbImage, crop_x: u32, crop_y: u32, buf: &mut RgbImage, sigma: f64) {
    let kernel = gaussian_kernel_2d(sigma);
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

fn read_image(path: &PathBuf) -> Result<image::DynamicImage, std::io::ErrorKind> {
    if !path.exists() {
        return Err(std::io::ErrorKind::NotFound);
    }

    if path.extension().unwrap() != "jpg" {
        return Err(std::io::ErrorKind::InvalidInput);
    }

    match image::open(path) {
        Ok(img) => Ok(img),
        Err(_) => Err(std::io::ErrorKind::InvalidInput),
    }
}

fn write_image(buf: &RgbImage, path: &PathBuf) {
    if buf.save(&path).is_err() {
        panic!("could not save");
    }
}

fn main() -> Result<(), std::io::ErrorKind> {
    let opts: Opts = Opts::parse();

    let o_path = Path::new(&opts.output);
    o_path.exists();

    let mut img = match read_image(&opts.input) {
        Ok(DynamicImage::ImageRgb8(i)) => i,
        _ => return Err(std::io::ErrorKind::InvalidInput),
    };

    let (img_w, img_h) = img.dimensions();

    let crop_x = opts.x.unwrap();
    let crop_y = opts.y.unwrap();
    let crop_w = match opts.width {
        Some(w) if w > img_w => panic!("crop width exceeds image bounds"),
        Some(w) => w,
        None => img_w,
    };
    let crop_h = match opts.height {
        Some(h) if h > img_h => panic!("crop height exceeds image bounds"),
        Some(h) => h,
        None => img_h,
    };

    let mut buf = img.clone();
    let view = imageops::crop(&mut img, crop_x, crop_y, crop_w, crop_h);

    use std::time::Instant;
    let start = Instant::now();

    println!("Filtering image...");

    match opts.filter {
        Filter::Gaussian1D(Gaussian { sigma }) => 
            gaussian_1d(&view.to_image(), crop_x, crop_y, &mut buf, sigma)
,
        Filter::Gaussian2D(Gaussian { sigma }) => {
            gaussian_2d(&view.to_image(), crop_x, crop_y, &mut buf, sigma)
        }
    }

    println!("Total time: {:?}", start.elapsed());

    write_image(&buf, &opts.output);

    Ok(())
}
