use clap::{crate_authors, crate_version, AppSettings::SubcommandRequiredElseHelp, Clap};
use image::{imageops, DynamicImage, ImageBuffer, RgbImage};
use std::path::{Path, PathBuf};

mod filter;

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
        Filter::Gaussian1D(Gaussian { sigma }) => {
            filter::gaussian_1d(&view.to_image(), crop_x, crop_y, &mut buf, sigma)
        }
        Filter::Gaussian2D(Gaussian { sigma }) => {
            filter::gaussian_2d(&view.to_image(), crop_x, crop_y, &mut buf, sigma)
        }
    }

    println!("Total time: {:?}", start.elapsed());

    write_image(&buf, &opts.output);

    Ok(())
}

