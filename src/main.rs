use clap::{crate_authors, crate_version, AppSettings::SubcommandRequiredElseHelp, Clap};
use filters::*;
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
    #[clap(short, default_value = "0")]
    x: u32,
    #[clap(short, default_value = "0")]
    y: u32,
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
        panic!("could not save image");
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

    let crop_x = match opts.x {
        n if n >= img_w => {
            println!("crop -x set to 0, as crop exceeds image width");
            0
        }
        n => n,
    };
    let crop_y = match opts.y {
        n if n >= img_h => {
            println!("crop -y set to 0, as crop exceeds image height");
            0
        }
        n => n,
    };
    let crop_w = opts.width.map_or(img_w, |n| {
        if crop_x + n > img_w {
            println!("crop -w set to {}, as crop exceeds image width", img_w - n);
            return img_w - n;
        }
        n
    });
    let crop_h = opts.height.map_or(img_h, |n| {
        if crop_y + n > img_h {
            println!("crop -h set to {}, as crop exceeds image height", img_h - n);
            return img_h - n;
        }
        n
    });

    let mut buf = img.clone();
    let view = imageops::crop(&mut img, crop_x, crop_y, crop_w, crop_h);

    use std::time::Instant;
    let start = Instant::now();

    println!("Filtering image...");

    match opts.filter {
        Filter::Gaussian1D(Gaussian { sigma }) => {
            filters::gaussian_1d(&view.to_image(), crop_x, crop_y, &mut buf, sigma)
        }
        Filter::Gaussian2D(Gaussian { sigma }) => {
            filters::gaussian_2d(&view.to_image(), crop_x, crop_y, &mut buf, sigma)
        }
    }

    println!("Total time: {:?}", start.elapsed());

    write_image(&buf, &opts.output);

    Ok(())
}
