use clap::{crate_authors, crate_version, AppSettings::SubcommandRequiredElseHelp, Clap};
use filters::{box_blur_1d, box_blur_2d, gaussian_blur_1d, gaussian_blur_2d, sobel2d, Image};
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
    // #[clap(short, long, arg_enum, default_value = "crop")]
    // edges: Edges,
    #[clap(short, default_value = "0")]
    x: u32,
    #[clap(short, default_value = "0")]
    y: u32,
    #[clap(short, long)]
    width: Option<u32>,
    #[clap(short, long)]
    height: Option<u32>,
    #[clap(short, long)]
    verbose: bool,
}

#[derive(Clap)]
enum Edges {
    Clamp,
    Crop,
}

#[derive(Clap)]
enum Filter {
    #[clap(name = "box_blur_1d")]
    BoxBlur1D(BoxBlur),
    #[clap(name = "box_blur_2d")]
    BoxBlur2D(BoxBlur),
    #[clap(name = "gaussian_blur_1d")]
    GaussianBlur1D(GaussianBlur),
    #[clap(name = "gaussian_blur_2d")]
    GaussianBlur2D(GaussianBlur),
    #[clap(name = "sobel_2d")]
    Sobel2D(Sobel),
}

#[derive(Clap, Debug)]
struct BoxBlur {
    #[clap(short, long, default_value = "1")]
    radius: usize,
}

#[derive(Clap, Debug)]
struct GaussianBlur {
    #[clap(short, long, default_value = "0.84089642")]
    sigma: f64,
}

#[derive(Clap, Debug)]
struct Sobel {
    #[clap(short, long)]
    sigma: Option<f64>,
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

    use std::time::Instant;
    let start = Instant::now();

    if opts.verbose {
        println!("Filtering image...");
    }

    let crop = imageops::crop(&mut img, crop_x, crop_y, crop_w, crop_h);

    let mut source = crop.to_image();
    let mut buffer = crop.to_image();

    let mut image = Image {
        source: source.as_mut(),
        buffer: buffer.as_mut(),
        width: crop_w,
        height: crop_h,
        channel_count: 3,
    };

    match opts.filter {
        Filter::BoxBlur1D(BoxBlur { radius }) => box_blur_1d(&mut image, radius),
        Filter::BoxBlur2D(BoxBlur { radius }) => box_blur_2d(&mut image, radius),
        Filter::GaussianBlur1D(GaussianBlur { sigma }) => gaussian_blur_1d(&mut image, sigma),
        Filter::GaussianBlur2D(GaussianBlur { sigma }) => gaussian_blur_2d(&mut image, sigma),
        Filter::Sobel2D(Sobel { sigma }) => sobel2d(&mut image, sigma),
    }

    if opts.verbose {
        println!("Total time: {:?}", start.elapsed());
    }

    let result: ImageBuffer<image::Rgb<u8>, _> =
        ImageBuffer::from_raw(image.width, image.height, image.buffer).unwrap();

    imageops::replace(&mut img, &result, crop_x, crop_y);
    write_image(&img, &opts.output);

    Ok(())
}

