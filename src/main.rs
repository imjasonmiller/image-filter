use clap::{
    crate_authors, crate_version,
    AppSettings::{ColoredHelp, DeriveDisplayOrder, SubcommandRequiredElseHelp},
    Clap,
};
use filters::{
    box_blur_1d, box_blur_1d_gpu, box_blur_2d, gaussian_blur_1d, gaussian_blur_2d, sobel2d, Image,
};
use image::{
    flat::SampleLayout, imageops, GenericImage, GenericImageView, ImageBuffer, Rgba, SubImage,
};
use std::path::PathBuf;

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
    #[clap(short, default_value = "0", about = "Crop x-coordinate")]
    x: u32,
    #[clap(short, default_value = "0", about = "Crop y-coordinate")]
    y: u32,
    #[clap(short, long, about = "Crop width")]
    width: Option<u32>,
    #[clap(short, long, about = "Crop height")]
    height: Option<u32>,
    #[clap(short, long, about = "Force output file overwrite")]
    force: bool,
    #[clap(short, long, about = "Increase logging verbosity")]
    verbose: bool,
}

#[derive(Clap)]
enum Edges {
    Clamp,
    Crop,
}

#[derive(Clap)]
#[clap(setting = ColoredHelp, setting = DeriveDisplayOrder)]
enum Filter {
    #[clap(name = "box_blur_1d")]
    BoxBlur1D(BoxBlur),
    #[clap(name = "box_blur_1d_gpu")]
    BoxBlur1DGPU(BoxBlur),
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

fn crop_image<I>(
    img: &I,
    crop_x: u32,
    crop_y: u32,
    crop_w: Option<u32>,
    crop_h: Option<u32>,
) -> SubImage<&I>
where
    I: GenericImageView,
{
    let (width, height) = img.dimensions();

    // If no crop width or height was specified, use the remaining space.
    let crop_w = crop_w.unwrap_or(width - crop_x);
    let crop_h = crop_h.unwrap_or(height - crop_y);

    // Panic if crop exceeds image bounds
    assert!(crop_x <= width, "-x exceeds image bounds");
    assert!(crop_y <= height, "-y exceeds image bounds");
    assert!(crop_w + crop_x <= width, "--width exceeds image bounds");
    assert!(crop_h + crop_y <= height, "--height exceeds image bounds");

    imageops::crop_imm(img, crop_x, crop_y, crop_w, crop_h)
}

fn main() {
    let opts: Opts = Opts::parse();

    assert!(opts.input.exists(), "input not found");
    assert!(
        opts.force || !opts.output.exists(),
        "output already exists, if you wish to overwrite use --force"
    );

    let mut file = image::open(opts.input).expect("File could not be opened");

    let crop = crop_image(&file, opts.x, opts.y, opts.width, opts.height);

    // Create the read and write buffers
    let mut buf_read: ImageBuffer<Rgba<_>, _> = crop.to_image();
    let mut buf_write: ImageBuffer<Rgba<_>, _> = crop.to_image();

    let SampleLayout {
        width,
        height,
        channels,
        ..
    } = buf_read.sample_layout();

    let mut image = Image {
        buf_read: buf_read.as_mut(),
        buf_write: buf_write.as_mut(),
        width,
        height,
        channels: channels as usize,
    };

    // Measure elapsed time
    let start = std::time::Instant::now();

    if opts.verbose {
        eprintln!("Image channels: {}", channels);
        eprintln!("Filtering...");
    }

    match opts.filter {
        Filter::BoxBlur1D(BoxBlur { radius }) => box_blur_1d(&mut image, radius),
        Filter::BoxBlur1DGPU(BoxBlur { radius }) => {
            futures::executor::block_on(box_blur_1d_gpu(&mut image, radius))
        }
        Filter::BoxBlur2D(BoxBlur { radius }) => box_blur_2d(&mut image, radius),
        Filter::GaussianBlur1D(GaussianBlur { sigma }) => gaussian_blur_1d(&mut image, sigma),
        Filter::GaussianBlur2D(GaussianBlur { sigma }) => gaussian_blur_2d(&mut image, sigma),
        Filter::Sobel2D(Sobel { sigma }) => sobel2d(&mut image, sigma),
    }

    if opts.verbose {
        eprintln!("Filter duration: {:?}", start.elapsed());
    }

    // Overlay result on top of original image
    file.copy_from(&buf_write, opts.x, opts.y).unwrap();

    // Write image
    file.save(&opts.output).expect("could not save image");
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    #[test]
    fn test_valid_crop() {
        // Create a 3×1 image
        let pixels: ImageBuffer<Rgb<u8>, _> =
            ImageBuffer::from_raw(3, 1, vec![0, 0, 0, 255, 255, 255, 0, 0, 0]).unwrap();

        let actual = crop_image(&pixels, 1, 0, Some(1), Some(1))
            .to_image()
            .into_raw();

        assert_eq!(actual, [255, 255, 255]);
    }
}
