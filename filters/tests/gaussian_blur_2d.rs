use filters::{gaussian_blur_2d, Image};
use image::{flat::SampleLayout, GenericImage};
use std::fs;

// Remove generated file during a test
fn teardown() {
    fs::remove_file("tests/img/actual.jpg").unwrap();
}

#[test]
fn test_gaussian_blur_2d() {
    let mut file = image::open("tests/img/input.jpg").unwrap();

    let mut buf_read = file.clone().into_rgba();
    let mut buf_write = file.clone().into_rgba();

    let SampleLayout {
        width,
        height,
        channels,
        ..
    } = buf_read.sample_layout();

    let mut actual = Image {
        buf_read: &mut buf_read,
        buf_write: &mut buf_write,
        width,
        height,
        channels: channels as usize,
    };

    gaussian_blur_2d(&mut actual, 3.0);

    // Place generated buffer on top of image
    file.copy_from(&buf_write, 0, 0).unwrap();

    // Write out file
    file.save("tests/img/actual.jpg").unwrap();

    // Compare actual file data, as lossy compression might have changed the pixel values
    let expect = fs::read("tests/img/gaussian_blur_2d_sigma_3.jpg").unwrap();
    let actual = fs::read("tests/img/actual.jpg").unwrap();

    assert_eq!(expect, actual);

    teardown();
}
