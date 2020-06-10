use filters::{sobel2d, Image};
use image::{flat::SampleLayout, GenericImage};
use std::fs;

#[test]
fn test_sobel_2d() {
    // Setup
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

    // Test
    sobel2d(&mut actual, Some(1.0));

    // Write buffer to image
    file.copy_from(&buf_write, 0, 0).unwrap();

    // Write file to disk
    file.save("tests/img/actual_sobel_2d_sigma_1.jpg").unwrap();

    // Compare actual file data, as lossy compression might have changed the pixel values
    let expect = fs::read("tests/img/expect_sobel_2d_sigma_1.jpg").unwrap();
    let actual = fs::read("tests/img/actual_sobel_2d_sigma_1.jpg").unwrap();
    assert_eq!(expect, actual);

    // Teardown and remove generated artifacts
    fs::remove_file("tests/img/actual_sobel_2d_sigma_1.jpg").unwrap();
}

