use filters::{box_blur_1d, box_blur_2d, Image};
use image::{flat::SampleLayout, GenericImage};
use std::fs;

#[test]
fn test_box_blur_1d() {
    // Setup
    let mut file = image::open("tests/fixtures/input.jpg").unwrap();

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
    box_blur_1d(&mut actual, 3);

    // Write buffer to image
    file.copy_from(&buf_write, 0, 0).unwrap();

    // Write file to disk
    file.save("tests/fixtures/actual_box_blur_1d_radius_3.jpg")
        .unwrap();

    // Compare actual file data, as lossy compression might have changed the pixel values
    let expect = fs::read("tests/fixtures/expect_box_blur_1d_radius_3.jpg").unwrap();
    let actual = fs::read("tests/fixtures/actual_box_blur_1d_radius_3.jpg").unwrap();
    assert_eq!(expect, actual);

    // Teardown and remove generated artifacts
    fs::remove_file("tests/fixtures/actual_box_blur_1d_radius_3.jpg").unwrap();
}

#[test]
fn test_box_blur_2d() {
    // Setup
    let mut file = image::open("tests/fixtures/input.jpg").unwrap();

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
    box_blur_2d(&mut actual, 3);

    // Write buffer to image
    file.copy_from(&buf_write, 0, 0).unwrap();

    // Write file to disk
    file.save("tests/fixtures/actual_box_blur_2d_radius_3.jpg")
        .unwrap();

    // Compare actual file data, as lossy compression might have changed the pixel values
    let expect = fs::read("tests/fixtures/expect_box_blur_2d_radius_3.jpg").unwrap();
    let actual = fs::read("tests/fixtures/actual_box_blur_2d_radius_3.jpg").unwrap();

    assert_eq!(expect, actual);

    // Teardown and remove generated artifacts
    fs::remove_file("tests/fixtures/actual_box_blur_2d_radius_3.jpg").unwrap();
}

