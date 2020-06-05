use criterion::{criterion_group, criterion_main, Criterion};
use filters::{gaussian_1d, gaussian_2d, Image};
use image::RgbImage;

pub fn bench_gaussian_1d(c: &mut Criterion) {
    let (width, height) = (500, 500);

    let mut source = RgbImage::new(width, height);
    let mut buffer = source.clone();

    let mut image = Image {
        source: source.as_mut(),
        buffer: buffer.as_mut(),
        width,
        height,
        channel_count: 3,
    };

    c.bench_function("gaussian_1d", |b| b.iter(|| gaussian_1d(&mut image, 5.0)));
}

pub fn bench_gaussian_2d(c: &mut Criterion) {
    let (width, height) = (500, 500);

    let mut source = RgbImage::new(width, height);
    let mut buffer = source.clone();

    let mut image = Image {
        source: source.as_mut(),
        buffer: buffer.as_mut(),
        width,
        height,
        channel_count: 3,
    };

    c.bench_function("gaussian_2d", |b| b.iter(|| gaussian_2d(&mut image, 5.0)));
}

criterion_group!(benches, bench_gaussian_1d, bench_gaussian_2d);
criterion_main!(benches);

