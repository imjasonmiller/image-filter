use criterion::{criterion_group, criterion_main, Criterion};
use filters::{gaussian_1d, gaussian_2d};
use image::RgbImage;

pub fn bench_gaussian_1d(c: &mut Criterion) {
    let (width, height) = (500, 500);
    let img = RgbImage::new(width, height);
    let mut buf = img.clone();

    c.bench_function("gaussian_1d", |b| {
        b.iter(|| gaussian_1d(&img, &mut buf, width, height, 5.0))
    });
}

pub fn bench_gaussian_2d(c: &mut Criterion) {
    let (width, height) = (500, 500);
    let img = RgbImage::new(width, height);
    let mut buf = img.clone();

    c.bench_function("gaussian_2d", |b| {
        b.iter(|| gaussian_2d(&img, &mut buf, width, height, 5.0))
    });
}

criterion_group!(benches, bench_gaussian_1d, bench_gaussian_2d);
criterion_main!(benches);

