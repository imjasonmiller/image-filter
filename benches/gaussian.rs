use criterion::{criterion_group, criterion_main, Criterion};
use filters::{gaussian_1d, gaussian_2d};
use image::RgbImage;

pub fn gaussian_benchmark(c: &mut Criterion) {
    let img = RgbImage::new(250, 250);
    let mut buf = RgbImage::new(250, 250);

    let mut group = c.benchmark_group("gaussian_250x250");

    group.bench_function("gaussian_kernel_1d", |b| {
        b.iter(|| gaussian_1d(&img, 0, 0, &mut buf, 1.0))
    });

    group.bench_function("gaussian_kernel_2d", |b| {
        b.iter(|| gaussian_2d(&img, 0, 0, &mut buf, 1.0))
    });

    group.finish();
}

criterion_group!(benches, gaussian_benchmark);
criterion_main!(benches);

