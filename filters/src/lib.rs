use ndarray::prelude::*;
use rayon::prelude::*;

mod kernel;

#[derive(Debug, PartialEq, Default)]
pub struct Image<'a, T>
where
    T: Sync + Send + Copy + Into<f32>,
{
    pub buf_read: &'a mut [T],
    pub buf_write: &'a mut [T],
    pub width: u32,
    pub height: u32,
    pub channels: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Weight(f32);

impl Into<u8> for Weight {
    fn into(self) -> u8 {
        self.0.min(255.0).max(0.0) as u8
    }
}

impl std::ops::AddAssign for Weight {
    fn add_assign(&mut self, other: Self) {
        *self = Self(self.0 + other.0);
    }
}

pub fn box_blur_1d<T>(img: &mut Image<T>, radius: usize)
where
    T: Sync + Send + Copy + Into<f32>,
    Weight: Into<T>,
{
    let (kernel_x, kernel_y) = kernel::box_blur_kernel_1d(radius);

    // Blur along the x-axis
    convolve(img, &kernel_x);

    // Use the previous buffer as source for the second pass
    img.buf_read.copy_from_slice(img.buf_write);

    // Blur along the y-axis
    convolve(img, &kernel_y)
}

pub fn box_blur_2d<T>(img: &mut Image<T>, radius: usize)
where
    T: Sync + Send + Copy + Into<f32>,
    Weight: Into<T>,
{
    let kernel = kernel::box_blur_kernel_2d(radius);

    convolve(img, &kernel);
}

pub async fn box_blur_1d_gpu<'a, T>(image: &mut Image<'a, T>, radius: usize)
where
    T: Sync + Send + Copy + Into<f32> + bytemuck::Pod + std::fmt::Debug,
    Weight: Into<T>,
{
    let kernel = kernel::box_blur_kernel_2d(radius);
    let kernel_slice = kernel.iter().map(|x| *x as f32).collect::<Vec<_>>();

    // Create a handle to the graphics/compute device
    let adapter = wgpu::Instance::new()
        .request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::Default,
                compatible_surface: None,
            },
            wgpu::UnsafeExtensions::disallow(),
            wgpu::BackendBit::PRIMARY,
        )
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                extensions: wgpu::Extensions::empty(),
                limits: wgpu::Limits::default(),
                shader_validation: true,
            },
            None,
        )
        .await
        .unwrap();

    let image_bytes = bytemuck::cast_slice(image.buf_read.as_ref());

    // Create pipeline layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        bindings: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                ..Default::default()
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                ..Default::default()
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    dimension: wgpu::TextureViewDimension::D2,
                    component_type: wgpu::TextureComponentType::Float,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    readonly: false,
                },
                ..Default::default()
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout],
    });

    // Create the kernel buffers
    let kernel_radius_buffer = device.create_buffer_with_data(
        bytemuck::cast_slice(&[kernel.ncols() as i32 / 2]),
        wgpu::BufferUsage::UNIFORM,
    );

    let kernel_weights_buffer = device.create_buffer_with_data(
        bytemuck::cast_slice(&kernel_slice),
        wgpu::BufferUsage::UNIFORM,
    );

    // Create the texture
    let texture_extent = wgpu::Extent3d {
        width: image.width,
        height: image.height,
        depth: 1,
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: texture_extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsage::COPY_DST
            | wgpu::TextureUsage::COPY_SRC
            | wgpu::TextureUsage::STORAGE,
    });

    let texture_view = texture.create_default_view();

    queue.write_texture(
        wgpu::TextureCopyView {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        &image_bytes,
        wgpu::TextureDataLayout {
            offset: 0,
            bytes_per_row: image.width * 4,
            rows_per_image: 0,
        },
        texture_extent,
    );

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        bindings: &[
            wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(kernel_radius_buffer.slice(..)),
            },
            wgpu::Binding {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(kernel_weights_buffer.slice(..)),
            },
            wgpu::Binding {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },
        ],
    });

    // Load compute shader
    let compute_shader = include_bytes!("convolve.comp.spv");
    let compute_module = device.create_shader_module(
        &wgpu::read_spirv(std::io::Cursor::new(&compute_shader[..])).unwrap(),
    );

    // Create the compute pipeline
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        layout: &pipeline_layout,
        compute_stage: wgpu::ProgrammableStageDescriptor {
            module: &compute_module,
            entry_point: "main",
        },
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let mut cpass = encoder.begin_compute_pass();
        cpass.set_pipeline(&compute_pipeline);
        // Assign bind group to set 0
        cpass.set_bind_group(0, &bind_group, &[]);
        // Assign a multiple of local_size workgroups to the texture
        let local_size = 32;
        cpass.dispatch(
            (image.width + local_size - 1) / local_size,
            (image.height + local_size - 1) / local_size,
            1,
        );
    }

    // Create buffer to write the computed result to
    let texture_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (4 * 768 * image.height) as wgpu::BufferAddress,
        usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_texture_to_buffer(
        wgpu::TextureCopyView {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        wgpu::BufferCopyView {
            buffer: &texture_output_buffer,
            layout: wgpu::TextureDataLayout {
                offset: 0,
                bytes_per_row: 768 * std::mem::size_of::<u32>() as u32,
                rows_per_image: image.height,
            },
        },
        texture_extent,
    );

    queue.submit(Some(encoder.finish()));

    // Note that we're not calling `.await` here.
    let texture_output_slice = texture_output_buffer.slice(..);
    let texture_output_future = texture_output_slice.map_async(wgpu::MapMode::Read);

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device.poll(wgpu::Maintain::Wait);

    if let Ok(()) = texture_output_future.await {
        let data = texture_output_slice.get_mapped_range();

        // TODO: Handle arbitrary image width and height
        for row in 0..720 {
            let bytes = bytemuck::cast_slice(&data[row * 768 * 4..row * 768 * 4 + 720 * 4]);
            // let bytes = bytemuck::cast_slice(&example[row * 720 * 4..row * 720 * 4 + 720 * 4]);
            image.buf_write[row * 720 * 4..row * 720 * 4 + 720 * 4].copy_from_slice(bytes);
        }

        drop(data);
        texture_output_buffer.unmap();
    } else {
        panic!("failed to run compute on gpu!")
    }
}

pub fn gaussian_blur_1d<T>(img: &mut Image<T>, sigma: f32)
where
    T: Sync + Send + Copy + Into<f32>,
    Weight: Into<T>,
{
    let (kernel_x, kernel_y) = kernel::gaussian_blur_kernel_1d(sigma);

    // Blur along the x-axis
    convolve(img, &kernel_x);

    // Use the previous buffer as source for the second pass
    img.buf_read.copy_from_slice(img.buf_write);

    // Blur along the y-axis
    convolve(img, &kernel_y)
}

pub fn gaussian_blur_2d<T>(img: &mut Image<T>, sigma: f32)
where
    T: Sync + Send + Copy + Into<f32>,
    Weight: Into<T>,
{
    let kernel = kernel::gaussian_blur_kernel_2d(sigma);

    convolve(img, &kernel);
}

pub fn sobel2d<T>(img: &mut Image<T>, sigma: Option<f32>)
where
    T: Sync + Send + Copy + Into<f32>,
    Weight: Into<T>,
{
    // Apply Gaussian blur if -s / --sigma is passed
    if let Some(sigma) = sigma {
        gaussian_blur_1d(img, sigma);

        // Write the result to the read buffer for the second pass
        img.buf_read.copy_from_slice(img.buf_write);
    }

    let (kernel_x, kernel_y) = kernel::sobel_2d();

    // Change color to Luma
    // See: https://www.wikiwand.com/en/Grayscale#/Luma_coding_in_video_systems
    img.buf_read.par_chunks_mut(img.channels).for_each(|p| {
        #[rustfmt::skip]
        let y = Weight(
            0.299 * p[0].into() + // R
            0.587 * p[1].into() + // G
            0.114 * p[2].into()   // B
        );

        p[0] = y.into();
        p[1] = y.into();
        p[2] = y.into();
    });

    // Find the gradient along the x-axis
    convolve(img, &kernel_x);

    // Create an extra buffer, as one is required for each gradient
    let mut tmp = img.buf_read.to_vec();

    // Find the gradient along the y-axis
    convolve(
        &mut Image {
            buf_read: &mut img.buf_read,
            buf_write: &mut tmp,
            ..*img
        },
        &kernel_y,
    );

    let Image { channels, .. } = *img;

    // Apply Pythagorean theorem to both buffers for the gradient magnitude
    img.buf_write
        .par_chunks_mut(channels)
        .zip(tmp.par_chunks(channels))
        .for_each(|(gx, gy)| {
            for c in 0..channels {
                gx[c] = Weight(((gx[c].into()).powi(2) + (gy[c].into()).powi(2)).sqrt()).into();
            }
        });
}

pub fn convolve<T>(img: &mut Image<T>, kernel: &Array2<f32>)
where
    T: Sync + Send + Copy + Into<f32>,
    Weight: Into<T>,
{
    let rows_half = kernel.nrows() as isize / 2;
    let cols_half = kernel.ncols() as isize / 2;

    let Image {
        ref buf_read,
        width,
        height,
        channels,
        ..
    } = *img;

    img.buf_write
        // Process one row of pixels for each thread
        .par_chunks_exact_mut(width as usize * channels)
        .enumerate()
        .for_each(|(y, pixels)| {
            // Save on instantiation by only cloning the initialized array
            let weighted_sum = [Weight(0.0); 4];

            pixels
                .par_chunks_exact_mut(channels)
                .map(|pixel| (pixel, weighted_sum.clone()))
                .enumerate()
                .for_each(|(x, (pixel, mut weighted_sum))| {
                    for ((i, j), kernel_element) in kernel.indexed_iter() {
                        // Clamp kernel to image bounds
                        let edge_x = (x as isize + (j as isize - cols_half))
                            .min(width as isize - 1)
                            .max(0) as usize;
                        let edge_y = (y as isize + (i as isize - rows_half))
                            .min(height as isize - 1)
                            .max(0) as usize;

                        // Get pixel x- and y-coordinate
                        let p_x = edge_x * channels;
                        let p_y = edge_y * channels * width as usize;

                        // Get pixel channels as a slice
                        let pixel = buf_read.get((p_x + p_y)..(p_x + p_y) + channels).unwrap();

                        for (weight, &channel) in weighted_sum.iter_mut().zip(pixel) {
                            *weight += Weight(channel.into() * kernel_element);
                        }
                    }

                    for (channel, &weight) in pixel.iter_mut().zip(&weighted_sum) {
                        *channel = weight.into();
                    }
                });
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;

    #[test]
    fn weightedelement_into_u8() {
        for expect in 0..=u8::MAX {
            let result: u8 = Weight(expect as f32).into();
            assert_eq!(expect, result);
        }
    }

    #[test]
    fn weightedelement_clamp_min() {
        let values: Vec<f32> = vec![-0.0, -0.25, -0.5, -1.0, -1.5, -2.0, -100.0 - 1000.0];

        for value in values.into_iter() {
            assert_eq!(0u8, Weight(value).into());
        }
    }

    #[test]
    fn weightedelement_clamp_max() {
        let values: Vec<f32> = vec![255.0, 255.25, 255.5, 256.0, 300.0, 2000.0, 5000.0];

        for value in values.into_iter() {
            assert_eq!(255u8, Weight(value).into());
        }
    }

    #[test]
    fn weightedelement_add_assign() {
        for n in (0..=255).combinations(2) {
            let expect = (n[0] + n[1]) as f32;

            let mut result = Weight(n[0] as f32);
            result += Weight(n[1] as f32);

            assert_eq!(expect, result.0);
        }
    }

    use image::{flat::SampleLayout, ImageBuffer, Rgb};

    #[test]
    fn text_box_blur_1d_rgb() {
        // Create a 3×1 image
        #[rustfmt::skip]
        let pixels: ImageBuffer<Rgb<u8>, _> =
            ImageBuffer::from_raw(3, 1, vec![
                0, 0, 0,        // R
                255, 255, 255,  // G
                0, 0, 0         // B
            ]).unwrap();

        let SampleLayout {
            width,
            height,
            channels,
            ..
        } = pixels.sample_layout();

        let mut actual = Image {
            buf_read: &mut pixels.clone(),
            buf_write: &mut pixels.clone(),
            width,
            height,
            channels: channels as usize,
        };

        box_blur_1d(&mut actual, 1);

        #[rustfmt::skip]
        assert_eq!(actual.buf_write, [
            85, 85, 85, // R
            85, 85, 85, // G
            85, 85, 85  // B
        ]);
    }

    #[test]
    fn test_image_default() {
        let actual = Image::<u8>::default();
        let expect = Image {
            buf_read: &mut [],
            buf_write: &mut [],
            width: 0,
            height: 0,
            channels: 0,
        };

        assert_eq!(actual, expect);
    }
}
