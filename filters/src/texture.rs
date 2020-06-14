use image::GenericImageView;
use std::io;

pub struct Texture {
    pub descriptor: wgpu::TextureDescriptor<'static>,
    pub buffer: wgpu::Texture,
    pub view: wgpu::TextureView,
}

impl Texture {
    // pub fn from_bytes(
    //     device: &wgpu::Device,
    //     bytes: &[u8],
    //     label: &str,
    // ) -> Result<(Self, wgpu::CommandBuffer), io::Error> {
    //     let img = image::load_from_memory(bytes).map_err(|_| io::ErrorKind::InvalidData)?;
    //     Self::from_image(device, &img, label)
    // }

    pub fn new(device: &wgpu::Device, descriptor: wgpu::TextureDescriptor) -> Texture {
        let texture = device.create_texture(&descriptor);

        let descriptor = wgpu::TextureDescriptor::<'static> {
            label: None,
            size: descriptor.size,
            mip_level_count: descriptor.mip_level_count,
            sample_count: descriptor.sample_count,
            dimension: descriptor.dimension,
            format: descriptor.format,
            usage: descriptor.usage,
        };

        let view = texture.create_default_view();

        Texture {
            descriptor,
            buffer: texture,
            view,
        }
    }

    pub fn from_bytes(
        device: &wgpu::Device,
        data: &[u8],
        width: u32,
        height: u32,
        encoder: &mut wgpu::CommandEncoder,
        _label: &str,
    ) -> Result<Texture, image::ImageError> {
        // let rgba = img.as_rgba8().unwrap();
        // let dimensions = img.dimensions();

        // TODO: Take width arg as usize to remove excessive casting
        // WebGPU requires width % COPY_BYTES_PER_ROW_ALIGNMENT == 0
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;

        let bytes_per_pixel = std::mem::size_of::<u32>();
        let unpadded_width = width as usize * bytes_per_pixel;

        // Padding until the next multiple of COPY_BYTES_PER_ROW_ALIGNMENT
        let padding_remainder = (align - unpadded_width % align) % align;
        let padded_width = unpadded_width + padding_remainder;

        // Align data by adding remaining padding to each row
        let mut padded_data = vec![0; padded_width * height as usize];

        for row in 0..height as usize {
            padded_data[(row * padded_width)..(row * padded_width + width as usize)]
                .copy_from_slice(&data[(row * width as usize)..(row + 1) * width as usize]);
        }

        let texture_extent = wgpu::Extent3d {
            width,
            height,
            depth: 1,
        };

        let descriptor = wgpu::TextureDescriptor {
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // format: wgpu::TextureFormat::Rgba8Unorm,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsage::SAMPLED
                | wgpu::TextureUsage::STORAGE
                | wgpu::TextureUsage::COPY_DST
                | wgpu::TextureUsage::COPY_SRC,
            label: Some("texture"),
        };

        let texture = Self::new(device, descriptor);

        let buffer = device.create_buffer_with_data(&padded_data, wgpu::BufferUsage::COPY_SRC);

        // let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        //     label: Some("texture_buffer_copy_encoder"),
        // });

        println!("Padded width: {}", padded_width);

        encoder.copy_buffer_to_texture(
            wgpu::BufferCopyView {
                buffer: &buffer,
                layout: wgpu::TextureDataLayout {
                    offset: 0,
                    bytes_per_row: padded_width as u32,
                    rows_per_image: height as u32,
                },
            },
            wgpu::TextureCopyView {
                texture: &texture.buffer,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            texture_extent,
        );

        // let cmd_buffer = encoder.finish();

        // let view = texture.create_default_view();
        // let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        //     label: Some("texture_sampler"),
        //     address_mode_u: wgpu::AddressMode::ClampToEdge,
        //     address_mode_v: wgpu::AddressMode::ClampToEdge,
        //     address_mode_w: wgpu::AddressMode::ClampToEdge,
        //     mag_filter: wgpu::FilterMode::Linear,
        //     min_filter: wgpu::FilterMode::Nearest,
        //     mipmap_filter: wgpu::FilterMode::Nearest,
        //     ..Default::default()
        // });
        Ok(texture)

        // Ok((
        //     Self {
        //         texture,
        //         view,
        //         sampler,
        //     },
        //     cmd_buffer,
        // ))
    }
}

