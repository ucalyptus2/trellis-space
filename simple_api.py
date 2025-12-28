#!/usr/bin/env python3
"""
Simple API for TRELLIS image-to-3D generation
Processes images and outputs .glb files
"""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ATTN_BACKEND"] = "flash_attn_3"
os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autotune_cache.json')

import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
import traceback

# Monkey patch to fix the meta tensor issue in rembg models
original_linspace = torch.linspace
def patched_linspace(*args, **kwargs):
    """Patched linspace that avoids meta tensors"""
    result = original_linspace(*args, **kwargs)
    # If result is on meta device, move to CPU
    if hasattr(result, 'device') and result.device.type == 'meta':
        # Create on CPU instead
        return torch.linspace(*args, **{k: v for k, v in kwargs.items() if k != 'device'}).cpu()
    return result

torch.linspace = patched_linspace

from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel


def preprocess_image(input_image: Image.Image) -> Image.Image:
    """
    Preprocess the input image - resize and handle alpha channel
    """
    # Check if has alpha channel
    has_alpha = False
    if input_image.mode == 'RGBA':
        alpha = np.array(input_image)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True

    # Resize if too large
    max_size = max(input_image.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        input_image = input_image.resize(
            (int(input_image.width * scale), int(input_image.height * scale)),
            Image.Resampling.LANCZOS
        )

    # If no alpha channel, just convert to RGB
    if not has_alpha:
        print("WARNING: Image has no alpha channel. For best results, use images with transparent backgrounds.")
        # Create a white background version
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')
        # Add a full alpha channel
        output = input_image.copy()
        output.putalpha(255)
    else:
        output = input_image

    # Crop to bounding box of non-transparent pixels
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    if len(bbox) == 0:
        print("ERROR: Image is completely transparent!")
        return output

    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox)

    # Premultiply alpha
    output = np.array(output).astype(np.float32) / 255
    output = output[:, :, :3] * output[:, :, 3:4]
    output = Image.fromarray((output * 255).astype(np.uint8))

    return output


def generate_3d_from_image(
    pipeline,
    image_path: str,
    output_dir: str,
    seed: int = 42,
    resolution: str = "1024",
    decimation_target: int = 300000,
    texture_size: int = 2048,
):
    """
    Generate a 3D GLB file from an input image

    Args:
        pipeline: The TRELLIS pipeline
        image_path: Path to input image
        output_dir: Directory to save output GLB file
        seed: Random seed
        resolution: Resolution ('512', '1024', or '1536')
        decimation_target: Target face count for mesh decimation
        texture_size: Texture resolution

    Returns:
        Path to generated GLB file
    """
    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print(f"{'='*60}")

    # Load and preprocess image
    print("Loading image...")
    image = Image.open(image_path)
    print(f"Original size: {image.size}, mode: {image.mode}")

    print("Preprocessing image...")
    image = preprocess_image(image)
    print(f"Preprocessed size: {image.size}")

    # Run pipeline
    print(f"Generating 3D model (resolution: {resolution}, seed: {seed})...")
    torch.manual_seed(seed)

    pipeline_type = {
        "512": "512",
        "1024": "1024_cascade",
        "1536": "1536_cascade",
    }[resolution]

    outputs, latents = pipeline.run(
        image,
        seed=seed,
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": 12,
            "guidance_strength": 7.5,
            "guidance_rescale": 0.7,
            "rescale_t": 5.0,
        },
        shape_slat_sampler_params={
            "steps": 12,
            "guidance_strength": 7.5,
            "guidance_rescale": 0.5,
            "rescale_t": 3.0,
        },
        tex_slat_sampler_params={
            "steps": 12,
            "guidance_strength": 1.0,
            "guidance_rescale": 0.0,
            "rescale_t": 3.0,
        },
        pipeline_type=pipeline_type,
        return_latent=True,
    )

    mesh = outputs[0]
    print(f"Generated mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")

    # Simplify mesh
    print("Simplifying mesh...")
    mesh.simplify(16777216)  # nvdiffrast limit

    # Extract GLB
    print(f"Extracting GLB (decimation: {decimation_target}, texture: {texture_size})...")
    shape_slat, tex_slat, res = latents
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )

    # Save GLB
    os.makedirs(output_dir, exist_ok=True)
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    input_name = Path(image_path).stem
    glb_filename = f'{input_name}_{timestamp}.glb'
    glb_path = os.path.join(output_dir, glb_filename)

    print(f"Saving GLB to: {glb_path}")
    glb.export(glb_path, extension_webp=True)

    # Clean up
    torch.cuda.empty_cache()

    print(f"✓ Successfully generated: {glb_path}")
    return glb_path


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='TRELLIS Image to 3D API')
    parser.add_argument('--input', '-i', required=True, help='Input image file or directory')
    parser.add_argument('--output', '-o', default='./output', help='Output directory for GLB files')
    parser.add_argument('--resolution', '-r', choices=['512', '1024', '1536'], default='1024', help='Generation resolution')
    parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed')
    parser.add_argument('--decimation', '-d', type=int, default=300000, help='Decimation target (face count)')
    parser.add_argument('--texture-size', '-t', type=int, default=2048, help='Texture size')
    parser.add_argument('--model', '-m', default='microsoft/TRELLIS.2-4B', help='Model name or path')

    args = parser.parse_args()

    # Load pipeline
    print("Loading TRELLIS pipeline...")
    print(f"Model: {args.model}")

    try:
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        print("This might be due to the rembg model issue. Trying workaround...")

        # Try to load without rembg
        try:
            # Temporarily disable rembg loading
            import trellis2.pipelines.rembg as rembg_module
            original_birefnet = None
            if hasattr(rembg_module, 'BiRefNet'):
                original_birefnet = rembg_module.BiRefNet
                # Create a dummy class that returns None
                class DummyBiRefNet:
                    def __init__(self, *args, **kwargs):
                        pass
                    def __call__(self, *args, **kwargs):
                        return None
                rembg_module.BiRefNet = DummyBiRefNet

            pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)

            # Restore original
            if original_birefnet:
                rembg_module.BiRefNet = original_birefnet
        except Exception as e2:
            print(f"Failed to load pipeline even with workaround: {e2}")
            traceback.print_exc()
            sys.exit(1)

    # Disable rembg since we're preprocessing ourselves
    pipeline.rembg_model = None
    pipeline.low_vram = False

    # Move pipeline to GPU
    pipeline.cuda()
    pipeline._device = 'cuda'

    print("✓ Pipeline loaded successfully")
    print(f"Device: {pipeline._device}")

    # Verify models are on GPU
    for name, model in pipeline.models.items():
        if hasattr(model, 'device'):
            print(f"  {name}: {model.device}")
        elif hasattr(model, 'parameters'):
            print(f"  {name}: {next(model.parameters()).device}")

    # Process input
    input_path = Path(args.input)

    if input_path.is_file():
        # Single file
        image_files = [input_path]
    elif input_path.is_dir():
        # Directory - find all images
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            image_files.extend(input_path.glob(ext))
        image_files = sorted(image_files)
    else:
        print(f"ERROR: Input path does not exist: {args.input}")
        sys.exit(1)

    if not image_files:
        print(f"ERROR: No image files found in: {args.input}")
        sys.exit(1)

    print(f"\nFound {len(image_files)} image(s) to process")

    # Process each image
    results = []
    errors = []

    for i, image_file in enumerate(image_files, 1):
        try:
            print(f"\n[{i}/{len(image_files)}] Processing {image_file.name}...")
            glb_path = generate_3d_from_image(
                pipeline,
                str(image_file),
                args.output,
                seed=args.seed,
                resolution=args.resolution,
                decimation_target=args.decimation,
                texture_size=args.texture_size,
            )
            results.append((image_file, glb_path))
        except Exception as e:
            print(f"✗ ERROR processing {image_file.name}: {e}")
            traceback.print_exc()
            errors.append((image_file, str(e)))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully processed: {len(results)}/{len(image_files)}")
    print(f"Errors: {len(errors)}")

    if results:
        print(f"\nGenerated GLB files in: {args.output}")
        for img, glb in results:
            print(f"  ✓ {img.name} -> {Path(glb).name}")

    if errors:
        print("\nErrors:")
        for img, err in errors:
            print(f"  ✗ {img.name}: {err}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
