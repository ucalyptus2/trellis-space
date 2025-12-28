#!/usr/bin/env python3
"""
run.py - Batch process images to 3D GLB models using TRELLIS.2

Usage:
    python run.py --input_dir /tmp/trellis-test-images/ --output_dir /tmp/trellis-outputs/
"""

# Set environment variables BEFORE importing trellis2
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ATTN_BACKEND"] = "flash_attn_3"
os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autotune_cache.json')
os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = '1'

import argparse
import glob
import sys
from pathlib import Path
from typing import Tuple, Optional
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Patch torch dynamo assertion error (compatibility workaround for torch 2.6.0)
try:
    import torch._dynamo.trace_rules
    # Skip the problematic assertion by monkey-patching
    original_init = torch._dynamo.trace_rules.__class__.__init__
except Exception:
    pass

from trellis2.modules.sparse import SparseTensor
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.pipelines.rembg.BiRefNet import BiRefNet
import o_voxel


MAX_SEED = np.iinfo(np.int32).max


def get_best_gpu():
    """
    Find the GPU with the most free memory.

    Returns:
        int: GPU device ID with most free memory
    """
    if not torch.cuda.is_available():
        return 0

    num_gpus = torch.cuda.device_count()
    max_free_memory = 0
    best_gpu = 0

    for gpu_id in range(num_gpus):
        torch.cuda.set_device(gpu_id)
        free_memory = torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_allocated(gpu_id)
        print(f"GPU {gpu_id}: {free_memory / 1024**3:.2f} GB free")
        if free_memory > max_free_memory:
            max_free_memory = free_memory
            best_gpu = gpu_id

    print(f"Selected GPU {best_gpu} with {max_free_memory / 1024**3:.2f} GB free")
    return best_gpu


def load_pipeline(gpu_id=None):
    """
    Load and initialize the Trellis2 pipeline and background removal model.

    Args:
        gpu_id: GPU device ID to use. If None, auto-selects GPU with most free memory.

    Returns:
        Tuple[Trellis2ImageTo3DPipeline, BiRefNet]: The pipeline and rembg model
    """
    # Auto-select best GPU if not specified
    if gpu_id is None:
        gpu_id = get_best_gpu()

    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)
    print(f"Loading TRELLIS.2 pipeline on GPU {gpu_id}...")

    # Load the pipeline, but catch the rembg error
    try:
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
    except RuntimeError as e:
        if "meta tensors" in str(e):
            # If it's the meta tensors error, we can ignore it for now
            # We'll load BiRefNet separately
            print(f"Note: Background removal model failed to load automatically ({e})")
            print("Will load BiRefNet separately...")
            # Try loading without the rembg model
            from trellis2.pipelines.base import Pipeline
            pipeline_base = Pipeline.from_pretrained('microsoft/TRELLIS.2-4B')
            pipeline = Trellis2ImageTo3DPipeline()
            pipeline.__dict__ = pipeline_base.__dict__
            args = pipeline_base._pretrained_args

            from trellis2.pipelines import samplers
            from trellis2.modules import image_feature_extractor

            pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
            pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

            pipeline.shape_slat_sampler = getattr(samplers, args['shape_slat_sampler']['name'])(**args['shape_slat_sampler']['args'])
            pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

            pipeline.tex_slat_sampler = getattr(samplers, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
            pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

            pipeline.shape_slat_normalization = args['shape_slat_normalization']
            pipeline.tex_slat_normalization = args['tex_slat_normalization']

            pipeline.image_cond_model = getattr(image_feature_extractor, args['image_cond_model']['name'])(**args['image_cond_model']['args'])

            # Skip loading rembg model from pretrained (we'll load it separately)
            pipeline.rembg_model = None

            pipeline.low_vram = args.get('low_vram', True)
            pipeline.default_pipeline_type = args.get('default_pipeline_type', '1024_cascade')
            pipeline.pbr_attr_layout = {
                'base_color': slice(0, 3),
                'metallic': slice(3, 4),
                'roughness': slice(4, 5),
                'alpha': slice(5, 6),
            }
        else:
            raise

    print("Loading BiRefNet for background removal...")
    try:
        rembg_model = BiRefNet(model_name="ZhengPeng7/BiRefNet")
        rembg_model.to(device)
        # Set pipeline to use our rembg model
        pipeline.rembg_model = rembg_model
    except Exception as e:
        print(f"WARNING: Failed to load BiRefNet: {e}")
        print("Proceeding without background removal model...")
        rembg_model = None
        pipeline.rembg_model = None

    # Disable low_vram mode - we want to use GPU memory, not offload to CPU
    pipeline.low_vram = False

    # Move pipeline to the selected GPU
    pipeline.to(device)

    print(f"Pipeline loaded successfully on GPU {gpu_id}!")
    return pipeline, rembg_model


def preprocess_image(image: Image.Image, rembg_model: BiRefNet) -> Image.Image:
    """
    Preprocess the input image: resize, remove background if needed, center and crop.

    Args:
        image: Input PIL Image
        rembg_model: BiRefNet model for background removal

    Returns:
        Preprocessed PIL Image
    """
    # Check if image has alpha channel
    has_alpha = False
    if image.mode == 'RGBA':
        alpha = np.array(image)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True

    # Resize if too large
    max_size = max(image.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Remove background if no alpha channel
    if has_alpha:
        output = image
    else:
        if rembg_model is not None:
            # Convert to RGB for background removal
            image_rgb = image.convert('RGB')
            output = rembg_model(image_rgb)
        else:
            # No background removal model, add opaque alpha channel
            image_rgb = image.convert('RGB')
            output = Image.new('RGBA', image_rgb.size)
            output.paste(image_rgb, (0, 0))
            # Set alpha to fully opaque
            alpha_channel = Image.new('L', image_rgb.size, 255)
            output.putalpha(alpha_channel)

    # Center and crop the foreground object
    output_np = np.array(output)
    alpha = output_np[:, :, 3]

    # Find bounding box of foreground
    bbox = np.argwhere(alpha > 0.8 * 255)
    if len(bbox) == 0:
        print("Warning: No foreground detected, using full image")
        return output

    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.0)  # No padding

    # Create square crop around center
    bbox = (
        center[0] - size // 2,
        center[1] - size // 2,
        center[0] + size // 2,
        center[1] + size // 2
    )
    output = output.crop(bbox)

    # Convert to RGB with black background
    output_np = np.array(output).astype(np.float32) / 255
    output_np = output_np[:, :, :3] * output_np[:, :, 3:4]
    output = Image.fromarray((output_np * 255).astype(np.uint8))

    return output


def process_single_image(
    image_path: str,
    output_path: str,
    pipeline: Trellis2ImageTo3DPipeline,
    rembg_model: BiRefNet,
    seed: int = 42,
    resolution: str = "1024",
    decimation_target: int = 300000,
    texture_size: int = 2048,
    ss_sampling_steps: int = 12,
    ss_guidance_strength: float = 7.5,
    ss_guidance_rescale: float = 0.7,
    ss_rescale_t: float = 5.0,
    shape_slat_sampling_steps: int = 12,
    shape_slat_guidance_strength: float = 7.5,
    shape_slat_guidance_rescale: float = 0.5,
    shape_slat_rescale_t: float = 3.0,
    tex_slat_sampling_steps: int = 12,
    tex_slat_guidance_strength: float = 1.0,
    tex_slat_guidance_rescale: float = 0.0,
    tex_slat_rescale_t: float = 3.0,
) -> bool:
    """
    Process a single image to GLB 3D model.

    Args:
        image_path: Path to input image
        output_path: Path to save GLB file
        pipeline: Trellis2 pipeline
        rembg_model: Background removal model
        seed: Random seed
        resolution: "512", "1024", or "1536"
        decimation_target: Target face count for mesh decimation
        texture_size: Texture resolution
        Other args: Sampling parameters for the three generation stages

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load image
        print(f"Loading image: {image_path}")
        image = Image.open(image_path)

        # Preprocess
        print("Preprocessing image...")
        image = preprocess_image(image, rembg_model)

        # Generate 3D model
        print("Generating 3D model (Stage 1: Sparse Structure)...")
        pipeline_type = {
            "512": "512",
            "1024": "1024_cascade",
            "1536": "1536_cascade",
        }[resolution]

        outputs, latents = pipeline.run(
            image,
            seed=seed,
            preprocess_image=False,  # Already preprocessed
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "guidance_strength": ss_guidance_strength,
                "guidance_rescale": ss_guidance_rescale,
                "rescale_t": ss_rescale_t,
            },
            shape_slat_sampler_params={
                "steps": shape_slat_sampling_steps,
                "guidance_strength": shape_slat_guidance_strength,
                "guidance_rescale": shape_slat_guidance_rescale,
                "rescale_t": shape_slat_rescale_t,
            },
            tex_slat_sampler_params={
                "steps": tex_slat_sampling_steps,
                "guidance_strength": tex_slat_guidance_strength,
                "guidance_rescale": tex_slat_guidance_rescale,
                "rescale_t": tex_slat_rescale_t,
            },
            pipeline_type=pipeline_type,
            return_latent=True,
        )

        # Get mesh and simplify (nvdiffrast has 16M face limit)
        mesh = outputs[0]
        mesh.simplify(16777216)

        # Extract latents
        shape_slat, tex_slat, res = latents

        # Decode latents to full mesh
        print("Decoding latents to full mesh...")
        mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
        mesh.simplify(16777216)

        # Export to GLB
        print("Exporting to GLB...")
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

        # Save GLB file
        glb.export(output_path, extension_webp=True)
        print(f"✓ Successfully saved: {output_path}")

        # Clear CUDA cache
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"✗ Error processing {image_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def process_directory(
    input_dir: str,
    output_dir: str,
    pipeline: Trellis2ImageTo3DPipeline,
    rembg_model: BiRefNet,
    max_images: Optional[int] = None,
    **kwargs
) -> Tuple[int, int]:
    """
    Process all images in a directory.

    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save GLB files
        pipeline: Trellis2 pipeline
        rembg_model: Background removal model
        max_images: Maximum number of images to process (None for all)
        **kwargs: Additional arguments passed to process_single_image

    Returns:
        Tuple[int, int]: (successful_count, failed_count)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp', '*.PNG', '*.JPG', '*.JPEG', '*.WEBP']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))

    if not image_files:
        print(f"No images found in {input_dir}")
        return 0, 0

    # Limit number of images if specified
    if max_images is not None:
        image_files = image_files[:max_images]

    print(f"\nFound {len(image_files)} image(s) to process")
    print(f"Output directory: {output_dir}\n")

    successful = 0
    failed = 0

    for image_path in tqdm(image_files, desc="Processing images"):
        # Generate output filename
        input_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{input_name}.glb")

        # Skip if already exists
        if os.path.exists(output_path):
            print(f"Skipping {input_name} (already exists)")
            continue

        # Process image
        success = process_single_image(
            image_path=image_path,
            output_path=output_path,
            pipeline=pipeline,
            rembg_model=rembg_model,
            **kwargs
        )

        if success:
            successful += 1
        else:
            failed += 1

        print()  # Empty line for readability

    return successful, failed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process images to 3D GLB models using TRELLIS.2"
    )

    # Input/output
    parser.add_argument(
        '--input_dir',
        type=str,
        default='/tmp/trellis-test-images/',
        help='Input directory containing images'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/tmp/trellis-outputs/',
        help='Output directory for GLB files'
    )

    # Processing parameters
    parser.add_argument(
        '--resolution',
        type=str,
        choices=['512', '1024', '1536'],
        default='1024',
        help='Generation resolution (higher = better quality but slower)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--decimation_target',
        type=int,
        default=300000,
        help='Target face count for mesh decimation'
    )
    parser.add_argument(
        '--texture_size',
        type=int,
        default=2048,
        help='Texture resolution (1024, 2048, or 4096)'
    )
    parser.add_argument(
        '--max_images',
        type=int,
        default=None,
        help='Maximum number of images to process (for testing)'
    )

    # Advanced sampling parameters
    parser.add_argument('--ss_sampling_steps', type=int, default=12)
    parser.add_argument('--ss_guidance_strength', type=float, default=7.5)
    parser.add_argument('--ss_guidance_rescale', type=float, default=0.7)
    parser.add_argument('--ss_rescale_t', type=float, default=5.0)

    parser.add_argument('--shape_slat_sampling_steps', type=int, default=12)
    parser.add_argument('--shape_slat_guidance_strength', type=float, default=7.5)
    parser.add_argument('--shape_slat_guidance_rescale', type=float, default=0.5)
    parser.add_argument('--shape_slat_rescale_t', type=float, default=3.0)

    parser.add_argument('--tex_slat_sampling_steps', type=int, default=12)
    parser.add_argument('--tex_slat_guidance_strength', type=float, default=1.0)
    parser.add_argument('--tex_slat_guidance_rescale', type=float, default=0.0)
    parser.add_argument('--tex_slat_rescale_t', type=float, default=3.0)

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 70)
    print("TRELLIS.2 Image to 3D GLB Converter")
    print("=" * 70)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Resolution: {args.resolution}")
    print(f"Seed: {args.seed}")
    print(f"Decimation target: {args.decimation_target}")
    print(f"Texture size: {args.texture_size}")
    print("=" * 70)
    print()

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires a GPU.")
        sys.exit(1)

    print(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()

    # Load pipeline
    pipeline, rembg_model = load_pipeline()
    print()

    # Process images
    successful, failed = process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pipeline=pipeline,
        rembg_model=rembg_model,
        max_images=args.max_images,
        seed=args.seed,
        resolution=args.resolution,
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        ss_sampling_steps=args.ss_sampling_steps,
        ss_guidance_strength=args.ss_guidance_strength,
        ss_guidance_rescale=args.ss_guidance_rescale,
        ss_rescale_t=args.ss_rescale_t,
        shape_slat_sampling_steps=args.shape_slat_sampling_steps,
        shape_slat_guidance_strength=args.shape_slat_guidance_strength,
        shape_slat_guidance_rescale=args.shape_slat_guidance_rescale,
        shape_slat_rescale_t=args.shape_slat_rescale_t,
        tex_slat_sampling_steps=args.tex_slat_sampling_steps,
        tex_slat_guidance_strength=args.tex_slat_guidance_strength,
        tex_slat_guidance_rescale=args.tex_slat_guidance_rescale,
        tex_slat_rescale_t=args.tex_slat_rescale_t,
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {successful + failed}")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
