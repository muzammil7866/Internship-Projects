import argparse
import os
from min_dalle import MinDalle
import torch

def generate_image_dalle(prompt, output_path="dalle_output.png", is_mega=False):
    """
    Generates an image using DALL-E Mini (or Mega) and saves it.
    
    Args:
        prompt (str): Text prompt for generation.
        output_path (str): File path to save the generated image.
        is_mega (bool): Whether to use the Mega version (better quality, more RAM).
    """
    print(f"Initializing DALL-E {'Mega' if is_mega else 'Mini'}...")
    # check for cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize model
    model = MinDalle(
        models_root='./pretrained',
        dtype=torch.float32, 
        device=device,
        is_mega=is_mega, 
        is_reusable=True
    )
    
    print(f"Generating image for prompt: '{prompt}'")
    # Generate image
    image = model.generate_image(
        text=prompt,
        seed=-1,
        grid_size=1
    )
    
    # Save image
    image.save(output_path)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images with DALL-E Mini/Mega")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the image")
    parser.add_argument("--output", type=str, default="dalle_result.png", help="Output filename")
    parser.add_argument("--mega", action="store_true", help="Use DALL-E Mega instead of Mini")
    
    args = parser.parse_args()
    
    generate_image_dalle(args.prompt, args.output, args.mega)
