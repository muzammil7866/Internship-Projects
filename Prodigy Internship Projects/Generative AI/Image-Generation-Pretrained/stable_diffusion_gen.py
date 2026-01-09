import argparse
import torch
from diffusers import StableDiffusionPipeline

def generate_image_sd(prompt, output_path="sd_output.png"):
    """
    Generates an image using Stable Diffusion and saves it.
    
    Args:
        prompt (str): Text prompt for generation.
        output_path (str): File path to save the generated image.
    """
    print("Initializing Stable Diffusion...")
    
    model_id = "runwayml/stable-diffusion-v1-5"
    
    if torch.cuda.is_available():
        print("Using CUDA (GPU).")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
    else:
        print("Using CPU. Warning: This will be slow.")
        pipe = StableDiffusionPipeline.from_pretrained(model_id)
        pipe = pipe.to("cpu")

    print(f"Generating image for prompt: '{prompt}'")
    image = pipe(prompt).images[0]
    
    image.save(output_path)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the image")
    parser.add_argument("--output", type=str, default="sd_result.png", help="Output filename")
    
    args = parser.parse_args()
    
    generate_image_sd(args.prompt, args.output)
