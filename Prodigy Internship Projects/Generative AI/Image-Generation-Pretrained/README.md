# Image Generation with Pretrained Models

This project demonstrates the capability of Generative AI by utilizing two powerful pre-trained models, **DALL-E Mini** and **Stable Diffusion**, to generate images from text prompts.

## Project Overview

The goal of this project is to provide a simple, programmatic interface to access state-of-the-art image generation models. This allows users to experiment with AI-driven art creation and understand the differences between various architectures (Transformer-based vs. Diffusion-based).

### Technologies Used
- **Python**: Core programming language.
- **DALL-E Mini**: A lighter version of OpenAI's DALL-E, good for quick, stylized generations.
- **Stable Diffusion**: A latent diffusion model capable of generating photo-realistic images.
- **PyTorch & Diffusers**: Libraries for deep learning model management.

## Business Goals Achieved

Implementing this project addresses several key business objectives for companies looking to leverage AI:

1.  **Rapid Prototyping & Mood Boarding**:
    -   *Goal*: Reduce time spent on conceptualizing visual ideas.
    -   *Achievement*: Designers can instantly generate visual representations of abstract concepts, speeding up the creative brainstorming phase.

2.  **Cost-Effective Asset Creation**:
    -   *Goal*: Minimize reliance on expensive stock photography or outsourced illustration.
    -   *Achievement*: Generates unique, royalty-free assets for internal presentations, social media posts, or mockups at zero marginal cost.

3.  **Scalable Content Production**:
    -   *Goal*: Automate the creation of visual content for marketing campaigns.
    -   *Achievement*: Scripts can be integrated into pipelines to bulk-generate thousands of variations of an image based on dynamic prompts (e.g., personalized ads).

4.  **On-Demand Customization**:
    -   *Goal*: Provide hyper-specific imagery that stock libraries cannot match.
    -   *Achievement*: Creates images tailored exactly to specific descriptions ("A cyberpunk cat eating pizza in styles of Van Gogh"), fulfilling niche requirements instantly.

## Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA recommended (Running on CPU is possible but slow).

### Setup
1. Clone the repository or navigate to the project directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. DALL-E Mini Generation
Uses `min-dalle` to generate images.

**Command:**
```bash
python dalle_mini_gen.py --prompt "A futuristic city in the clouds" --output my_city.png
```

**Options:**
- `--prompt`: The text description of the image.
- `--output`: File path to save the result.
- `--mega`: Use the larger "Mega" model (requires more RAM).

### 2. Stable Diffusion Generation
Uses huggingface `diffusers` to generate high-quality images.

**Command:**
```bash
python stable_diffusion_gen.py --prompt "A realistic portrait of an astronaut" --output astronaut.png
```

**Options:**
- `--prompt`: The text description of the image.
- `--output`: File path to save the result.

## Notes
- **First Run**: When running scripts for the first time, they will download several gigabytes of model weights from HuggingFace.
- **Hardware**: If you do not have a GPU, the scripts will auto-detect this and run on CPU, but generation might take several minutes per image.
