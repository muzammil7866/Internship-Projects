# GPT-2 Fine-Tuning for Custom Text Generation (Biryani Dataset)

This project was developed as part of my internship in **Generative AI**, focusing on the fine-tuning of OpenAI's GPT-2 model using a custom dataset.  
For demonstration purposes, I used a dataset of descriptive sentences about *Biryani* to train the model, enabling it to generate coherent and contextually relevant text that mimics the style and tone of the dataset.

## Project Overview

Generative AI models, such as GPT-2, can be adapted for specialized domains by fine-tuning them on domain-specific data.  
In this case:

- **Base Model:** GPT-2
- **Custom Dataset:** A collection of sentences about *Biryani*
- **Goal:** Generate creative, human-like text around the topic of *Biryani* for experimentation and skill polishing

The repository contains:
- A script to use GPT-2 without fine-tuning
- Fine-tuning pipeline using Hugging Face Transformers
- Trained model saving and loading
- Example text generation

## Business Goals & Applications

Fine-tuning GPT-2 like this can support multiple business objectives:

1. **Content Generation Automation**
   - Automatically create themed blog posts, menu descriptions, or marketing copy.
   
2. **Brand Voice Adaptation**
   - Ensure generated content matches a company’s tone and vocabulary.
   
3. **Conversational Agents**
   - Build chatbots that can speak knowledgeably in a specific domain (e.g., food, travel, culture).
   
4. **Customer Engagement**
   - Generate engaging product descriptions to attract customers.
   
5. **Prototyping for NLP Solutions**
   - Quickly test generative models for niche markets before large-scale deployment.


## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Generative-AI-Internship-Task-GPT-2-Model-Finetuning-on-Biryani-master.git
cd Generative-AI-Internship-Task-GPT-2-Model-Finetuning-on-Biryani-master
```

### 2. Install Dependencies
Make sure you have Python 3.8+ installed, then run:

```bash

pip install -r requirements.txt
```

### requirements.txt

```txt
transformers
datasets
torch
```
### 3. Prepare the Dataset
Create a text file named custom_dataset.txt containing one sentence per line.

Place it in the project root directory.

Example:

```txt
The biryani of Lahore is rich and flavorful.
Lahore’s biryani is a blend of spices and tradition.
```

### 4. Run Without Fine-Tuning (Optional)
```bash
python gpt2_basic.py
```
### 5. Fine-Tune GPT-2 
```bash
python gpt2_finetune.py
```
## 6. Running for Testing 
After fine-tuning, you can generate text with:

```bash
python generate.py
```

Example:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="./gpt2-finetuned")
prompt = "Biryani is very"
output = generator(prompt, max_length=50, num_return_sequences=1)
print(output[0]['generated_text'])
```

## Outputs

### Before
```txt
Response without Finetuning:

Biryani is very proud of the work he is doing in that area.
"We're doing a lot of work in this area and we're looking to see how we can make sure that we make sure that all of our members are happy in their communities."
This is an edited extract from the Radio-Television Correspondent's Association (RTA) podcast, which is available on iTunes, Stitcher and other devices.
Topics: aboriginal-aboriginal-and-torres-strait-islander, community-and-society, community-and-society, sydney-2000, sydney-2000
```
### After
```txt
Response after Model Finetuning:

Biryani is very rich in flavour.
```

## Repository Structure
```bash
├── custom_dataset.txt      # Training data
├── gpt2_basic.py           # GPT-2 text generation without fine-tuning
├── gpt2_finetune.py        # Fine-tuning script
├── generate.py             # Generate text using the fine-tuned model
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Internship Learning Outcomes
Learned the Hugging Face Transformers workflow for GPT-2

Understood tokenization, padding, and truncation in NLP

Implemented custom dataset fine-tuning for a specific domain

Applied hyperparameter tuning for better text generation

Explored practical business applications of domain-specific language models

## License
This project is open-source and available under the MIT License.
