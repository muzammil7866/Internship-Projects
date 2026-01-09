# =========================================
# GPT-2 Text Generation (No Fine-Tuning)
# =========================================

# Install dependencies if needed:
# pip install transformers torch

from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

# 1. Load pre-trained GPT-2 and tokenizer
model_name = "gpt2"  # can also use "gpt2-medium", "gpt2-large", etc.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 2. Create a text-generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 3. Provide a prompt
prompt = "Biryani"

# 4. Generate text
output = generator(
    prompt,
    max_length=20,           # maximum total tokens (prompt + generated)
    num_return_sequences=1,  # number of completions to generate
    temperature=0.7,         # creativity level (lower = more predictable, higher = more random)
    top_p=0.9,                # nucleus sampling
    do_sample=True            # enable sampling (not greedy)
)

# 5. Print the generated text
print("Response without Finetuning: \n\n")
print(output[0]['generated_text'])
