from transformers import pipeline

generator = pipeline("text-generation", model="./gpt2-finetuned")
prompt = "Biryani is very"
output = generator(prompt, max_length=50, num_return_sequences=1)
print("Response after Model Finetuning: \n\n")
print(output[0]['generated_text'])
