from transformers import GPT2LMHeadModel, AutoTokenizer, pipeline
import torch


# Function to load model and tokenizer
def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token
    return model, tokenizer


# Load your three models
model1, tokenizer1 = load_model('genaforvena/the_soft_scum_gospel_delerizome_machine_a_thousand_guattaris')
model2, tokenizer2 = load_model('genaforvena/the_soft_scum_delerizome_machine_a_thousand_guattaris')
model3, tokenizer3 = load_model('genaforvena/the_soft_delerizome_machine_a_thousand_guattaris_fourth_of_plateaus_once')

# Define the prompt
prompt = "ghost"


# Function to generate text
def generate_text(model, tokenizer, prompt, max_length=250):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(inputs, max_length=max_length, do_sample=True, top_p=0.95, top_k=50)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text


# Generate outputs
output1 = generate_text(model1, tokenizer1, prompt)
output2 = generate_text(model2, tokenizer2, prompt)
output3 = generate_text(model3, tokenizer3, prompt)

print("Output 1:\n", output1)
print("\nOutput 2:\n", output2)
print("\nOutput 3:\n", output3)

# Generate the final ensembled output
final_output = output1 + '\n' + output2 + '\n' + output3

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

summary = summarizer(final_output, max_length=530, min_length=150, do_sample=False)

print("Summary:\n", summary[0]['summary_text'])


