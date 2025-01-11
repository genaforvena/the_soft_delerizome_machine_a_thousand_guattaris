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
deleuze, tokenizer1 = load_model('genaforvena/the_soft_delerizome_machine_a_thousand_guattaris_fourth_of_plateaus_once')
gospel, tokenizer2 = load_model('genaforvena/the_soft_scum_gospel_delerizome_machine_a_thousand_guattaris')
scum, tokenizer3 = load_model('genaforvena/the_soft_scum_delerizome_machine_a_thousand_guattaris')

# Define the prompt
prompt = "ghost"


# Function to generate text
def generate_text(model, tokenizer, prompt, max_new_tokens, temperature):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.95, top_k=50)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text


# Generate outputs
output1 = generate_text(deleuze, tokenizer1, prompt, 
                        max_new_tokens=700, temperature=0.6)
output2 = generate_text(scum, tokenizer2, output1, 
                        max_new_tokens=300, temperature=1.7)
output3 = generate_text(gospel, tokenizer3, prompt, 
                        max_new_tokens=350, temperature=1.0)

print("Output 1:\n", output1)
print("\nOutput 2:\n", output2)
print("\nOutput 3:\n", output3)

# Generate the final ensembled output
final_output = output2 + '\n' + output3
print("\n\n\n------------\nmodels sum output: " + final_output)

summarizer = pipeline("summarization", model="Falconsai/text_summarization")

summary = summarizer(final_output, max_length=600, min_length=250, do_sample=False)

print("Summary:\n", summary[0]['summary_text'])


