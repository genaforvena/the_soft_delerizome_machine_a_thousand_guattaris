from transformers import GPT2Tokenizer, GPT2LMHeadModel
import gradio as gr

#  @title Load Fine-tuned Model

# %%
model = GPT2LMHeadModel.from_pretrained("soft_delerizome_machine_epochs1_fourth_of_plateaus")
tokenizer = GPT2Tokenizer.from_pretrained("soft_delerizome_machine_epochs1_fourth_of_plateaus")
tokenizer.pad_token = tokenizer.eos_token

# @title Define Text Generation Function

# %%
def generate_text(prompt):
  """Generates text using the fine-tuned model."""
  inputs = tokenizer(prompt, return_tensors="pt", padding=True)
  outputs = model.generate(
      inputs["input_ids"],
      attention_mask=inputs["attention_mask"],
      max_length=150,
      num_return_sequences=1,
      do_sample=True,
      temperature=0.8,
      top_k=50,
      pad_token_id=tokenizer.eos_token_id
  )
  return tokenizer.decode(outputs[0], skip_special_tokens=True)

# @title Generate Text

# %%
prompt = "it is not a projection;"
print(generate_text(prompt))

#  @title Launch Gradio Interface

# %%
iface = gr.Interface(fn=generate_text, inputs="text", outputs="text")
iface.launch()
