import random
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
import torch
import gradio as gr

base_model = "./finneone"
train_epochs = 1

dataset = load_dataset("genaforvena/huivam_finnegans_wake_paragraphs")

text = ""
for example in dataset["train"]:
    text += example["context"] + " " + example["instruction"] + " " + example["response"]

def clean_text(text):
    text = text.lower()
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace("\r", " ")
    return text

text = clean_text(text)

def get_random_single_words(text, sample_size=1000):
    words = text.split()
    random.shuffle(words)
    selected_words = words[:sample_size]
    print(f"Single-word samples: {selected_words[:20]}")
    return selected_words

def tokenize_function(tokenizer, examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=16
    )

def prepare_dataset_single_words(text, sample_size=1000):
    selected_words = get_random_single_words(text, sample_size=sample_size)
    ds = Dataset.from_dict({"text": selected_words})
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenised_dataset = ds.map(
        lambda x: tokenize_function(tokenizer, x),
        batched=True,
        remove_columns=["text"]
    )
    return tokenised_dataset, tokenizer

def finetuned_model(tokenised_dataset, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=train_epochs,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
        report_to=None
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised_dataset,
        data_collator=data_collator
    )

    trainer.train()

    model.save_pretrained("./results")
    tokenizer.save_pretrained("./results")
    return "./results"

def generate_text(device, model, tokenizer, prompt, max_length=15, temperature=0.8, top_k=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def launch_ui(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def generate(prompt, max_length=15, temperature=0.8, top_k=50):
        return generate_text(device, model, tokenizer, prompt, max_length, temperature, top_k)
    
    iface = gr.Interface(fn=generate, inputs="text", outputs="text")
    iface.launch()

def main():
    tokenised_dataset, tokenizer = prepare_dataset_single_words(text, sample_size=3000)
    tuned_path = finetuned_model(tokenised_dataset, tokenizer)
    launch_ui(tuned_path)

if __name__ == "__main__":
    main()
