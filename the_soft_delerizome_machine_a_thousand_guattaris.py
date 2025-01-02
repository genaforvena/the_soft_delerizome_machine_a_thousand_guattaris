from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import requests
from bs4 import BeautifulSoup
import re
import torch
import gradio as gr


def clean_text(text):
    soup = BeautifulSoup(text, "html.parser")  # Parse HTML
    text = soup.get_text()  # Extract text content
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-z\s\.,!?;:'\"-]+", " ", text)  # Keep only English letters, spaces, and punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text


def scrape_text_from_url(url):
    """Scrapes text from a given URL."""
    try:
        response = requests.get(url)
        if url.endswith(".txt"):  # Plain text files
            return clean_text(response.text)
        else:  # HTML files
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            text = " ".join(p.get_text() for p in paragraphs)
            return clean_text(text)
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""


def split_text_into_chunks(text, chunk_size=512):
    """Split text into chunks of approximately `chunk_size` words."""
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    print(f"Created {len(chunks)} chunks.")
    return chunks


def tokenize_function(tokenizer, examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,  # Adjust as needed
        # Removed return_tensors="pt"
    )


def scrape_training_texts():
    # Scrape and prepare text data
    text_urls = [
        "https://raw.githubusercontent.com/genaforvena/skiffs/main/resources/worstward_hoe.txt",  # Raw URL
        "https://raw.githubusercontent.com/dscape/spell/master/test/resources/big.txt",
        "https://raw.githubusercontent.com/dscape/spell/master/test/resources/alice.txt",
        # Add more raw URLs here for a more diverse dataset
    ]
    training_text = "training_text.txt"

    all_texts = []
    for url in text_urls:
        print(f"Scraping {url}...")
        text = scrape_text_from_url(url)
        if text:
            print(f"Length of scraped text: {len(text)} characters")
            all_texts.append(text)

    with open(training_text, "w", encoding="utf-8") as outfile:
        for text in all_texts:
            outfile.write(text)
            outfile.write("\n\n")

    print(f"All texts have been combined into {training_text}")
    return training_text  # Ensure this function returns the path


def prepare_dataset(training_text):
    # Load and chunk text data
    with open(training_text, "r", encoding="utf-8") as infile:
        full_text = infile.read()

    chunks = split_text_into_chunks(full_text, chunk_size=256)  # Adjust chunk_size as needed
    print(f"Total number of chunks created: {len(chunks)}")

    # Create a Dataset from the chunks
    dataset = Dataset.from_dict({"text": chunks})

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token

    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(tokenizer, x),
        batched=True,
        remove_columns=["text"]
    )

    # Inspect some tokenized examples
    print("Sample tokenized inputs:")
    print(tokenized_datasets[0])
    print(tokenized_datasets[1])

    return tokenized_datasets, tokenizer


def finetuned_model(tokenized_datasets, tokenizer):
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))  # Adjust token embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    finetuned_model_name = "./results"
    training_args = TrainingArguments(
        output_dir=finetuned_model_name,
        num_train_epochs=1,  # Adjust as needed
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,  # Set learning rate
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
        report_to=None,  # Enable TensorBoard logging
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator
    )

    trainer.train()

    model.save_pretrained(finetuned_model_name)
    tokenizer.save_pretrained(finetuned_model_name)
    return finetuned_model_name


def generate_text(device, model, tokenizer, prompt, max_length=150, temperature=0.8, top_k=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to device
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
    # Load the fine-tuned model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def generate(prompt, max_length=150, temperature=0.8, top_k=50):
        return generate_text(device, model, tokenizer, prompt, max_length, temperature, top_k)

    iface = gr.Interface(
        fn=generate,
        inputs=[
            gr.inputs.Textbox(lines=2, placeholder="Enter your prompt here...", label="Prompt"),
            gr.inputs.Slider(minimum=50, maximum=500, step=10, default=150, label="Max Length"),
            gr.inputs.Slider(minimum=0.1, maximum=1.0, step=0.1, default=0.8, label="Temperature"),
            gr.inputs.Slider(minimum=10, maximum=100, step=10, default=50, label="Top K"),
        ],
        outputs="text",
        title="Fine-Tuned GPT-2 Text Generator",
        description="Enter a prompt and adjust parameters to generate text using the fine-tuned GPT-2 model."
    )
    iface.launch()


def main():
    training_text = scrape_training_texts()
    tokenized_dataset, tokenizer = prepare_dataset(training_text)
    tuned_model = finetuned_model(tokenized_dataset, tokenizer)
    launch_ui(tuned_model)


if __name__ == "__main__":
    main()
