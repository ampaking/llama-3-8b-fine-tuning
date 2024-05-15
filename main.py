import streamlit as st
import os
import pandas as pd
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from unsloth import FastLanguageModel

# Ensure the sample_data directory exists
os.makedirs('sample_data', exist_ok=True)

# Title of the app
st.title("CSV Uploader and Manager")

# Function to list files in the sample_data directory
def list_files():
    files = os.listdir('sample_data')
    if files:
        st.write("### Existing files in sample_data directory:")
        for file in files:
            st.write(file)
    else:
        st.write("No files in sample_data directory.")

# Function to delete all files in the sample_data directory
def delete_all_files():
    files = os.listdir('sample_data')
    for file in files:
        os.remove(os.path.join('sample_data', file))

# Show existing files in the sample_data directory
list_files()

# Add a button to delete all files and reset the app
if st.button("Reset App (Delete all uploaded files)"):
    delete_all_files()
    st.success("All files deleted from sample_data directory.")
    st.experimental_rerun()

# File uploader allowing multiple file uploads
uploaded_files = st.file_uploader("Choose CSV files", accept_multiple_files=True, type=["csv"])

# Process each uploaded file
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save the file to the sample_data directory
        save_path = os.path.join('sample_data', uploaded_file.name)
        with open(save_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"File {uploaded_file.name} saved to sample_data directory.")

    # Show the updated list of files after upload
    list_files()

# Function to load and prepare the model
@st.cache_resource
def load_model():
    max_seq_length = 2048
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    return model, tokenizer

# Function to train the model
def train_model(model, tokenizer):
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output)
            texts.append(text)
        return {"text": texts}

    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )
    trainer_stats = trainer.train()
    return trainer_stats

# Function to chat with the model
def chat_with_llama3_8b(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=2048)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Load the model once and store it in Streamlit's session state
if 'model' not in st.session_state:
    st.session_state.model, st.session_state.tokenizer = load_model()

# Button to train the model
if st.button("Train Model"):
    if os.listdir('sample_data'):
        train_model(st.session_state.model, st.session_state.tokenizer)
        st.success("Model trained successfully.")
    else:
        st.warning("No files to train the model. Please upload CSV files first.")

# Text input for chatting with the model
prompt = st.text_input("Enter your prompt for the model:")
if st.button("Chat with Model"):
    if prompt:
        response = chat_with_llama3_8b(st.session_state.model, st.session_state.tokenizer, prompt)
        st.write("### Model's Response:")
        st.write(response)
    else:
        st.warning("Please enter a prompt.")
