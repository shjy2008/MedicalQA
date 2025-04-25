# https://www.datacamp.com/tutorial/phi-3-tutorial
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from huggingface_hub import ModelCard, ModelCardData, HfApi
from datasets import load_dataset
from jinja2 import Template
from trl import SFTTrainer, SFTConfig
import yaml
import torch


# Step 2: Import required libraries and set configuration
# MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
NEW_MODEL_NAME = "opus-samantha-phi-3-mini-4k"
DATASET_NAME = "macadeliccc/opus_samantha"
SPLIT = "train"
MAX_SEQ_LENGTH = 1024
num_train_epochs = 1
license = "apache-2.0"
learning_rate = 1.41e-5
per_device_train_batch_size = 1
gradient_accumulation_steps = 1

is_bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

# Step 3: Load the model, tokenizer, and dataset
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
dataset = load_dataset(DATASET_NAME, split="train")

# Step 4: Preprocess the dataset
EOS_TOKEN=tokenizer.eos_token_id

# Select a subset of the data for test
dataset = dataset.select(range(100))


# This function is useless, because 'text' field is replaced in SFTTrainer.py
# def formatting_prompts_func(examples):
#     convos = examples["conversations"]
#     texts = []
#     mapper = {"system": "system\n", "human": "\nuser\n", "gpt": "\nassistant\n"}
#     end_mapper = {"system": "", "human": "", "gpt": ""}
#     for convo in convos:
#         text = "".join(f"{mapper[(turn := x['from'])]} {x['value']}\n{end_mapper[turn]}" for x in convo)
#         texts.append(f"{text}{EOS_TOKEN}")
#     return {"text": texts}

# dataset = dataset.map(formatting_prompts_func, batched=True)
# print(dataset['text'][0])

def change_role_name(examples):
    convos = examples["conversations"]
    for convo in convos:
        for x in convo:
            if x["from"] == "gpt":
                x["from"] = "assistant"
            elif x["from"] == "human":
                x["from"] = "user"
    return {"conversations": convos}
dataset = dataset.map(change_role_name, batched = True)
print(dataset)


# Step 5: Set training arguments
args = SFTConfig(
    # evaluation_strategy="steps",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=1e-4,
    fp16 = torch.cuda.is_available() and not is_bf16_supported,
    bf16 = is_bf16_supported,
    max_steps=-1,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_steps=10,
    output_dir=NEW_MODEL_NAME,
    optim="paged_adamw_32bit",
    lr_scheduler_type="linear",
    # These two parameters was in SFTTrainer, maybe the tutorial place them to the wrong place
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH
)

# Step 6: Fine-tune the model
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    # formatting_func=formatting_prompts_func
)
trainer.train()

# Step 7: Fine-tuning result
# TrainOutput(global_step=9, training_loss=0.7428549660576714, metrics={'train_runtime': 570.4105, 'train_samples_per_second': 0.526, 'train_steps_per_second': 0.016, 'total_flos': 691863632216064.0, 'train_loss': 0.7428549660576714, 'epoch': 2.4})