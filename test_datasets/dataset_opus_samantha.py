# https://www.datacamp.com/tutorial/phi-3-tutorial

# https://huggingface.co/datasets/macadeliccc/opus_samantha/viewer/default/train?row=0&views%5B%5D=train

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
DATASET_NAME = "macadeliccc/opus_samantha"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
dataset = load_dataset(DATASET_NAME, split="train")

EOS_TOKEN=tokenizer.eos_token_id

dataset = load_dataset(DATASET_NAME, split="train")

# Select a subset of the data for faster processing
dataset = dataset.select(range(10))

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = []
    mapper = {"system": "system\n", "human": "\nuser\n", "gpt": "\nassistant\n"}
    end_mapper = {"system": "", "human": "", "gpt": ""}
    for convo in convos:
        text = "".join(f"{mapper[(turn := x['from'])]} {x['value']}\n{end_mapper[turn]}" for x in convo)
        texts.append(f"{text}{EOS_TOKEN}")
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)
print(dataset['text'][8])