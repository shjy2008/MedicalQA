# pip install transformers datasets

# datasets: hugging face

from datasets import load_dataset
from transformers import BertForMultipleChoice, BertTokenizer, BertForQuestionAnswering, T5Tokenizer, T5ForConditionalGeneration
import tensorflow as tf
import torch

def main():
    model_name = "dmis-lab/biobert-base-cased-v1.1"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMultipleChoice.from_pretrained(model_name)

    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    context = "Metformin is used to treat diabetes."
    question = "Which drug is used to treat diabetes?"
    choices = ["Lisinopril", "Ibuprofen", "Metformin", "Aspirin"]
    
    # Tokenize inputs for each choice
    encoding = tokenizer(
        [[f"{context} {question}", choice] for choice in choices],  # Format: [(question, choice1), (question, choice2), ...]
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # Move tensors to device
    input_ids = encoding["input_ids"].unsqueeze(0).to(device)  # Add batch dimension
    attention_mask = encoding["attention_mask"].unsqueeze(0).to(device)

    # Get model predictions
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits

    # Select the choice with the highest score
    predicted_idx = logits.argmax().item()
    answer = choices[predicted_idx]

    print(f"Predicted answer: {answer}")

if __name__ == "__main__":
    main()