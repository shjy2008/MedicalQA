# pip install transformers datasets

# datasets: hugging face



from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import tensorflow as tf
import torch

def main():
    model_name = "dmis-lab/biobert-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # context = "Metformin is used to treat diabetes."
    # question = "Which drug is used to treat diabetes?"
    # # choices = ["Lisinopril", "Ibuprofen", "Metformin", "Aspirin"]

    # question = "What is the primary treatment for hypertension?"
    # context = "The primary treatment for hypertension includes lifestyle changes and antihypertensive medications such as ACE inhibitors, beta-blockers, and diuretics."

    # Example biomedical context and question
    # context = "Metformin is a first-line medication for type 2 diabetes treatment. It helps lower blood sugar levels."
    # question = "What is Metformin used for?"

    context = "abcde abcde. haha. ACB is used for fever. DEFka is used for diabetes"
    # context = "Metformin is a medication used primarily for the treatment of type 2 diabetes. Other medications include insulin, sulfonylureas, and GLP-1 receptor agonists."
    question = "What medicine is used for diabetes?"

    # Tokenize input
    inputs = tokenizer(question, context, return_tensors="pt").to(device)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract answer span
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits)
    answer = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx+1])

    print("Answer:", answer)  # Expected output: "type 2 diabetes treatment"
    
    
if __name__ == "__main__":
    main()