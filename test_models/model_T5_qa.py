# pip install transformers datasets

# datasets: hugging face

# T5: text-to-text model, it can generate new text

from datasets import load_dataset
from transformers import BertForMultipleChoice, BertTokenizer, BertForQuestionAnswering, T5Tokenizer, T5ForConditionalGeneration
import tensorflow as tf
import torch

def main():
    # model_name = "dmis-lab/biobert-base-cased-v1.1"
    model_name = "google-t5/t5-base"
    # tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    # model = BertForMultipleChoice.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define context and question
    context = "abcde abcde. haha. ACB is used for fever. DEFka is used for diabetes"
    # context = "Metformin is a medication used primarily for the treatment of type 2 diabetes. Other medications include insulin, sulfonylureas, and GLP-1 receptor agonists."
    # context = "abc"
    # question = "What medicine is used for fever?"
    question = "What for afever?"

    # Prepare the input text for the T5 model
    input_text = f"question: {question} context: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate the answer
    with torch.no_grad():
        output = model.generate(input_ids)

    # Decode the generated answer
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Predicted answer: {answer}")

if __name__ == "__main__":
    main()