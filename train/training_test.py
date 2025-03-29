from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from datasets import Dataset

class TrainingTest:

    def load_model(self):
        # model_name = "microsoft/Phi-3-mini-4k-instruct"
        model_name = "Qwen/Qwen2.5-0.5B-instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = False)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code = False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        model.to(device)

        print("cuda available:", torch.cuda.is_available())
        print("device:", device)

        return model, tokenizer, device
    
    def infer_test(self, model, tokenizer, device):
        print ("FOR TEST")

        print ("model: ", model.name_or_path)


        prompt = """
        Question: A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?
        Choices:
        A: Disclose the error to the patient but leave it out of the operative report
        B: Disclose the error to the patient and put it in the operative report
        C: Tell the attending that he cannot fail to disclose this mistake
        D: Report the physician to the ethics committee
        E: Refuse to dictate the operative report
        Given five answer candidates, A, B, C, D, and E, choose the best answer choice.
        The answer is:
        """
        # Respond only with letter A, B, C, D, E. Don't need other information.

        print ("Prompt: ", prompt)
        print ("--------------")

        import time
        start_time = time.time()

        inputs = tokenizer(prompt, return_tensors = "pt", padding=False).to(device)
        with torch.no_grad(): # NO gradient calculation for inference
            output = model.generate(
                inputs["input_ids"],
                max_new_tokens = 10,
                attention_mask=inputs["attention_mask"],  # Use attention mask
                pad_token_id=tokenizer.eos_token_id,  # Set pad token ID
                # use_cache=False,
                do_sample = False
            )

        answer = tokenizer.decode(output[0], skip_special_tokens = True)

        answer = answer.replace(prompt, '').strip()

        finish_time = time.time()
        elapse_time = finish_time - start_time
        print("elapse_time: ", elapse_time)
        print ("--------------")

        print ("Answer: ", answer)


    def train_simple_test(self, model, tokenizer, device):
        # Train model on "TsinghuaC3I/UltraMedical" dataset

        print ("model: ", model.name_or_path)

        output_dir = "./Documents/fine_tuned_model"

        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)
        print ("previous model removed.")

        from datasets import load_dataset
        from transformers import Trainer, TrainingArguments

        ultraMedical = load_dataset("TsinghuaC3I/UltraMedical")

        data_list = ultraMedical["train"]
        print(len(data_list), type(data_list))

        # print((data_list[0:10]))
        first_N_data = data_list.select(range(0, 1)) # TODO

        print(type(first_N_data))

        def preprocess_function(data):
            input_text = [f"{conversation[0]['value']}" for conversation in data["conversations"]] # (context) + question + options
            target_text = [f"{conversation[1]['value']}" for conversation in data["conversations"]]  # CoT + final answer

            # print("input:", input_text)
            # print("target:", target_text)
            
            # inputs = tokenizer(input_text, padding="max_length", truncation=True, max_length=1024)
            # targets = tokenizer(target_text, padding="max_length", truncation=True, max_length=1024)
            # inputs["labels"] = targets["input_ids"]  # Set the target tokens as the 'labels'

            # return inputs

            prompt = input_text
            data['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
            data['labels'] = tokenizer(target_text, padding="max_length", truncation=True, return_tensors="pt").input_ids
            
            return data

        tokenized_train = first_N_data.map(preprocess_function, batched=True)

        training_args = TrainingArguments(
            output_dir = "./Documents/fine_tuned_model_checkpoints",
            save_strategy = "no", # TODO, now don't save checkpoints #"epoch",
            per_device_train_batch_size = 1, # As specified in the paper: batch_size: 32
            num_train_epochs = 1, # As specified in the paper: 3 epochs
            learning_rate = 1e-4, # As specified in the paper: 1e-4
            # save_total_limit = 1, # TODO, now keep only the last checkpoint
            # fp16 = True, # TODO
            logging_steps = 1,
            logging_dir = "./Documents/logs",  # Directory for logs
            
        )

        trainer = Trainer(
            model = model,
            args = training_args,
            train_dataset = tokenized_train
        )

        trainer.train()
        trainer.save_state()

        # Save the final model

        model.to(torch.bfloat16)  # Convert to bfloat16
        model.save_pretrained(output_dir, 
                                # save_function=torch.save,  # Use standard PyTorch save
                                # state_dict=model.state_dict(),  # Only save the model weights
                                # safe_serialization=True,  # More efficient serializationsave_optimizer_state=False
                            )
        tokenizer.save_pretrained(output_dir,
                                # legacy_format=False  # Use newer, more efficient format
                            )


if __name__ == "__main__":
    test = TrainingTest()
    model, tokenizer, device = test.load_model()
    # test.infer_test(model, tokenizer, device)
    test.train_simple_test(model, tokenizer, device)

    print("finished.")
