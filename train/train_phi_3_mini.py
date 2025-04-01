from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from datetime import datetime
import shutil
from test_performance import TestPerformance

class ModelTrainer():
    def __init__(self):

        self.training_data_count = None # 30000 # set to None if train on the entire dataset
        self.max_length = 1024
        self.output_dir = "./fine_tuned_model_entire_UltraMedical_batch_32"

        self.model = None
        self.tokenizer = None
        self.device = None

        self.dataset = None
        self.tokenized_training_data = None

        self.check_gpu()

    # Check GPU name and status
    def check_gpu(self):
        print(f"---------- start checking GPU -----------")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print("torch.cuda.is_bf16_supported(): ", torch.cuda.is_bf16_supported())
        else:
            print("GPU not available.")
        print(f"---------- finish checking GPU -----------")

    # Load model
    def load_model(self, model_name):
        print(f"---------- start loading model:{model_name} -----------")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = False)
        print("finish loading tokenizer")
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code = False, 
                                                    #  torch_dtype= torch.float16 # Sometimes RuntimeError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'
                                                     torch_dtype = torch.bfloat16
                                                     )
        print("finish loading model")
        print("torch_dtype:", model.config.torch_dtype)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print("cuda available:", torch.cuda.is_available())
        print("device:", device)

        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        print(f"---------- finish loading model:{self.model.name_or_path} -----------")

    # Load dataset
    def load_dataset(self):
        # self.dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en")["train"]
        self.dataset = load_dataset("TsinghuaC3I/UltraMedical")["train"]

        print(f"---------- finish loading dataset -----------\n{self.dataset}")

    # Preprocess training data
    def preprocess_training_data(self):
        print(f"----------- start preprocessing training data ------------")

        if self.training_data_count != None:
            first_N_data = self.dataset.select(range(0, self.training_data_count))
        else:
            first_N_data = self.dataset

        # print(first_N_data[0])

        # conversations = [{'from': 'human', 'value': 'A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient?\n\nA. Ampicillin\nB. Ceftriaxone\nC. Doxycycline\nD. Nitrofurantoin'}, {'from': 'gpt', 'value': "The clinical scenario suggests that the pregnant woman is likely experiencing a urinary tract infection (UTI), given her symptoms of burning upon urination. During pregnancy, the risk of UTIs increases due to physiological changes, and it is important to treat them to prevent complications such as pyelonephritis or preterm labor.\n\nWhen considering antibiotic therapy for a UTI in a pregnant woman, it's crucial to choose an agent that is safe for both the mother and the fetus. Let's evaluate the options:\n\nA. Ampicillin - This is a penicillin antibiotic that is generally considered safe during pregnancy and can be used to treat UTIs. However, it may not be the first choice due to resistance patterns.\n\nB. Ceftriaxone - This is a cephalosporin antibiotic that is also generally safe in pregnancy and effective against a broad range of bacteria. It is typically used for more complicated infections or when other antibiotics are not suitable.\n\nC. Doxycycline - This is a tetracycline antibiotic that is contraindicated during pregnancy because it can affect fetal bone growth and discoloration of teeth.\n\nD. Nitrofurantoin - This antibiotic is commonly used to treat uncomplicated UTIs and is considered safe during most of the pregnancy. However, it is not recommended at the very end of pregnancy (after 38 weeks) due to the risk of hemolytic anemia in the newborn.\n\nGiven that the patient is at 22 weeks gestation and has an uncomplicated UTI, the best choice would be an antibiotic that is safe and commonly used for UTIs in pregnancy.\n\nSo, the answer is D. Nitrofurantoin."}]

        # Convert training data to training format:
        # inputs: <|user|> <question>\n\n[A] : xxx\n[B] : xxx\n[C] : xxx\n[D] : xxx\n<|end|><|assistant|>xxx(CoT), the answer is xxx<|end|><|end_of_text|>
        # labels: -100, -100, ..., -100, xxx(CoT), the answer is xxx<|end|><|end_of_text|>

        self.tokenized_training_data = first_N_data.map(self.convert_to_tokenized_training_data, batched = False)

        print(f"--------- finish preprocessing training data -----------\n{self.tokenized_training_data}")

    # Process UltraMedical dataset, convert data format to chat template format https://huggingface.co/docs/transformers/v4.50.0/en/chat_templating#applychattemplate
    def convert_to_chat_format(self, conversations, is_input):
        chat = []
        for conversation in conversations:
            if conversation["from"] == "human":
                for option_letter in ["A", "B", "C", "D"]:
                    conversation["value"] = conversation["value"].replace(f"{option_letter}.", f"[{option_letter}] :")
                chat.append({"role": "user", "content": conversation["value"]})
            elif is_input == False and conversation["from"] == "gpt":
                for option_letter in ["A", "B", "C", "D"]:
                    conversation["value"] = conversation["value"].replace(f"{option_letter}.", f"[{option_letter}].")
                chat.append({"role": "assistant", "content": conversation["value"]})
        return chat

    def convert_to_tokenized_training_data(self, data):
        questions = self.convert_to_chat_format(data["conversations"], is_input = True) #[convert_to_chat_format(conversations, is_input = True) for conversations in conversations_list]
        template_questions = self.tokenizer.apply_chat_template(questions, tokenize = False, add_generation_prompt = True)
        
        questions_and_answers = self.convert_to_chat_format(data["conversations"], is_input = False) #[convert_to_chat_format(conversations, is_input = False) for conversations in conversations_list]
        template_questions_and_answers = self.tokenizer.apply_chat_template(questions_and_answers, tokenize = False, add_generation_prompt = False)

        # print("template_questions", '\n', template_questions, '\n------')
        # print("template_expected_outputs", '\n', template_questions_and_answers, '\n-----')

        tokenized_questions = self.tokenizer(template_questions, padding = False, truncation=True, max_length=self.max_length)
        tokenized_inputs = self.tokenizer(template_questions_and_answers, padding=False, truncation=True, max_length=self.max_length)

        # print("tokenized_questions", '\n', tokenized_questions, '\n------')
        # print("tokenized_inputs", '\n', tokenized_inputs, '\n-----')

        questions_input_ids = tokenized_questions["input_ids"]
        all_content_input_ids = tokenized_inputs["input_ids"]
        tokenized_labels = [-100] * len(questions_input_ids) + all_content_input_ids[len(questions_input_ids):]

        # print("tokenized_labels", '\n', tokenized_labels, '\n-----')
        
        # Add padding
        padding_num = self.max_length - len(tokenized_inputs["input_ids"])
        if padding_num > 0:
            tokenized_inputs["input_ids"] = [self.tokenizer.eos_token_id] * padding_num + tokenized_inputs["input_ids"]
            tokenized_inputs["attention_mask"] = [0] * padding_num + tokenized_inputs["attention_mask"]
            tokenized_labels = [self.tokenizer.eos_token_id] * padding_num + tokenized_labels

        # print(tokenized_inputs)
        # print(len(tokenized_inputs["input_ids"]), len(tokenized_inputs["attention_mask"]), len(tokenized_inputs["labels"]))

        training_data = {"input_ids": tokenized_inputs["input_ids"], 
                        "attention_mask": tokenized_inputs["attention_mask"],
                        "labels": tokenized_labels}
        
        # print("input_ids:", len(training_data["input_ids"]), training_data["input_ids"])

        # print("attention_mask:", len(training_data["attention_mask"]), training_data["attention_mask"])

        # print("labels:", len(training_data["labels"]), training_data["labels"])


        training_data = {k: torch.tensor(v) for k, v in training_data.items()}

        # print("training_data:", training_data)
        
        return training_data

    # Train model
    def train_model(self):
        print(f"----------- start training model:{self.model.name_or_path} ------------")
        training_args = TrainingArguments(
            output_dir = "./fine_tuned_model_checkpoints",
            save_strategy = "steps", # save checkpoints # "epoch", "steps", "no"
            save_steps = 20000, # total_steps = dataset_size / batch_size
            # save_total_limit = 5, # keep only the last N checkpoint
            per_device_train_batch_size = 4, # As specified in the paper: batch_size: 32
            num_train_epochs = 3, # As specified in the paper: 3 epochs
            learning_rate = 1e-4, # As specified in the paper: 1e-4
            fp16 = False, # TODO
            bf16 = True, # Sometimes RuntimeError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'
            logging_steps = 100,
            logging_dir = "./logs",  # Directory for logs
            # report_to = ["tensorboard"],  # Enable logging to TensorBoard
        )

        trainer = Trainer(
            model = self.model,
            args = training_args,
            train_dataset = self.tokenized_training_data
        )

        trainer.train()

        print(f"----------- finish training model:{self.model.name_or_path} ------------")
    
    def remove_previous_model(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)
        print ("---------- finish removing previous fine-tuned model ----------")


    # Save the final model
    def save_trained_model(self):
        print(f"----------- start saving model {self.model.name_or_path} to:{self.output_dir} ------------")

        self.model.to(torch.bfloat16)  # Convert to bfloat16
        self.model.save_pretrained(self.output_dir, 
                                # save_function=torch.save,  # Use standard PyTorch save
                                # state_dict=model.state_dict(),  # Only save the model weights
                                # safe_serialization=True,  # More efficient serializationsave_optimizer_state=False
                            )
        self.tokenizer.save_pretrained(self.output_dir,
                                # legacy_format=False  # Use newer, more efficient format
                            )
        
        print(f"----------- finish saving model {self.model.name_or_path} to: {self.output_dir} -----------")
    
    def load_fine_tuned_model(self):
        self.load_model(self.output_dir)

    # Test the response format of current model (MedQA question 0)
    def test_model_MedQA_response(self):
        print(f"----------- start testing model {self.model.name_or_path} performance ------------")
        test = TestPerformance(self.model, self.tokenizer, self.device)
        test.test_MedQA_response()
        print(f"----------- finish testing model {self.model.name_or_path} performance ------------")
    
    # Test the performance on MedQA test dataset of current model
    def test_model_MedQA_accuracy(self):
        print(f"----------- start testing model {self.model.name_or_path} accuracy ------------")
        test = TestPerformance(self.model, self.tokenizer, self.device)
        test.test_MedQA_test_data_accuracy()
        print(f"----------- finish testing model {self.model.name_or_path} accuracy ------------")


if __name__ == "__main__":
    print("time:", datetime.now())

    trainer = ModelTrainer()

    model_name = "microsoft/Phi-3-mini-4k-instruct"
    # model_name = "Qwen/Qwen2.5-0.5B"
    # model_name = "Qwen/Qwen2.5-0.5B-instruct"
    # model_name = "KrithikV/MedMobile"
    trainer.load_model(model_name)
    # trainer.load_fine_tuned_model()

    # For training
    trainer.load_dataset()
    trainer.preprocess_training_data()
    trainer.train_model()
    trainer.save_trained_model()

    # For Testing
    trainer.load_fine_tuned_model()
    # trainer.load_model(model_name)
    trainer.test_model_MedQA_response()
    trainer.test_model_MedQA_accuracy()

    print("All done.")

