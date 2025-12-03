from transformers import AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from trl import SFTTrainer, SFTConfig
from datetime import datetime
import shutil
from test_performance import TestPerformance, DatasetPath, MMLU_Subset

class ModelTrainer():
    def __init__(self, output_dir = None):

        self.training_data_count = None #10 # None #30000 # None # set to None if train on the entire dataset
        self.max_length = 1024
        # 1: inputs: <|user|>xxx<|end|><|assistant|>xxx<|end|>    labels: -100, -100, ..., -100, xxx<|end|>  labels mask the question, inputs contain answer
        # 2: only pass plain text <|user|>xxx<|end|><|assistant|>xxx<|end|> as training data
        # 3: only pass plain text user\n xxx assistant\n xxx as training data
        # 4: reproduce 1
        # 5: inputs: <|user|>xxx<|end|><|assistant|>   labels: -100, -100, ..., -100, xxx<|end|>  labels mask the question, inputs don't contain answer
        # self.output_dir = "./fine_tuned_model"
        # self.output_dir = "/projects/sciences/computing/sheju347/MedicalQA/train/saved_models/base/10-18-UltraMedical-batchsize8-1e-4-epoch3-5e-5-epoch2"

        self.output_dir = output_dir

        self.model = None
        self.tokenizer = None
        self.device = None

        self.dataset = None
        self.training_data = None

        self.is_bf16_supported = False
        self.check_gpu()

    # Check GPU name and status
    def check_gpu(self):
        print(f"---------- start checking GPU -----------")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            self.is_bf16_supported = torch.cuda.is_bf16_supported()
            print("torch.cuda.is_bf16_supported(): ", self.is_bf16_supported)
        else:
            print("GPU not available.")
        print(f"---------- finish checking GPU -----------")

    # Load model
    def load_model(self, model_name, lora_adapter_path = None, tokenizer_path = None):
        print(f"---------- start loading model:{model_name, tokenizer_path} -----------")
        if tokenizer_path == None:
            tokenizer_path = model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code = False)
        print("finish loading tokenizer")
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code = False,
                                                    #  torch_dtype= torch.float16 # Sometimes RuntimeError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'
                                                     torch_dtype = torch.bfloat16 if self.is_bf16_supported else torch.float16
                                                     # torch_dtype = torch.float32
                                                     )

        if lora_adapter_path != None:
            print(f"applying LoRA adapter from: {lora_adapter_path}")
            model = PeftModel.from_pretrained(model, lora_adapter_path)
            print("LoRA adapter applied")

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

    # Load model
    def load_model_t5(self, model_name):
        print(f"---------- start loading model:{model_name} -----------")
        tokenizer = T5Tokenizer.from_pretrained(model_name, trust_remote_code = False)
        print("finish loading tokenizer")
        model = T5ForConditionalGeneration.from_pretrained(model_name, trust_remote_code = False, 
                                                    #  torch_dtype= torch.float16 # Sometimes RuntimeError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'
                                                     torch_dtype = torch.bfloat16 if self.is_bf16_supported else torch.float16
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

        if self.training_data_count != None:
            first_N_data = self.dataset.select(range(0, self.training_data_count))
        else:
            first_N_data = self.dataset

        print(f"----------- start preprocessing training data ------------")
        print(first_N_data)

        # conversations = [{'from': 'human', 'value': 'A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient?\n\nA. Ampicillin\nB. Ceftriaxone\nC. Doxycycline\nD. Nitrofurantoin'}, {'from': 'gpt', 'value': "The clinical scenario suggests that the pregnant woman is likely experiencing a urinary tract infection (UTI), given her symptoms of burning upon urination. During pregnancy, the risk of UTIs increases due to physiological changes, and it is important to treat them to prevent complications such as pyelonephritis or preterm labor.\n\nWhen considering antibiotic therapy for a UTI in a pregnant woman, it's crucial to choose an agent that is safe for both the mother and the fetus. Let's evaluate the options:\n\nA. Ampicillin - This is a penicillin antibiotic that is generally considered safe during pregnancy and can be used to treat UTIs. However, it may not be the first choice due to resistance patterns.\n\nB. Ceftriaxone - This is a cephalosporin antibiotic that is also generally safe in pregnancy and effective against a broad range of bacteria. It is typically used for more complicated infections or when other antibiotics are not suitable.\n\nC. Doxycycline - This is a tetracycline antibiotic that is contraindicated during pregnancy because it can affect fetal bone growth and discoloration of teeth.\n\nD. Nitrofurantoin - This antibiotic is commonly used to treat uncomplicated UTIs and is considered safe during most of the pregnancy. However, it is not recommended at the very end of pregnancy (after 38 weeks) due to the risk of hemolytic anemia in the newborn.\n\nGiven that the patient is at 22 weeks gestation and has an uncomplicated UTI, the best choice would be an antibiotic that is safe and commonly used for UTIs in pregnancy.\n\nSo, the answer is D. Nitrofurantoin."}]

        # Convert training data to training format:
        # inputs: <|user|> <question>\n\n[A] : xxx\n[B] : xxx\n[C] : xxx\n[D] : xxx\n<|end|><|assistant|>xxx(CoT), the answer is xxx<|end|><|end_of_text|>
        # labels: -100, -100, ..., -100, xxx(CoT), the answer is xxx<|end|><|end_of_text|>

        # self.training_data = first_N_data.map(self.convert_training_data, batched = True, 
        #                                       remove_columns = [col for col in first_N_data.column_names if col != "conversations"])
        # print("Example training data:\n", self.training_data["text"][8])
        self.training_data = first_N_data.map(self.convert_to_tokenized_training_data, batched = False)

        # print("example data:", self.converted_training_data["text"][0])
        print(f"--------- finish preprocessing training data -----------\n{self.training_data}")

    # Process UltraMedical dataset, convert data format to chat template format https://huggingface.co/docs/transformers/v4.50.0/en/chat_templating#applychattemplate
    def convert_to_chat_format(self, conversations, is_input):
        chat = []
        for conversation in conversations:
            if conversation["from"] == "human":
                value = conversation["value"]
                for option_letter in ["A", "B", "C", "D"]:
                    value = value.replace(f"{option_letter}.", f"[{option_letter}] :")
                chat.append({"role": "user", "content": value})
            elif is_input == False and conversation["from"] == "gpt":
                value = conversation["value"]
                for option_letter in ["A", "B", "C", "D"]:
                    value = value.replace(f"{option_letter}.", f"[{option_letter}].")
                chat.append({"role": "assistant", "content": value})
        return chat

    # 1. convert to chat format: [{"role": user, "content": <question>}, {"role": assistant, "content": <answer>}]
    # 2. apply_chat_template: <|user|> question <|end|> <|assistant|> answer <|end|> <|endoftext|>
    # 3. mask -100 to the question part of labels: -100, -100, ..., -100, answer <|end|> <|endoftext|>
    # inputs: question, answer
    # labels: -100, -100, ..., -100, answer
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

    # This function is simply feed the JSON format to the model and use SFTTrainer, don't use any -100 mask
    # The same as the Phi-3 tutorial https://www.datacamp.com/tutorial/phi-3-tutorial
    def convert_training_data(self, data):
        convos = data["conversations"]
        new_convos = []
        for conv in convos:
            new_conv = self.convert_to_chat_format(conv, False)
            new_convos.append(new_conv)
        return {"conversations": new_convos}

    # Train model
    def train_model(self, resume_from_checkpoint=False, batch_size = 4, num_train_epochs = 3):
        print(f"----------- start training model:{self.model.name_or_path} ------------")

        # The same as the Phi-3 tutorial https://www.datacamp.com/tutorial/phi-3-tutorial
        # training_args = SFTConfig(
        #     output_dir = self.output_dir,
        #     save_strategy = "steps", #"steps", # save checkpoints # "epoch", "steps", "no"
        #     save_steps = 50000, # total_steps = dataset_size / batch_size
        #     # save_total_limit = 5, # keep only the last N checkpoint
        #     per_device_train_batch_size = 4, # As specified in the paper: batch_size: 32
        #     num_train_epochs = 3, # As specified in the paper: 3 epochs
        #     learning_rate = 1e-4, # As specified in the paper: 1e-4
        #     fp16 = torch.cuda.is_available() and not self.is_bf16_supported,
        #     bf16 = self.is_bf16_supported, # Sometimes RuntimeError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'
        #     logging_steps = 100,
        #     logging_dir = "./logs",  # Directory for logs
        #     # report_to = ["tensorboard"],  # Enable logging to TensorBoard
        #     max_length = self.max_length, # SFTTrainer will truncate (no this param in TrainingArgument)

        #     # dataset_text_field = "text",
        # )

        # trainer = SFTTrainer(
        #     model = self.model,
        #     args = training_args,
        #     train_dataset = self.training_data,
        #     # dataset_text_field = "text",
        #     # formatting_func = self.formatting_prompts_func
        # )

        training_args = TrainingArguments(
            output_dir = self.output_dir,
            save_strategy = "epoch", #"steps", # save checkpoints # "epoch", "steps", "no"
            # save_steps = 50000, # total_steps = dataset_size / batch_size
            # save_total_limit = 5, # keep only the last N checkpoint
            per_device_train_batch_size = batch_size, # As specified in the paper: batch_size: 32
            num_train_epochs = num_train_epochs, # As specified in the paper: 3 epochs
            learning_rate = 1e-4, # As specified in the paper: 1e-4
            fp16 = torch.cuda.is_available() and not self.is_bf16_supported,
            bf16 = self.is_bf16_supported, # Sometimes RuntimeError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'
            logging_steps = 100,
            logging_dir = "./logs",  # Directory for logs
            # report_to = ["tensorboard"],  # Enable logging to TensorBoard
        )

        trainer = Trainer(
            model = self.model,
            args = training_args,
            train_dataset = self.training_data
        )

        trainer.train(resume_from_checkpoint = resume_from_checkpoint)

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
    def test_model_accuracy(self, dataset_path, subset_name = None, is_ensemble = False):
        print(f"----------- start testing model {self.model.name_or_path} accuracy ------------")
        test = TestPerformance(self.model, self.tokenizer, self.device)
        test.test_accuracy(dataset_path, subset_name = subset_name, is_ensemble = is_ensemble)
        print(f"----------- finish testing model {self.model.name_or_path} accuracy ------------")


if __name__ == "__main__":
    print("time:", datetime.now())

    output_dir = "/projects/sciences/computing/sheju347/MedicalQA/train/saved_models/base_phi4/11-20-phi4-mini-base-UltraMedical"
    
    trainer = ModelTrainer(output_dir = output_dir)

    # model_name = "microsoft/Phi-3-mini-4k-instruct"
    # model_name = "microsoft/Phi-4-mini-instruct"
    
    # model_name = "/projects/sciences/computing/sheju347/MedicalQA/train/saved_models/base/10-15-UltraMedical-batchsize8-bf16"
    # model_name = "google/flan-t5-xl"
    # model_name = "Qwen/Qwen3-4B-Instruct-2507"
    # model_name = "Qwen/Qwen2.5-0.5B"
    # model_name = "Qwen/Qwen2.5-0.5B-instruct"
    # model_name = "KrithikV/MedMobile"
    # model_name = "./saved_models/fine_tuned_model_entire_UltraMedical_batch_4"
    # model_name = "/projects/sciences/computing/sheju347/MedicalQA/train/saved_models/base_qwen/11-11-Qwen3-4B-base-UltraMedical/checkpoint-307197"
    # tokenizer_path = "/projects/sciences/computing/sheju347/MedicalQA/train/saved_models/base_qwen/11-11-Qwen3-4B-base-UltraMedical"
    # trainer.load_model(model_name, tokenizer_path = tokenizer_path)
    
    trainer.load_model(model_name)
    # trainer.load_fine_tuned_model()

    # # For training
    trainer.load_dataset()
    # # trainer.convert_to_tokenized_training_data(trainer.dataset[0])
    trainer.preprocess_training_data()
    trainer.train_model(resume_from_checkpoint = model_name)
    # trainer.train_model_t5()
    trainer.save_trained_model()

    # # # For Testing
    # trainer.load_fine_tuned_model()
    # trainer.test_model_MedQA_response()
    # trainer.test_model_accuracy(DatasetPath.MedMCQA, is_ensemble = False)

    print("All done.")

