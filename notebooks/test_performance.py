import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch

# Ensure deterministic
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import random
import numpy as np
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from datasets import load_dataset
import re
import time
import logging

from context_retriever import ContextRetriever
from nltk.tokenize import sent_tokenize

# system_prompt = "You are a medical question answering assistant."
system_prompt = None

prompt_RAG = '''
Context:
{context}

Question:
{question}

{choices}
'''

# prompt_RAG = '''
# You are a medical question answering assistant.

# The following context may or may not be useful. Use it only if it helps answer the question.
# INSTRUCTIONS:
# - If the context directly helps answer the question, use it and cite appropriately
# - If the context is topically related but not diagnostically relevant, acknowledge it but rely on your medical knowledge
# - If the context might mislead you toward a less likely diagnosis, explicitly state why you're not following it

# Context:
# {context}

# Question:
# {question}

# {choices}
# '''

prompt_normal = '''
{question}

{choices}
'''

class DatasetPath:
    MedQA = "GBaker/MedQA-USMLE-4-options"
    MedMCQA = "openlifescienceai/medmcqa"
    PubMedQA = "qiaojin/PubMedQA"
    MMLU = "cais/mmlu"

class MMLU_Subset:
    Clinical_knowledge = "clinical_knowledge"
    Medical_genetics = "medical_genetics"
    Anatomy = "anatomy"
    Professional_medicine = "professional_medicine"
    College_biology = "college_biology"
    College_medicine = "college_medicine"
    

class TestPerformance():

    def __init__(self, model, tokenizer, device, is_encoder_decoder = False, temperature = 0.00001):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.is_encoder_decoder = is_encoder_decoder # like T5, no apply_chat_template
        self.temperature = temperature

        self.MAX_TOKEN_OUTPUT = 1024
        self.SPLIT = "test"

        regex_pattern=r"[\(\[]([A-Z])[\)\]]"
        self.regex = re.compile(regex_pattern)

        self.context_retriever = ContextRetriever(device)


    # input: a row of data in the dataset
    # output: (question, choices({"A":xxx, "B":xxx, ...}), answer_key("A" or "B"...))
    def get_question_info(self, data, dataset_path):
        if dataset_path == DatasetPath.MedQA:
            question = data["question"]
            choices = data["options"]
            answer_key = data["answer_idx"]
        elif dataset_path == DatasetPath.MedMCQA:
            question = data["question"]
            choices = {"A": data['opa'], "B": data['opb'], "C": data['opc'], "D": data['opd']}
            answer_key = chr(data['cop']+65)
        elif dataset_path == DatasetPath.PubMedQA:
            contexts = data["context"]["contexts"]
            question = f"Context: \n"
            for context in contexts:
                question += context + '\n'
            question += data["question"]
            choices = {'A': "yes", 'B': "no", 'C': "maybe"}
            answer = data["final_decision"]
            answer_dict = {"yes": 'A', "no": 'B', "maybe": 'C'}
            answer_key = answer_dict.get(answer)
        elif dataset_path == DatasetPath.MMLU:
            question = data['question']
            choices = {"A": data['choices'][0], "B": data['choices'][1], "C": data['choices'][2], "D": data['choices'][3]}
            answer_key = chr(data['answer']+65)
        
        return (question, choices, answer_key)

    def test_MedQA_response(self, use_RAG = True):# Test apply_chat_template // https://huggingface.co/docs/transformers/main/en/chat_templating
        print(self.model.name_or_path)

        def format_choices(choices):
            a = zip(list(choices.keys()), choices.values())
            final_answers = []
            for x,y in a:
                final_answers.append(f'[{x}] : {y}')
            return "\n".join(final_answers)


        def run_inference(content, model, tokenizer, max_new_tokens, temperature, do_sample):
            generate_kwargs = {
                "max_new_tokens": max_new_tokens, 
                "do_sample": do_sample, 
            }
            if do_sample == True:
                generate_kwargs["temperature"] = temperature

            if self.is_encoder_decoder:
                inputs = tokenizer(content, return_tensors="pt").to(self.device)
                outputs = model.generate(**inputs, **generate_kwargs)
            else:
                messages = [{"role": "user", "content": f"{content}"}]
                # add_generation_prompt indicates the start of a response
                inputs = tokenizer.apply_chat_template(messages, add_generation_prompt = True, return_tensors = "pt").to(self.device)
                # print("inputs:", tokenizer.apply_chat_template(messages, add_generation_prompt = True, tokenize = False))
                outputs = model.generate(inputs, **generate_kwargs)
            
            text = tokenizer.batch_decode(outputs)[0]
            return text

        if use_RAG:
            prompt = prompt_RAG
        else:
            prompt = prompt_normal
            

        examples = [
        # Training data index 0 (GBaker/MedQA-USMLE-4-options)
        # correct: D
        {"question": "A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient?",
        "choices": {
        "A": "Ampicillin",
        "B": "Ceftriaxone",
        "C": "Doxycycline",
        "D": "Nitrofurantoin"
        },
        "correct": "D",
        "source": "Training data index 0 (GBaker/MedQA-USMLE-4-options)"
        },

        # Training data index 1
        # correct: A
        {"question": "A 3-month-old baby died suddenly at night while asleep. His mother noticed that he had died only after she awoke in the morning. No cause of death was determined based on the autopsy. Which of the following precautions could have prevented the death of the baby?",
        "choices": {
        "A": "Placing the infant in a supine position on a firm mattress while sleeping",
        "B": "Keeping the infant covered and maintaining a high room temperature",
        "C": "Application of a device to maintain the sleeping position",
        "D": "Avoiding pacifier use during sleep"
        },
        "correct": "A",
        "source": "Training data index 1 (GBaker/MedQA-USMLE-4-options)"
        },

        # Training data index 2
        # correct: A
        {"question": "A mother brings her 3-week-old infant to the pediatrician's office because she is concerned about his feeding habits. He was born without complications and has not had any medical problems up until this time. However, for the past 4 days, he has been fussy, is regurgitating all of his feeds, and his vomit is yellow in color. On physical exam, the child's abdomen is minimally distended but no other abnormalities are appreciated. Which of the following embryologic errors could account for this presentation?",
        "choices": {
        "A": "Abnormal migration of ventral pancreatic bud",
        "B": "Complete failure of proximal duodenum to recanalize",
        "C": "Abnormal hypertrophy of the pylorus",
        "D": "Failure of lateral body folds to move ventrally and fuse in the midline"
        },
        "correct": "A",
        "source": "Training data index 2 (GBaker/MedQA-USMLE-4-options)"
        },

        # Test data index 0
        # correct: B
        {"question": "A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?",
        "choices":{
        "A": "Disclose the error to the patient and put it in the operative report",
        "B": "Tell the attending that he cannot fail to disclose this mistake",
        "C": "Report the physician to the ethics committee",
        "D": "Refuse to dictate the operative report"
        },
        "correct": "B",
        "source": "Test data index 0 (GBaker/MedQA-USMLE-4-options)"
        },

        ]

        for example in examples:
            formated_choices = format_choices(example["choices"])

            if use_RAG:
                context = self.get_RAG_context(example["question"], formated_choices)
                model_prompt = prompt.format(context = context, question = example["question"], choices = formated_choices)
            else:
                model_prompt = prompt.format(question = example["question"], choices = formated_choices)
    
            # print(model_prompt)
    
            output_text = run_inference(model_prompt, self.model, self.tokenizer, max_new_tokens = 1024, temperature = self.temperature, do_sample = False)
            # output_text = output_text.split("<|assistant|>")[-1]
            print("model output:", output_text)
            print("\nsource: ", example["source"])
            print("correct: ", example["correct"])
            print("\n\n")

    def run_inference_get_answer_letter(self, content, temperature, do_sample = False):
        self.model.eval()
        
        generate_kwargs = {
            "max_new_tokens": self.MAX_TOKEN_OUTPUT,
            "do_sample": do_sample,
        }
        if do_sample == True:
            generate_kwargs["temperature"] = temperature

        if self.is_encoder_decoder:
            inputs = self.tokenizer(content, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generate_kwargs)
        else:
            messages = []
            if system_prompt != None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": f"{content}"})
            inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
            # print("inputs:", inputs, inputs.shape)
            with torch.no_grad():
                outputs = self.model.generate(inputs, **generate_kwargs)
                # print("outputs:", outputs.shape)
        

        # print("temperature:", temperature)
        # print("inputs", inputs)
        
        text = self.tokenizer.batch_decode(outputs)[0]
        print("outputs text:", text)
        if not self.is_encoder_decoder:
            text = text.split("<|assistant|>")[-1]
        # answer = tokenizer.decode(output[0], skip_special_tokens = True)

        answer = self.extract_answer(text).strip("()")
        
        # print(f"answer: {answer}")
        # logging.info(f"answer: {answer}")
        
        return answer

    # def extract_answer(self, response):
    #     # matchFirst = re.search(r'the answer is .(\w).', response)
    #     matchFirst = re.search(r'correct answer is \[([A-D])\]', response)
    #     if matchFirst:
    #         return f"({matchFirst.group(1)})"
    #     match = self.find_match(self.regex, response) 
    #     if match:
    #         return f"({match})"
    #     return "[invalid]"
    
    def extract_answer(self, response):
        # Pattern for either [A-D] or (A-D), allow optional spaces or ** around it
        bracket_pattern = r'[\*\s]*[\[\(]\s*([A-D])\s*[\]\)][\*\s]*'
    
        # 1. Match "answer is ... [A]" or "(A)"
        m1 = re.search(r'answer is[:\s]*' + bracket_pattern, response, re.IGNORECASE)
        if m1:
            return f"({m1.group(1)})"
    
        # 2. Match "is ... [A]" or "(A)"
        m2 = re.search(r'is[:\s]*' + bracket_pattern, response, re.IGNORECASE)
        if m2:
            return f"({m2.group(1)})"
    
        # 3. Match the first [A] or (A) anywhere
        m3 = re.search(bracket_pattern, response)
        if m3:
            return f"({m3.group(1)})"
    
        return "[invalid]"
        
    def format_choices(self, choices, answer_key, mask_correct_answer):
        final_answers = []
        for i, (x, y) in enumerate(choices.items()):
            if mask_correct_answer and x == answer_key:
                y = "None of the other answers"
            final_answers.append(f'[{x}] : {y}')
        return "\n".join(final_answers)

    def find_match(self, regex, resp, convert_dict={}):
        match = regex.findall(resp)
        if match:
            match = match[-1]
            if isinstance(match, tuple):
                match = [m for m in match if m][0]
            match = match.strip()
            if match and match in convert_dict: 
                match = convert_dict[match]
        return match
            
    def test_accuracy(self, dataset_path, subset_name = None, is_ensemble = False, use_RAG = False, data_range = None, file_name = None,
                      topK_searchEngine = None, topK_SPLADE = None, topK_denseEmbedding = None, topK_crossEncoder = None, topK_LLM = None, score_threshold = None, 
                     pick_rag_index = None, mask_correct_answer = False, get_classifier_training_data = False, use_classifier = False, sentence_level_RAG = False, RRF_models = None):
        
        self.context_retriever.set_params(topK_searchEngine = topK_searchEngine, topK_SPLADE = topK_SPLADE, topK_denseEmbedding = topK_denseEmbedding, topK_crossEncoder = topK_crossEncoder, topK_LLM = topK_LLM, score_threshold = score_threshold, pick_rag_index = pick_rag_index, use_classifier = use_classifier, RRF_models = RRF_models)
        
        # Remove all handlers associated with the root logger
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Set log file config
        model_name = self.model.name_or_path.split("/")[-1]
        log_file_name = ""
        if file_name != None:
            log_file_name = file_name
        else:
            log_file_name = f'{model_name} - {dataset_path.split("/")[-1]}{"_" + subset_name if subset_name != None else ""}.txt'
        logging.basicConfig(
            filename = log_file_name,      # Log file name
            filemode = 'a',                    # Append mode
            format = '%(asctime)s - %(levelname)s - %(message)s',
            level = logging.INFO               # Log level
        )

        print(f"use_RAG: {use_RAG}")
        logging.info(f"use_RAG: {use_RAG}")
        print(f"topK_searchEngine: {topK_searchEngine}")
        logging.info(f"topK_searchEngine: {topK_searchEngine}")
        print(f"topK_SPLADE: {topK_SPLADE}")
        logging.info(f"topK_SPLADE: {topK_SPLADE}")
        print(f"topK_denseEmbedding: {topK_denseEmbedding}")
        logging.info(f"topK_denseEmbedding: {topK_denseEmbedding}")
        print(f"topK_crossEncoder: {topK_crossEncoder}")
        logging.info(f"topK_crossEncoder: {topK_crossEncoder}")
        print(f"topK_LLM: {topK_LLM}")
        logging.info(f"topK_LLM: {topK_LLM}")
        print(f"score_threshold: {score_threshold}")
        logging.info(f"score_threshold: {score_threshold}")
        print(f"RRF_models: {RRF_models}")
        logging.info(f"RRF_models: {RRF_models}")
        
        

        if subset_name == None:
            if dataset_path == DatasetPath.PubMedQA:
                subset_name = "pqa_labeled" # pqa_artificial, pqa_labeled, pqa_unlabeled

        # Load dataset
        dataset = load_dataset(dataset_path, name = subset_name, trust_remote_code = True)

        print(f"model: {self.model.name_or_path}")
        logging.info(f"model: {self.model.name_or_path}")
        
        start_time = time.time()

        split = self.SPLIT
        if dataset_path == DatasetPath.MedMCQA:
            split = "validation" # because test dataset doesn't have an answer
        elif dataset_path == DatasetPath.PubMedQA:
            split = "train" # only train in PubMedQA

        if get_classifier_training_data:
            split = "train"
        
        data_list = dataset[split]
        if data_range != None:
            data_list = data_list.select(data_range)
        count = 0
        correct_count = 0
        for data in data_list:
            question, choices, answer_key = self.get_question_info(data, dataset_path)

            # prompt = f'''\n{{question}}\n{{choices}}\n'''
            
            if use_RAG:
                prompt = prompt_RAG
            else:
                prompt = prompt_normal

            formated_choices = self.format_choices(choices, answer_key, mask_correct_answer)
            
            # content = prompt.format(question = question, choices = formated_choices)

            content_list = []
            if use_RAG:
                # print(f"question: {count}")
                # logging.info(f"question: {count}")

                if False:
                    doc_data_list = self.context_retriever.get_RAG_data_list(question, formated_choices)
                    to_try_index_list = [[], [0], [1], [2], [3], [4], [0, 1], [0, 1, 2]]
                    for doc_index_list in to_try_index_list:
                        context = ""
                        for doc_index in doc_index_list:
                            context += doc_data_list[doc_index]["content"]
                            context += "\n\n"
                        if context == "":
                            prompt = prompt_normal
                        else:
                            prompt = prompt_RAG
                        content = prompt.format(context = context, question = question, choices = formated_choices)
                        content_list.append(content)
                    
                    for doc_data in doc_data_list:
                        del doc_data["content"]
                        
                    print(f"RAG data: {doc_data_list}")
                    logging.info(f"RAG data: {doc_data_list}")
                else:
                    context = self.context_retriever.get_RAG_context(question, formated_choices)
                    if sentence_level_RAG:
                        sentences = sent_tokenize(context)
                        for sent in sentences:
                            content = prompt.format(context = sent, question = question, choices = formated_choices)
                            content_list.append(content)
                    else:
                        if context == None or context == "":
                            prompt = prompt_normal
                        content = prompt.format(context = context, question = question, choices = formated_choices)
                        content_list.append(content)
                    

                # The following 4 lines of code is for logging the scores of documents
                # count += 1
                # print(f"question {count}/{len(data_list)} score_list: {score_list}")
                # logging.info(f"question {count}/{len(data_list)} score_list: {score_list}")
                # continue
                
            else:
                content = prompt.format(question = question, choices = formated_choices)
                content_list.append(content)
            
            # print("messages: ", messages)
            # break
            
            # print("correct answer: ", answer_idx)

            count += 1
            
            for content in content_list:
                if is_ensemble:
                    answer_dict = {}
                    for i in range(0, 5):
                        # inputs = tokenizer(model_prompt, return_tensors = "pt", padding=False).to(device)
                        current_answer = self.run_inference_get_answer_letter(content, temperature = 0.7, do_sample = True)
                        if current_answer in answer_dict:
                            answer_dict[current_answer] += 1
                        else:
                            answer_dict[current_answer] = 1
                    answer = max(answer_dict, key = answer_dict.get)
                else:
                    answer = self.run_inference_get_answer_letter(content, temperature = self.temperature, do_sample = False)
                
                correct_answer = answer_key
            
                is_correct = (answer == correct_answer)
                # print("Correct!!!" if is_correct else "Wrong")
    
                if get_classifier_training_data:
                    print(f"[input]{content}[output]{is_correct}")
                    logging.info(f"[input]{content}[output]{is_correct}")
            
            
                if is_correct:
                    correct_count += 1
    
                if data_range != None:
                    print(f"question {count}/{len(data_list)} num:{data_range[count - 1] + 1} answer:{answer} correct_answer:{correct_answer} {is_correct}")
                    logging.info(f"question {count}/{len(data_list)} num:{data_range[count - 1] + 1} answer:{answer} correct_answer:{correct_answer} {is_correct}")
                else:
                    print(f"question {count}/{len(data_list)} answer:{answer} correct_answer:{correct_answer} {is_correct}")
                    logging.info(f"question {count}/{len(data_list)} answer:{answer} correct_answer:{correct_answer} {is_correct}")
        
        accuracy = correct_count / count
        print(f"Total questions: {count}, correct: {correct_count}, accuracy: {accuracy}")
        logging.info(f"Total questions: {count}, correct: {correct_count}, accuracy: {accuracy}")
        
        finish_time = time.time()
        elapse_time = finish_time - start_time
        print(f"elapse_time: {elapse_time}")
        logging.info(f"elapse_time: {elapse_time}")
