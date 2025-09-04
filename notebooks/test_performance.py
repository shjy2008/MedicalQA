import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from datasets import load_dataset
import torch
import re
import time
import logging
import requests
from sentence_transformers import CrossEncoder
from FlagEmbedding import FlagReranker
from sentence_transformers import SparseEncoder

# RankLLM
from rank_llm.data import Query, Candidate, Request
from rank_llm.rerank import Reranker
from rank_llm.rerank.pairwise.duot5 import DuoT5
from rank_llm.rerank.pointwise.monot5 import MonoT5
from rank_llm.rerank.listwise import ZephyrReranker, VicunaReranker, RankListwiseOSLLM

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



prompt_RAG = '''
You are a medical question answering assistant.

The following context may or may not be useful. Use it only if it helps answer the question.
INSTRUCTIONS:
- If the context directly helps answer the question, use it and cite appropriately
- If the context is topically related but not diagnostically relevant, acknowledge it but rely on your medical knowledge
- If the context might mislead you toward a less likely diagnosis, explicitly state why you're not following it

Context:
{context}

Question:
{question}

{choices}
'''

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

        # Step 0: BM25 search engine

        # Step 1: SPLADE model (lexical sparse retrieval)
        self.splade_model = SparseEncoder("naver/splade-v3")
        # self.splade_model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

        # Step 2: Cross-encoder model (e.g. MonoBERT)
        # self.crossEncoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
        # self.crossEncoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L12-v2")
        # self.crossEncoder_model = CrossEncoder("cross-encoder/ms-marco-electra-base")
        # self.crossEncoder_model = FlagReranker('BAAI/bge-reranker-v2-m3')
        # self.crossEncoder_model = FlagReranker('BAAI/bge-reranker-v2-gemma')
        self.crossEncoder_model = Reranker(MonoT5("castorini/monot5-3b-med-msmarco", context_size = 4096, batch_size = 16))


        # Step 3: More powerful model (DuoBERT, LLM, ...)
        # self.llm_reranker = Reranker(DuoT5(model = "castorini/duot5-3b-med-msmarco"))
        # self.llm_reranker = ZephyrReranker()
        # self.llm_reranker = VicunaReranker()

        model_coordinator = RankListwiseOSLLM(
            model="Qwen/Qwen2.5-7B-Instruct",
        )
        self.llm_reranker = Reranker(model_coordinator)
        

        # RAG
        self.topK_searchEngine = 100
        self.topK_SPLADE = 10
        self.topK_crossEncoder = 3
        self.topK_LLM = 3


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

    def get_RAG_context(self, query, formated_choices, score_threshold = None):
        ip = "localhost"
        port = 8080
        endpoint = "search"
        response = requests.get(f"http://{ip}:{port}/{endpoint}", params = {"q": query, "k": self.topK_searchEngine})
        if response.status_code == 200:


            # 1. BM25
            # text = response.text
            # doc_list = text.split("###RAG_DOC###")
            # for result in results:
            results = response.json() # [{docId, docNo, score, content}, ...]
            doc_data_list = []
            for i in range(len(results)):
                result = results[i]
                data = {"docId": result["docNo"], "BM25_score": result["score"], "BM25_ranking": i + 1, "content": result["content"]}
                doc_data_list.append(data)
            
            # print(f"1 len(doc_list): {len(doc_list)}")

            # 2. SPLADE
            if self.topK_SPLADE > 0:
                doc_data_list = self.RAG_SPLADE_filter(query, doc_data_list)
            # print(f"2 len(doc_list): {len(doc_list)}")
            

            # 3. MonoT5
            if self.topK_crossEncoder > 0:
                # doc_data_list = self.RAG_CrossEncoder_rerank(query + '\n' + formated_choices, doc_data_list)
                doc_data_list = self.RAG_MonoT5_rerank(query + '\n' + formated_choices, doc_data_list, score_threshold)
                # print(f"3 len(doc_list): {len(doc_list)}")
    
                # TODO: this is for test, only feed the nth retrieved document into the model
                # doc_list = [doc_list[4]]
                

            # 4. LLM list reranker
            if self.topK_LLM > 0:
                doc_data_list = self.RAG_LLM_rerank(query + '\n' + formated_choices, doc_data_list)

            # doc_list -> context (str)
            context = ""
            for doc_data in doc_data_list:
                if self.check_RAG_doc_useful(query, formated_choices, doc_data):
                    context += doc_data["content"]
                    context += "\n\n"
                
            return context
        else:
            return f"HTTPError: {response.status_code} - {response.text}"

    def RAG_SPLADE_filter(self, query, doc_data_list):
        # SPLADE
        doc_list = [data["content"] for data in doc_data_list]
        query_embeddings = self.splade_model.encode_query([query], show_progress_bar=False)
        document_embeddings = self.splade_model.encode_document(doc_list, show_progress_bar=False)
        similarities = self.splade_model.similarity(query_embeddings, document_embeddings)
        score_list = similarities[0].tolist()

        for i in range(len(doc_data_list)):
            doc_data_list[i]["SPLADE_score"] = score_list[i]
        
        doc_score_list = list(zip(doc_data_list, score_list))
        top_k = self.topK_SPLADE
        top_doc_score_list = sorted(doc_score_list, key = lambda x: x[1], reverse = True)[:top_k]
        top_doc_list = [doc_score[0] for doc_score in top_doc_score_list]
        # print("top_doc_list", len(top_doc_list), top_doc_list)
        
        for i in range(len(top_doc_list)):
            top_doc_list[i]["SPLADE_ranking"] = i + 1

        return top_doc_list

    def RAG_CrossEncoder_rerank(self, query, doc_data_list, score_threshold = None):
        doc_list = [data["content"] for data in doc_data_list]
        pair_list = [(query, doc) for doc in doc_list]
        # print("RAG_CrossEncoder_rerank pair_list", pair_list)
        
        # score_list = self.crossEncoder_model.predict(pair_list, show_progress_bar=False) # For CrossEncoder
        score_list = self.crossEncoder_model.compute_score(pair_list) # For FlagReranker

        doc_score_list = list(zip(doc_data_list, score_list))
        doc_score_list = sorted(doc_score_list, key = lambda x: x[1], reverse = True)
        doc_score_list = doc_score_list[:self.topK_crossEncoder]
        # print("reranked top k score:", [doc_score[1] for doc_score in top_doc_score_list])

        reranked_doc_data_list = []
        for i in range(len(doc_score_list)):
            doc_score = doc_score_list[i]
            if score_threshold == None or doc_score[1] > score_threshold:
                doc_data = doc_score[0]
                doc_data["crossEncoder_score"] = doc_score[1]
                doc_data["crossEncoder_ranking"] = i + 1
                reranked_doc_data_list.append(doc_data)
        
        return reranked_doc_data_list

    def RAG_MonoT5_rerank(self, query, doc_data_list, score_threshold = None):
        doc_list = [data["content"] for data in doc_data_list]
        candidates = [Candidate(docid = i, score = 0, doc = {"segment": doc_list[i]}) for i in range(len(doc_list))]
        request = Request(query = Query(text = query, qid = 0), candidates = candidates)
        rerank_results = self.crossEncoder_model.rerank(request, logging = False)

        reranked_doc_data_list = []
        count = 0
        for candidate in rerank_results.candidates:
            i = candidate.docid
            doc_data_list[i]["MonoT5_score"] = candidate.score
            reranked_doc_data_list.append(doc_data_list[i])

            count += 1
            if count >= self.topK_crossEncoder:
                break
        
        # reranked_doc_list = [candidate.doc["segment"] for candidate in rerank_results.candidates]
        #scores = [candidate.score for candidate in rerank_results.candidates]
        # doc_list = reranked_doc_list[:self.topK_crossEncoder]

        # score_list = [candidate.score for candidate in rerank_results.candidates][:self.topK_crossEncoder]
        score_list = [data["MonoT5_score"] for data in reranked_doc_data_list]

        # Filter by threshold
        if score_threshold != None:
            threshold_filtered_count = 0
            for score in score_list:
                if score >= score_threshold:
                    threshold_filtered_count += 1
                else:
                    break
            reranked_doc_data_list = reranked_doc_data_list[:threshold_filtered_count]
            
        for i in range(len(reranked_doc_data_list)):
            reranked_doc_data_list[i]["MonoT5_ranking"] = i + 1
            
        return reranked_doc_data_list

    def RAG_LLM_rerank(self, query, doc_data_list):
        doc_list = [data["content"] for data in doc_data_list]
        candidates = [Candidate(docid = i, score = 0, doc = {"segment": doc_list[i]}) for i in range(len(doc_list))]
        request = Request(query = Query(text = query, qid = 0), candidates = candidates)
        rerank_results = self.llm_reranker.rerank(request, logging = False)
        if isinstance(rerank_results, list):
            rerank_results = rerank_results[0]

        reranked_doc_data_list = []
        count = 0
        for candidate in rerank_results.candidates:
            i = candidate.docid
            doc_data = doc_data_list[i]
            doc_data["LLM_ranking"] = count + 1
            reranked_doc_data_list.append(doc_data)

            count += 1
            if count >= self.topK_LLM:
                break
            
        # reranked_doc_list = [candidate.doc["segment"] for candidate in rerank_results.candidates]
        # #scores = [candidate.score for candidate in rerank_results.candidates]
        # reranked_doc_data_list = reranked_doc_list[:self.topK_LLM]
        
        return reranked_doc_data_list
    
    def check_RAG_doc_useful(self, question, formated_choices, doc_data):
        return True
        doc = doc_data["content"]
        model_prompt = f"Question: {question} \n Choices: {formated_choices} \n Context: {doc} \n Is the context useful for answering the question? \n [A] Yes \n [B] No"
        answer = self.run_inference_get_answer_letter(model_prompt, self.temperature, do_sample = False)
        print("check RAG useful", answer)
        if answer == "A":
            return True
        else:
            return False

    
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
            messages = [{"role": "user", "content": f"{content}"}]
            inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
            # print("inputs:", inputs, inputs.shape)
            with torch.no_grad():
                outputs = self.model.generate(inputs, **generate_kwargs)
                # print("outputs:", outputs.shape)
        

        # print("temperature:", temperature)
        # print("inputs", inputs)
        
        text = self.tokenizer.batch_decode(outputs)[0]
        # print("outputs text:", text)
        if not self.is_encoder_decoder:
            text = text.split("<|assistant|>")[-1]
        # answer = tokenizer.decode(output[0], skip_special_tokens = True)

        answer = self.extract_answer(text).strip("()")
        
        # print(f"answer: {answer}")
        # logging.info(f"answer: {answer}")
        
        return answer

    def extract_answer(self, response):
        matchFirst = re.search(r'the answer is .(\w).', response)
        if matchFirst:
            return f"({matchFirst.group(1)})"
        match = self.find_match(self.regex, response) 
        if match:
            return f"({match})"
        return "[invalid]"

    def format_choices(self, choices):
        a = zip(list(choices.keys()), choices.values())
        final_answers = []
        for x,y in a:
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
                      topK_searchEngine = None, topK_SPLADE = None, topK_crossEncoder = None, topK_LLM = None, score_threshold = None):
        if topK_searchEngine != None:
            self.topK_searchEngine = topK_searchEngine
        if topK_SPLADE != None:
            self.topK_SPLADE = topK_SPLADE
        if topK_crossEncoder != None:
            self.topK_crossEncoder = topK_crossEncoder
        if topK_LLM != None:
            self.topK_LLM = topK_LLM
        
        # Remove all handlers associated with the root logger
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
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
        
        print(f"topK_searchEngine: {self.topK_searchEngine}")
        logging.info(f"topK_searchEngine: {self.topK_searchEngine}")
        print(f"topK_SPLADE: {self.topK_SPLADE}")
        logging.info(f"topK_SPLADE: {self.topK_SPLADE}")
        print(f"topK_crossEncoder: {self.topK_crossEncoder}")
        logging.info(f"topK_crossEncoder: {self.topK_crossEncoder}")
        print(f"topK_LLM: {self.topK_LLM}")
        logging.info(f"topK_LLM: {self.topK_LLM}")

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

            formated_choices = self.format_choices(choices)
            
            # content = prompt.format(question = question, choices = formated_choices)
            
            if use_RAG:
                context = self.get_RAG_context(question, formated_choices, score_threshold = score_threshold)
                content = prompt.format(context = context, question = question, choices = formated_choices)

                # The following 4 lines of code is for logging the scores of documents
                # count += 1
                # print(f"question {count}/{len(data_list)} score_list: {score_list}")
                # logging.info(f"question {count}/{len(data_list)} score_list: {score_list}")
                # continue
                
            else:
                content = prompt.format(question = question, choices = formated_choices)
            
            # print("messages: ", messages)
            # break
            
            # print("correct answer: ", answer_idx)

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
        
        
            if is_correct:
                correct_count += 1
        
            count += 1

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
