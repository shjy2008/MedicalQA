import torch
from sentence_transformers import CrossEncoder
from FlagEmbedding import FlagReranker
from sentence_transformers import SparseEncoder
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch.nn.functional as F
import requests
import logging

# RankLLM
from rank_llm.data import Query, Candidate, Request
from rank_llm.rerank import Reranker
from rank_llm.rerank.pairwise.duot5 import DuoT5
from rank_llm.rerank.pointwise.monot5 import MonoT5
from rank_llm.rerank.listwise import ZephyrReranker, VicunaReranker, RankListwiseOSLLM


class ContextRetriever:

    def __init__(self, device = None):
        self.device = device
        if self.device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # RAG
        self.topK_searchEngine = 100
        self.topK_SPLADE = 10
        self.topK_denseEmbedding = 0
        self.topK_crossEncoder = 3
        self.topK_LLM = 3
        self.score_threshold = None
        self.pick_rag_index = None
        self.use_classifier = False
        self.RRF_models = [] # Reciprocal Rank Fusion model list ["SPLADE", "DenseEmbedding", "MonoT5", "Zephyr", "Vicuna", "Qwen", "FirstMistral"]

        # models
        self.splade_model = None
        self.denseEmbedding_model = None
        self.crossEncoder_model = None
        self.llm_reranker = None
        self.zephyr_reranker = None
        self.vicuna_reranker = None
    
    def set_params(self, topK_searchEngine, topK_SPLADE, topK_denseEmbedding, topK_crossEncoder, topK_LLM, score_threshold = None, pick_rag_index = None, use_classifier = False, RRF_models = None):
        self.topK_searchEngine = topK_searchEngine
        self.topK_SPLADE = topK_SPLADE
        self.topK_denseEmbedding = topK_denseEmbedding
        self.topK_crossEncoder = topK_crossEncoder
        self.topK_LLM = topK_LLM
        self.score_threshold = score_threshold
        self.pick_rag_index = pick_rag_index
        self.use_classifier = use_classifier
        self.RRF_models = RRF_models

        # Step 0: BM25 search engine

        # -------- Initialize the models --------

        RRF_model_names = [] if self.RRF_models == None else [RRF_model[0] for RRF_model in self.RRF_models]
        
        # Step 1: SPLADE model (lexical sparse retrieval)
        if (self.topK_SPLADE != None and self.topK_SPLADE > 0) or ("SPLADE" in RRF_model_names):
            if self.splade_model == None:
                self.splade_model = SparseEncoder("naver/splade-v3")
                # self.splade_model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

        # Step 2: Dense Embedding Model
        if (self.topK_denseEmbedding != None and self.topK_denseEmbedding > 0) or ("DenseEmbedding" in RRF_model_names):
            if self.denseEmbedding_model == None:
                self.denseEmbedding_model = SentenceTransformer("sentence-transformers/embeddinggemma-300m-medical")

        # Step 3: Cross-encoder model (e.g. MonoBERT)
        if (self.topK_crossEncoder != None and self.topK_crossEncoder > 0) or ("MonoT5" in RRF_model_names):
            if self.crossEncoder_model == None:
                self.crossEncoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
                # self.crossEncoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L12-v2")
                # self.crossEncoder_model = CrossEncoder("cross-encoder/ms-marco-electra-base")
                # self.crossEncoder_model = FlagReranker('BAAI/bge-reranker-v2-m3')
                # self.crossEncoder_model = FlagReranker('BAAI/bge-reranker-v2-gemma')
                # self.crossEncoder_model = Reranker(MonoT5("castorini/monot5-3b-med-msmarco", context_size = 4096, batch_size = 16))


        # Step 4: More powerful model (DuoBERT, LLM, ...)
        if self.topK_LLM != None and self.topK_LLM > 0:
            if self.llm_reranker == None:
                # self.llm_reranker = Reranker(DuoT5(model = "castorini/duot5-3b-med-msmarco"))
                # self.llm_reranker = ZephyrReranker()
                self.llm_reranker = VicunaReranker()
                
                # model_coordinator = RankListwiseOSLLM(
                #     model="castorini/first_mistral",
                #     use_logits=True,
                #     use_alpha=True,
                # )
                # self.llm_reranker = Reranker(model_coordinator)
        
                # model_coordinator = RankListwiseOSLLM(
                #     model="Qwen/Qwen2.5-7B-Instruct",
                # )
                # self.llm_reranker = Reranker(model_coordinator)

        if "Zephyr" in RRF_model_names:
            if self.zephyr_reranker == None:
                self.zephyr_reranker = ZephyrReranker()
            
        if "Vicuna" in RRF_model_names:
            if self.vicuna_reranker == None:
                self.vicuna_reranker = VicunaReranker()
        
        # Step 4: Use a classifier model to determine which context+question can produce correct answer
        if use_classifier:
            pass
            # classifier_model_name = "/projects/sciences/computing/sheju347/RAG/classifier/t5-large-epoch-10/checkpoint-94290"
            # self.classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_model_name)
            # self.classifier_model = AutoModelForSeq2SeqLM.from_pretrained(classifier_model_name).to(self.device)

    def get_RAG_data_list(self, question, formated_choices):
        ip = "localhost"
        port = 8080
        endpoint = "search"
        response = requests.get(f"http://{ip}:{port}/{endpoint}", params = {"q": question, "k": self.topK_searchEngine})
        if response.status_code != 200:
            print (f"HTTPError: {response.status_code} - {response.text}")
            return []
        
        # 0. BM25
        # text = response.text
        # doc_list = text.split("###RAG_DOC###")
        # for result in results:
        results = response.json() # [{docId, docNo, score, content}, ...]
        doc_data_list = []
        for i in range(len(results)):
            result = results[i]
            data = {"docId": result["docNo"], "BM25_score": result["score"], "BM25_ranking": i + 1, "content": result["content"]}
            doc_data_list.append(data)

        if len(doc_data_list) == 0:
            return doc_data_list
        
        # print(f"1 len(doc_list): {len(doc_list)}")

        question_and_choices = question + '\n' + formated_choices

        # 1. SPLADE
        if self.topK_SPLADE != None and self.topK_SPLADE > 0:
            doc_data_list = self.RAG_SPLADE_filter(question, doc_data_list, self.topK_SPLADE)
        # print(f"2 len(doc_list): {len(doc_list)}")

        # 2. Dense Embedding
        if self.topK_denseEmbedding != None and self.topK_denseEmbedding > 0:
            doc_data_list = self.RAG_Dense_Embedding(question, doc_data_list, self.topK_denseEmbedding)

        # 3. MonoT5
        if self.topK_crossEncoder != None and self.topK_crossEncoder > 0:
            if isinstance(self.crossEncoder_model, Reranker):
                doc_data_list = self.RAG_MonoT5_rerank(question_and_choices, doc_data_list, self.score_threshold, self.topK_crossEncoder)
            else:
                doc_data_list = self.RAG_CrossEncoder_rerank(question_and_choices, doc_data_list, self.score_threshold, self.topK_crossEncoder)
            # print(f"3 len(doc_list): {len(doc_list)}")            

        # 4. LLM list reranker
        if self.topK_LLM != None and self.topK_LLM > 0:
            doc_data_list = self.RAG_LLM_rerank(self.llm_reranker, question_and_choices, doc_data_list, self.topK_LLM)

        # 5. RRF (Resiprocal Rank Fusion)
        if self.RRF_models != None and len(self.RRF_models) > 0:
            doc_data_list = self.RAG_run_RRF(question, formated_choices, doc_data_list)
            doc_data_list = [doc_data_list[0]] # TODO: for now pick the first one only
        
        # only feed the nth retrieved document into the model
        if self.pick_rag_index != None:
            doc_data_list = [doc_data_list[self.pick_rag_index]]

        # Use a classifier model to select which context+query can produce correct answer
        if self.use_classifier:
            doc_data_list = self.RAG_classifier(question_and_choices, doc_data_list)
            
            
        return doc_data_list
        
    def get_RAG_context(self, question, formated_choices):
        doc_data_list = self.get_RAG_data_list(question, formated_choices)
        
        # doc_list -> context (str)
        context = ""
        for doc_data in doc_data_list:
            if self.check_RAG_doc_useful(question, formated_choices, doc_data):
                context += doc_data["content"]
                context += "\n\n"

        # Print RAG scores and rankings
        for doc_data in doc_data_list:
            del doc_data["content"]
        print(f"RAG data: {doc_data_list}")
        logging.info(f"RAG data: {doc_data_list}")
        
        return context

    def RAG_run_RRF(self, question, formated_choices, doc_data_list):
        # print("doc_data_list:", len(doc_data_list), doc_data_list)
        question_and_choices = question + '\n' + formated_choices
        all_ranked_data_lists = []
        for RRF_model in self.RRF_models:
            model_name = RRF_model[0]
            ranked_data_list = None
            if model_name == "SPLADE":
                ranked_data_list = self.RAG_SPLADE_filter(question, doc_data_list)
            elif model_name == "DenseEmbedding":
                ranked_data_list = self.RAG_Dense_Embedding(question, doc_data_list)
            elif model_name == "MonoT5":
                ranked_data_list = self.RAG_MonoT5_rerank(question_and_choices, doc_data_list)
            elif model_name == "Zephyr":
                ranked_data_list = self.RAG_LLM_rerank(self.zephyr_reranker, question_and_choices, doc_data_list)
            elif model_name == "Vicuna":
                ranked_data_list = self.RAG_LLM_rerank(self.vicuna_reranker, question_and_choices, doc_data_list)
            # elif model_name == "Qwen":
            # elif model_name == "FirstMistral"

            if ranked_data_list:
                all_ranked_data_lists.append(ranked_data_list)

        # print("all_ranked_data_lists:", len(all_ranked_data_lists), all_ranked_data_lists)
        
        weights = [RRF_model[1] for RRF_model in self.RRF_models]
        
        docId_to_fused_scores = {}

        k = 60
        for weight, ranked_data_list in zip(weights, all_ranked_data_lists):
            for rank, doc_data in enumerate(ranked_data_list, start=1):
                docId = doc_data["docId"]
                if docId not in docId_to_fused_scores:
                    docId_to_fused_scores[docId] = 0
                docId_to_fused_scores[docId] += weight * (1 / (k + rank))

        # print("docId_to_fused_scores:", docId_to_fused_scores)
        
        # Sort by fused score (descending)
        sorted_docId_and_score = sorted(docId_to_fused_scores.items(), key=lambda x: x[1], reverse=True)
        docId_to_doc_data = {d["docId"]: d for d in doc_data_list}
        return [docId_to_doc_data[docId] for (docId, fused_score) in sorted_docId_and_score]
                

    def RAG_SPLADE_filter(self, query, doc_data_list, top_k = None):
        # SPLADE
        doc_list = [data["content"] for data in doc_data_list]
        query_embeddings = self.splade_model.encode_query([query], show_progress_bar=False)
        document_embeddings = self.splade_model.encode_document(doc_list, show_progress_bar=False)
        similarities = self.splade_model.similarity(query_embeddings, document_embeddings)
        score_list = similarities[0].tolist()

        for i in range(len(doc_data_list)):
            doc_data_list[i]["SPLADE_score"] = score_list[i]
        
        doc_score_list = list(zip(doc_data_list, score_list))
        top_doc_score_list = sorted(doc_score_list, key = lambda x: x[1], reverse = True)
        if top_k != None:
            top_doc_score_list = top_doc_score_list[:top_k]
        top_doc_list = [doc_score[0] for doc_score in top_doc_score_list]
        # print("top_doc_list", len(top_doc_list), top_doc_list)
        
        for i in range(len(top_doc_list)):
            top_doc_list[i]["SPLADE_ranking"] = i + 1

        return top_doc_list

    def RAG_Dense_Embedding(self, query, doc_data_list, top_k = None):
        doc_list = [data["content"] for data in doc_data_list]
        query_embeddings = self.denseEmbedding_model.encode_query([query], show_progress_bar=False)
        document_embeddings = self.denseEmbedding_model.encode_document(doc_list, show_progress_bar=False)
        similarities = self.denseEmbedding_model.similarity(query_embeddings, document_embeddings)
        score_list = similarities[0].tolist()

        for i in range(len(doc_data_list)):
            doc_data_list[i]["denseEmbedding_score"] = score_list[i]
        
        doc_score_list = list(zip(doc_data_list, score_list))
        top_doc_score_list = sorted(doc_score_list, key = lambda x: x[1], reverse = True)
        if top_k != None:
            top_doc_score_list = top_doc_score_list[:top_k]
        top_doc_list = [doc_score[0] for doc_score in top_doc_score_list]
        # print("top_doc_list", len(top_doc_list), top_doc_list)
        
        for i in range(len(top_doc_list)):
            top_doc_list[i]["denseEmbedding_ranking"] = i + 1

        return top_doc_list

    def RAG_CrossEncoder_rerank(self, query, doc_data_list, score_threshold = None, top_k = None):
        doc_list = [data["content"] for data in doc_data_list]
        pair_list = [(query, doc) for doc in doc_list]
        # print("RAG_CrossEncoder_rerank pair_list", pair_list)

        if isinstance(self.crossEncoder_model, CrossEncoder):
            score_list = self.crossEncoder_model.predict(pair_list, show_progress_bar=False) # For CrossEncoder
        else:
            score_list = self.crossEncoder_model.compute_score(pair_list) # For FlagReranker

        doc_score_list = list(zip(doc_data_list, score_list))
        doc_score_list = sorted(doc_score_list, key = lambda x: x[1], reverse = True)
        if top_k != None:
            doc_score_list = doc_score_list[:top_k]
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

    def RAG_MonoT5_rerank(self, query, doc_data_list, score_threshold = None, top_k = None):
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
            if top_k != None and count >= top_k:
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

    def RAG_LLM_rerank(self, llm_model, query, doc_data_list, top_k = None):
        doc_list = [data["content"] for data in doc_data_list]
        candidates = [Candidate(docid = i, score = 0, doc = {"segment": doc_list[i]}) for i in range(len(doc_list))]
        request = Request(query = Query(text = query, qid = 0), candidates = candidates)
        rerank_results = llm_model.rerank(request, logging = False)
        if isinstance(rerank_results, list):
            rerank_results = rerank_results[0]

        reranked_doc_data_list = []
        count = 0
        for candidate in rerank_results.candidates:
            i = candidate.docid
            doc_data = doc_data_list[i]
            doc_data["LLM_ranking"] = count + 1
            doc_data["LLM_score"] = candidate.score
            reranked_doc_data_list.append(doc_data)

            count += 1
            if top_k != None and count >= top_k:
                break
            
        # reranked_doc_list = [candidate.doc["segment"] for candidate in rerank_results.candidates]
        # #scores = [candidate.score for candidate in rerank_results.candidates]
        # if top_k != None:
        # reranked_doc_data_list = reranked_doc_list[:top_k]
        
        return reranked_doc_data_list

    def RAG_classifier(self, query, doc_data_list):
        tokenizer = self.classifier_tokenizer
        model = self.classifier_model

        ret_doc_data_list = []

        best_doc_data = None
        largest_confidence = 0
        
        for doc_data in doc_data_list:
            doc = doc_data["content"]
            input_text = f"question: {query} context: {doc}"
            
            encoding = tokenizer(
                input_text,
                truncation=True,
                padding="max_length",
                max_length=1024,
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            
            decoder_start_token_id = model.config.decoder_start_token_id
            
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=torch.tensor([[decoder_start_token_id]], device = self.device)
                )
                logits = outputs.logits
        
            first_token_logits = logits[0, 0]
            probs = F.softmax(first_token_logits, dim=-1)
        
            yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
            
            pred_ids = torch.argmax(logits, dim=-1)  # [batch, seq_len]
            first_pred_id = pred_ids[0, 0].item()

            is_yes = first_pred_id == yes_id
            confidence = probs[yes_id].item()
            
            log_str = f"docId: {doc_data['docId']} pred_ids: {first_pred_id} yes_id: {yes_id} Confidence yes: {confidence}, is_yes: {is_yes}"
            print(log_str)
            logging.info(log_str)

            if is_yes:
                if confidence > largest_confidence:
                    largest_confidence = confidence
                    best_doc_data = doc_data
                # break

        if best_doc_data != None:
            ret_doc_data_list.append(best_doc_data)
                
        return ret_doc_data_list
        
    
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

    
