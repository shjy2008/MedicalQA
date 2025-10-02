
import torch
from sentence_transformers import CrossEncoder
from FlagEmbedding import FlagReranker
from sentence_transformers import SparseEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch.nn.functional as F
import requests

# RankLLM
from rank_llm.data import Query, Candidate, Request
from rank_llm.rerank import Reranker
from rank_llm.rerank.pairwise.duot5 import DuoT5
from rank_llm.rerank.pointwise.monot5 import MonoT5
from rank_llm.rerank.listwise import ZephyrReranker, VicunaReranker, RankListwiseOSLLM


class ContextRetriever:

    def __init__(self):
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

        # model_coordinator = RankListwiseOSLLM(
        #     model="Qwen/Qwen2.5-7B-Instruct",
        # )
        # self.llm_reranker = Reranker(model_coordinator)

        
        # Step 4: Use a classifier model to determine which context+question can produce correct answer
        classifier_model_name = "/projects/sciences/computing/sheju347/RAG/classifier/t5-large-epoch-10/checkpoint-94290"
        self.classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_model_name)
        self.classifier_model = AutoModelForSeq2SeqLM.from_pretrained(classifier_model_name).to(self.device)

        # RAG
        self.topK_searchEngine = 100
        self.topK_SPLADE = 10
        self.topK_crossEncoder = 3
        self.topK_LLM = 3
    
    def set_params(self, topK_searchEngine, topK_SPLADE, topK_crossEncoder, topK_LLM):
        self.topK_searchEngine = topK_searchEngine
        self.topK_SPLADE = topK_SPLADE
        self.topK_crossEncoder = topK_crossEncoder
        self.topK_LLM = topK_LLM
        
    def get_RAG_context(self, query, formated_choices, score_threshold = None, pick_rag_index = None, use_classifier = False):
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

            # 4. LLM list reranker
            if self.topK_LLM > 0:
                doc_data_list = self.RAG_LLM_rerank(query + '\n' + formated_choices, doc_data_list)
            
            # only feed the nth retrieved document into the model
            if pick_rag_index != None:
                doc_data_list = [doc_data_list[pick_rag_index]]

            # Use a classifier model to select which context+query can produce correct answer
            if use_classifier:
                doc_data_list = self.RAG_classifier(query + '\n' + formated_choices, doc_data_list)
                
            # doc_list -> context (str)
            context = ""
            for doc_data in doc_data_list:
                if self.check_RAG_doc_useful(query, formated_choices, doc_data):
                    context += doc_data["content"]
                    context += "\n\n"
                
            # Print RAG scores and rankings
            for doc_data in doc_data_list:
                del doc_data["content"]
            print(f"RAG data: {doc_data_list}")
            logging.info(f"RAG data: {doc_data_list}")
            
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

    
