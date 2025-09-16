import re
import ast
import numpy as np

QUESTION_COUNT = 1273

path = "rag/analysis/logs/"
file_names = ["9-5-150_30_1_RAG_data_H100.txt", "9-5-150_30_2nd_RAG_data_H100.txt", "9-5-150_30_3rd_RAG_data_H100.txt", 
              "9-5-150_30_4th_RAG_data_H100.txt", "9-5-150_30_5th_RAG_data_H100.txt"]
# file_names = ["9-5-150_30_1_RAG_data_H100.txt"]
full_file_names = [path + file_name for file_name in file_names]

no_rag_file_name = path + "MedQA_810_noRAG.txt"

class QuestionAnswerData:
    def __init__(self, answer, correct_answer, RAG_data):
        self.answer = answer
        self.correct_answer = correct_answer
        self.is_correct = answer == correct_answer
        # RAG data: [{'docId': 'pubmed23n0827_1593', 'BM25_score': 95.4198, 'BM25_ranking': 44, 'SPLADE_score': 54.33488464355469, 'SPLADE_ranking': 13, 'MonoT5_score': 0.9597430488962883, 'MonoT5_ranking': 1}]
        self.RAG_data = RAG_data

# Get the data

# [ [QuestionAnswerData, QuestionAnswerData, ...], [...], ...]
all_question_answer_data_list = []

for i in range(len(full_file_names)):
    question_answer_data_list = []
    with open(full_file_names[i], "r") as f:
        count = 0
        curr_RAG_data = None
        for line in f:
            if "RAG data: " in line:
                curr_RAG_data_str = line[line.find("RAG data: ") + len("RAG data: "):].strip()
                curr_RAG_data = ast.literal_eval(curr_RAG_data_str)

            elif "question " in line:
                
                # Check question number error
                count += 1
                if "num:" in line:
                    number = line.strip().split('num:')[-1].split(' ')[0]
                else:
                    number = line.strip().split('/')[0].split(' ')[-1]
                if int(number) != count:
                    print(f"Error: {file_names[i]}:", number, count)
                    break

                match = re.search(r"answer:([^\s]+)\s+correct_answer:([^\s]+)", line)
                if match:
                    answer = match.group(1)
                    correct_answer = match.group(2)

                    question_answer_data = QuestionAnswerData(answer, correct_answer, curr_RAG_data)
                    question_answer_data_list.append(question_answer_data)
    
    all_question_answer_data_list.append(question_answer_data_list)


no_rag_results = []
with open(no_rag_file_name, "r") as f:
    count = 0
    for line in f:
        if "question " in line:
            
            # Check question number error
            count += 1
            if "num:" in line:
                number = line.strip().split('num:')[-1].split(' ')[0]
            else:
                number = line.strip().split('/')[0].split(' ')[-1]
            if int(number) != count:
                print(f"Error: {file_names[i]}:", number, count)
                break

            match = re.search(r"answer:([^\s]+)\s+correct_answer:([^\s]+)", line)
            if match:
                answer = match.group(1)
                correct_answer = match.group(2)
                no_rag_results.append(answer == correct_answer)


# Analyze the data

eps = 1e-9
def get_min_max_score(score_list):
    score_list = np.array(score_list, float)
    return (score_list - score_list.min()) / (score_list.max() - score_list.min() + eps)


# TODO:
# 1. find some patterns of using RAG result in True and False (print average BM25/SPLADE/monoT5 ranking and scores of True and False)
# 2. find some patterns of no-rag True and rag False, void them. Find some patterns of no-rag False and rag True, maximize them (add threshold)

total_BM25_score = 0
total_BM25_ranking = 0
total_SPLADE_score = 0
total_SPLADE_ranking = 0
total_MonoT5_score = 0
total_MonoT5_ranking = 0
counter = 0
for i in range(QUESTION_COUNT):
    for j in range(len(all_question_answer_data_list)):
        question_answer_data = all_question_answer_data_list[j][i]
        if no_rag_results[i] == True and question_answer_data.is_correct == False:
            total_BM25_score += question_answer_data.RAG_data[0]["BM25_score"]
            total_BM25_ranking += question_answer_data.RAG_data[0]["BM25_ranking"]
            total_SPLADE_score += question_answer_data.RAG_data[0]["SPLADE_score"]
            total_SPLADE_ranking += question_answer_data.RAG_data[0]["SPLADE_ranking"]
            total_MonoT5_score += question_answer_data.RAG_data[0]["MonoT5_score"]
            total_MonoT5_ranking += question_answer_data.RAG_data[0]["MonoT5_ranking"]
            counter += 1
            # print(i, j, question_answer_data.RAG_data)
    
    # if counter >= 2:
    #     break

print(f"Avg BM25 score: {total_BM25_score / counter}")
print(f"Avg BM25 ranking: {total_BM25_ranking / counter}")
print(f"Avg SPLADE score: {total_SPLADE_score / counter}")
print(f"Avg SPLADE ranking: {total_SPLADE_ranking / counter}")
print(f"Avg MonoT5 score: {total_MonoT5_score / counter}")
print(f"Avg MonoT5 ranking: {total_MonoT5_ranking / counter}")
print(counter)

# True
# Avg BM25 score: 97.75085817745826
# Avg BM25 ranking: 49.03333333333333
# Avg SPLADE score: 76.44035769080658
# Avg SPLADE ranking: 12.832613908872903
# Avg MonoT5 score: 0.5796811598613475
# Avg MonoT5 ranking: 3.000719424460432

# False
# Avg BM25 score: 102.3384445102504
# Avg BM25 ranking: 50.68564920273349
# Avg SPLADE score: 75.95030149664042
# Avg SPLADE ranking: 13.17630979498861
# Avg MonoT5 score: 0.5641854570388709
# Avg MonoT5 ranking: 2.998633257403189

BM25_weight = 0.01
SPLADE_weight = 0.01
MonoT5_weight = 1

def get_accuracy(bm25_weight, splade_weight, monoT5_weight):
    correct_counter = 0
    for i in range(QUESTION_COUNT):
        is_correct = False
        question_answer_data_of_all_pipelines = []
        for j in range(len(all_question_answer_data_list)):
            question_answer_data = all_question_answer_data_list[j][i]
            question_answer_data_of_all_pipelines.append(question_answer_data)
        
        chosen_data = None

        # bm25 = get_min_max_score([question_answer_data.RAG_data[0]["BM25_score"] for question_answer_data in question_answer_data_of_all_pipelines])
        # splade = get_min_max_score([question_answer_data.RAG_data[0]["SPLADE_score"] for question_answer_data in question_answer_data_of_all_pipelines])
        # monoT5 = get_min_max_score([question_answer_data.RAG_data[0]["MonoT5_score"] for question_answer_data in question_answer_data_of_all_pipelines])

        bm25 = np.array([question_answer_data.RAG_data[0]["BM25_score"] * 0.01 for question_answer_data in question_answer_data_of_all_pipelines])
        splade = np.array([question_answer_data.RAG_data[0]["SPLADE_score"] * 0.01 for question_answer_data in question_answer_data_of_all_pipelines])
        monoT5 = np.array([question_answer_data.RAG_data[0]["MonoT5_score"] for question_answer_data in question_answer_data_of_all_pipelines])

        # print(bm25, splade, monoT5)
        final_scores = bm25 * bm25_weight + splade * splade_weight + monoT5 * monoT5_weight
        # print(final_scores)
        best_score = np.max(final_scores)
        best_index = int(np.argmax(final_scores))
        chosen_data = question_answer_data_of_all_pipelines[best_index]

        # best_score = None
        # for question_answer_data in question_answer_data_of_all_pipelines:
        #     RAG_data_0 = question_answer_data.RAG_data[0]
        #     score = RAG_data_0["BM25_score"] * BM25_weight + RAG_data_0["SPLADE_score"] * SPLADE_weight + RAG_data_0["MonoT5_score"] * MonoT5_weight
        #     if best_score == None or score > best_score:
        #         best_score = score
        #         chosen_data = question_answer_data

        # best_MonoT5_ranking = None
        # for question_answer_data in question_answer_data_of_all_pipelines:
        #     if best_MonoT5_ranking == None or question_answer_data.RAG_data[0]["BM25_ranking"] < best_MonoT5_ranking:
        #         best_MonoT5_ranking = question_answer_data.RAG_data[0]["BM25_ranking"]
        #         chosen_data = question_answer_data
        
        is_correct = chosen_data.is_correct

        # if best_score < 1.4:
        #     is_correct = no_rag_results[i]
        
        if is_correct:
            correct_counter += 1

    accuracy = correct_counter / QUESTION_COUNT
    return accuracy


# best_accuracy = None
# best_params = None
# for bm25_weight in np.linspace(0.0, 1.0, 11):
#     for splade_weight in np.linspace(0.0, 1.0, 11):
#         for monoT5_weight in np.linspace(0.0, 1.0, 11):
#             accuracy = get_accuracy(bm25_weight, splade_weight, monoT5_weight)
#             params = [bm25_weight, splade_weight, monoT5_weight]
#             print(f"accuracy: {accuracy}, params: {params}")
#             if best_accuracy == None or accuracy > best_accuracy:
#                 best_accuracy = accuracy
#                 best_params = params

# print("all done")
# print(f"best accuracy: {best_accuracy}, params: {best_params}")

# print("Accuracy:", get_accuracy(0.2, 0.7, 0.6))






def get_accuracy_ranking_vote():
    correct_counter = 0
    for i in range(QUESTION_COUNT):
        is_correct = False
        question_answer_data_of_all_pipelines = []
        for j in range(len(all_question_answer_data_list)):
            question_answer_data = all_question_answer_data_list[j][i]
            question_answer_data_of_all_pipelines.append(question_answer_data)
        
        chosen_data = None

        

        # # bm25 = get_min_max_score([question_answer_data.RAG_data[0]["BM25_score"] for question_answer_data in question_answer_data_of_all_pipelines])
        # # splade = get_min_max_score([question_answer_data.RAG_data[0]["SPLADE_score"] for question_answer_data in question_answer_data_of_all_pipelines])
        # # monoT5 = get_min_max_score([question_answer_data.RAG_data[0]["MonoT5_score"] for question_answer_data in question_answer_data_of_all_pipelines])

        # bm25 = np.array([question_answer_data.RAG_data[0]["BM25_score"] * 0.01 for question_answer_data in question_answer_data_of_all_pipelines])
        # splade = np.array([question_answer_data.RAG_data[0]["SPLADE_score"] * 0.01 for question_answer_data in question_answer_data_of_all_pipelines])
        # monoT5 = np.array([question_answer_data.RAG_data[0]["MonoT5_score"] for question_answer_data in question_answer_data_of_all_pipelines])

        # # print(bm25, splade, monoT5)
        # final_scores = bm25 * bm25_weight + splade * splade_weight + monoT5 * monoT5_weight
        # # print(final_scores)
        # best_score = np.max(final_scores)
        # best_index = int(np.argmax(final_scores))
        # chosen_data = question_answer_data_of_all_pipelines[best_index]

        # best_score = None
        # for question_answer_data in question_answer_data_of_all_pipelines:
        #     RAG_data_0 = question_answer_data.RAG_data[0]
        #     score = RAG_data_0["BM25_score"] * BM25_weight + RAG_data_0["SPLADE_score"] * SPLADE_weight + RAG_data_0["MonoT5_score"] * MonoT5_weight
        #     if best_score == None or score > best_score:
        #         best_score = score
        #         chosen_data = question_answer_data

        # best_MonoT5_ranking = None
        # for question_answer_data in question_answer_data_of_all_pipelines:
        #     if best_MonoT5_ranking == None or question_answer_data.RAG_data[0]["BM25_ranking"] < best_MonoT5_ranking:
        #         best_MonoT5_ranking = question_answer_data.RAG_data[0]["BM25_ranking"]
        #         chosen_data = question_answer_data
        
        is_correct = chosen_data.is_correct

        # if best_score < 1.4:
        #     is_correct = no_rag_results[i]
        
        if is_correct:
            correct_counter += 1

    accuracy = correct_counter / QUESTION_COUNT
    return accuracy

