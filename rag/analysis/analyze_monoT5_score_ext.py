import re
import ast
import numpy as np

path = "rag/analysis/logs/"
file_path_accuracy_no_rag = path + "MedQA_810_noRAG.txt"
# file_paths_accuracy_rag = ["8-19-150_30_1.txt", "8-19-150_30_2.txt", "8-18-150_30_3.txt", "8-20-150_30_4.txt", "8-20-150_30_5.txt"]
file_paths_accuracy_rag = ["8-28-150_30_1_short.txt", "8-30-150_30_2_A100.txt", "8-30-150_30_3_short.txt", "8-20-150_30_4.txt", "8-20-150_30_5.txt"]
file_paths_accuracy_rag = [path + file_name for file_name in file_paths_accuracy_rag]
# file_path_score = path + "8-24-150_30_1_score.txt"
file_path_score = path + "8-25-150_30_5_score.txt"

# score
score_list = []
with open(file_path_score, "r") as f:
    for line in f:
        if "question " in line:
            score_list.append(ast.literal_eval(line.split("score_list: ")[-1]))

# print(len(score_list), score_list[-1])

# each question answer
no_rag_results = []
with open(file_path_accuracy_no_rag, "r") as f:
    for line in f:
        if "question " in line:
            # print(line.split(" ")[-1].)
            no_rag_results.append(True if line.split(" ")[-1].strip() == "True" else False)

rag_results_lists = []
for file_path in file_paths_accuracy_rag:
    rag_results = []
    with open(file_path, "r") as f:
        for line in f:
            if "question " in line:
                # print(line.split(" ")[-1].)
                rag_results.append(True if line.split(" ")[-1].strip() == "True" else False)
    rag_results_lists.append(rag_results)

# Test if setting a threshold to MonoT5 score, what's the new accuracy
def get_accuracy(threshold, max_doc_count):
    correct_counter = 0
    for i in range(len(no_rag_results)):
        doc_count = 0
        for j in range(max_doc_count):
            if score_list[i][j] < threshold:
                break
            doc_count += 1

        is_correct = False
        if doc_count > 0:
            is_correct = rag_results_lists[doc_count - 1][i]
            if is_correct == True:
                correct_counter += 1
        else:
            is_correct = no_rag_results[i]
            if is_correct == True:
                correct_counter += 1
        
        # if i < 1000:
        #     print(f"{i + 1}: ", is_correct)

    accuracy = float(correct_counter) / len(no_rag_results)
    return accuracy

def find_best_threshold(max_doc_count):
    threshold_list = np.linspace(0.0, 1.0, 101).tolist()

    # threshold = 0.8
    max_accuracy = 0
    max_threshold = 0
    for threshold in threshold_list:
        accuracy = get_accuracy(threshold, max_doc_count)
        print(threshold, "->", accuracy)

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            max_threshold = threshold

    print("max: ", max_threshold, "->", max_accuracy)

find_best_threshold(max_doc_count = 3)

# print(get_accuracy(threshold = 0.8, max_doc_count = 3))

