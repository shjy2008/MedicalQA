import re
import ast
import numpy as np

path = "rag/analysis/logs/"
file_path_accuracy_no_rag = path + "MedQA_810_noRAG.txt"
file_path_accuracy_rag = path + "9-5-150_30_5th_RAG_data_H100.txt" #"8-19-150_30_1.txt"
# file_path_score = path + "8-24-150_30_1_score.txt"
file_path_score = path + "8-25-150_30_5_score.txt"

# score
score_list = []
with open(file_path_score, "r") as f:
    for line in f:
        if "question " in line:
            score_list.append(ast.literal_eval(line.split("score_list: ")[-1]))

print(len(score_list))

# each question answer
no_rag_results = []
with open(file_path_accuracy_no_rag, "r") as f:
    for line in f:
        if "question " in line:
            # print(line.split(" ")[-1].)
            no_rag_results.append(True if line.split(" ")[-1].strip() == "True" else False)

rag_results = []
with open(file_path_accuracy_rag, "r") as f:
    for line in f:
        if "question " in line:
            # print(line.split(" ")[-1].)
            rag_results.append(True if line.split(" ")[-1].strip() == "True" else False)

# Test the MonoT5 score difference between wrong-to-right and right-to-wrong
wrong_to_right_indices = []
right_to_wrong_indices = []
for i in range(len(rag_results)):
    if no_rag_results[i] == False and rag_results[i] == True:
        wrong_to_right_indices.append(i)
    elif no_rag_results[i] == True and rag_results[i] == False:
        right_to_wrong_indices.append(i)

avg_score_wrong_to_right = sum(score_list[i][0] for i in wrong_to_right_indices) / len(wrong_to_right_indices)
print(avg_score_wrong_to_right) # 0.7253542855400607

avg_score_right_to_wrong = sum(score_list[i][0] for i in right_to_wrong_indices) / len(right_to_wrong_indices)
print(avg_score_right_to_wrong) # 0.6942496980507334

# Test if setting a threshold to MonoT5 score, what's the new accuracy
threshold_list = np.linspace(0.0, 1.0, 1001).tolist()
# threshold = 0.8
max_accuracy = 0
max_threshold = 0
for threshold in threshold_list:
    correct_counter = 0
    for i in range(len(rag_results)):
        if score_list[i][0] > threshold:
            if rag_results[i] == True:
                correct_counter += 1
        else:
            if no_rag_results[i] == True:
                correct_counter += 1

    new_accuracy = float(correct_counter) / len(rag_results)

    print(threshold, "->", new_accuracy)

    if new_accuracy > max_accuracy:
        max_accuracy = new_accuracy
        max_threshold = threshold

print("max: ", max_threshold, "->", max_accuracy)



