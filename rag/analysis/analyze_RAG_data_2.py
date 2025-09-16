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

top_2_file_name = path + "8-30-150_30_2_A100.txt"
top_3_file_name = path + "8-30-150_30_3_short.txt"

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


# get local ranking of BM25/SPLADE/MonoT5
for i in range(QUESTION_COUNT):
    question_answer_data_list = [data[i] for data in all_question_answer_data_list]
    sorted_list = sorted(question_answer_data_list, key = lambda data: data.RAG_data[0]["BM25_ranking"])
    for j in range(len(sorted_list)):
        sorted_list[j].RAG_data[0]["BM25_local_ranking"] = j + 1

    sorted_list = sorted(question_answer_data_list, key = lambda data: data.RAG_data[0]["SPLADE_ranking"])
    for j in range(len(sorted_list)):
        sorted_list[j].RAG_data[0]["SPLADE_local_ranking"] = j + 1

    sorted_list = sorted(question_answer_data_list, key = lambda data: data.RAG_data[0]["MonoT5_ranking"])
    for j in range(len(sorted_list)):
        sorted_list[j].RAG_data[0]["MonoT5_local_ranking"] = j + 1
    

def read_log_data(log_file_name, ret_list):
    with open(log_file_name, "r") as f:
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
                    ret_list.append(answer == correct_answer)

# Get NO rag results
no_rag_results = []
read_log_data(no_rag_file_name, no_rag_results)

# top 2 rag results
top_2_results = []
read_log_data(top_2_file_name, top_2_results)

# top 3 rag results
top_3_results = []
read_log_data(top_3_file_name, top_3_results)    


# Analyze the data

# TODO:
# 1. find some patterns of using RAG result in True and False (print average BM25/SPLADE/monoT5 ranking and scores of True and False)
# 2. find some patterns of no-rag True and rag False, void them. Find some patterns of no-rag False and rag True, maximize them (add threshold)

# results show BM25/SPLADE scores are meaningless
total_BM25_score = 0
total_BM25_ranking = 0
total_SPLADE_score = 0
total_SPLADE_ranking = 0
total_MonoT5_score = 0
total_MonoT5_ranking = 0
counter = 0
for i in range(QUESTION_COUNT):
    if no_rag_results[i] == True:
        for j in range(len(all_question_answer_data_list)):
            question_answer_data = all_question_answer_data_list[j][i]
            if question_answer_data.is_correct == False:
                total_BM25_score += question_answer_data.RAG_data[0]["BM25_score"]
                total_BM25_ranking += question_answer_data.RAG_data[0]["BM25_local_ranking"]
                total_SPLADE_score += question_answer_data.RAG_data[0]["SPLADE_score"]
                total_SPLADE_ranking += question_answer_data.RAG_data[0]["SPLADE_local_ranking"]
                total_MonoT5_score += question_answer_data.RAG_data[0]["MonoT5_score"]
                total_MonoT5_ranking += question_answer_data.RAG_data[0]["MonoT5_local_ranking"]
                counter += 1

print(f"Avg BM25 score: {total_BM25_score / counter}")
print(f"Avg BM25 local ranking: {total_BM25_ranking / counter}")
print(f"Avg SPLADE score: {total_SPLADE_score / counter}")
print(f"Avg SPLADE local ranking: {total_SPLADE_ranking / counter}")
print(f"Avg MonoT5 score: {total_MonoT5_score / counter}")
print(f"Avg MonoT5 local ranking: {total_MonoT5_ranking / counter}")
print(counter)

def get_accuracy_ranking_vote():
    correct_counter = 0
    correct_counter_no_rag = 0
    correct_counter_rag = 0
    use_RAG_counter = 0
    for i in range(QUESTION_COUNT):
        is_correct = False
        question_answer_data_of_all_pipelines = []
        for j in range(len(all_question_answer_data_list)):
            question_answer_data = all_question_answer_data_list[j][i]
            question_answer_data_of_all_pipelines.append(question_answer_data)
        
        # chosen_data = None

        use_rag = False

        # total_ranking_threshold = 6
        # for question_answer_data in question_answer_data_of_all_pipelines:
        #     rag_data = question_answer_data.RAG_data[0]
        #     if rag_data["BM25_local_ranking"] + rag_data["SPLADE_local_ranking"] + rag_data["MonoT5_local_ranking"] * 2 <= total_ranking_threshold and \
        #         rag_data["MonoT5_score"] > 0.5:
        #         chosen_data = question_answer_data
        #         use_RAG_counter += 1
        #         break

        if question_answer_data_of_all_pipelines[0].RAG_data[0]["BM25_local_ranking"] <= 5 and \
            question_answer_data_of_all_pipelines[0].RAG_data[0]["SPLADE_local_ranking"] <= 5 and \
            question_answer_data_of_all_pipelines[0].RAG_data[0]["MonoT5_score"] > 0.8 and \
            question_answer_data_of_all_pipelines[0].RAG_data[0]["MonoT5_score"] - question_answer_data_of_all_pipelines[1].RAG_data[0]["MonoT5_score"] > 0.15:
            is_correct = question_answer_data_of_all_pipelines[0].is_correct
            use_rag = True
            use_RAG_counter += 1
        elif question_answer_data_of_all_pipelines[1].RAG_data[0]["BM25_local_ranking"] <= 1 and \
            question_answer_data_of_all_pipelines[1].RAG_data[0]["SPLADE_local_ranking"] <= 1 and \
            question_answer_data_of_all_pipelines[1].RAG_data[0]["MonoT5_score"] > 0.0 and \
            question_answer_data_of_all_pipelines[1].RAG_data[0]["MonoT5_score"] - question_answer_data_of_all_pipelines[2].RAG_data[0]["MonoT5_score"] > 0.0:
            is_correct = question_answer_data_of_all_pipelines[1].is_correct
            use_rag = True
            use_RAG_counter += 1
        # elif question_answer_data_of_all_pipelines[1].RAG_data[0]["BM25_local_ranking"] <= 5 and \
        #     question_answer_data_of_all_pipelines[1].RAG_data[0]["SPLADE_local_ranking"] <= 5 and \
        #     question_answer_data_of_all_pipelines[1].RAG_data[0]["MonoT5_score"] > 0.8 and \
        #     question_answer_data_of_all_pipelines[1].RAG_data[0]["MonoT5_score"] - question_answer_data_of_all_pipelines[2].RAG_data[0]["MonoT5_score"] > 0.15:
        #     is_correct = top_2_results[i]
        #     use_rag = True
        #     use_RAG_counter += 1
        # elif question_answer_data_of_all_pipelines[2].RAG_data[0]["BM25_local_ranking"] <= 1 and \
        #     question_answer_data_of_all_pipelines[2].RAG_data[0]["SPLADE_local_ranking"] <= 1:
        #     chosen_data = question_answer_data_of_all_pipelines[2]
        #     use_RAG_counter += 1
        # elif question_answer_data_of_all_pipelines[3].RAG_data[0]["BM25_local_ranking"] <= 1 and \
        #     question_answer_data_of_all_pipelines[3].RAG_data[0]["SPLADE_local_ranking"] <= 1:
        #     chosen_data = question_answer_data_of_all_pipelines[3]
        #     use_RAG_counter += 1
        # elif question_answer_data_of_all_pipelines[4].RAG_data[0]["BM25_local_ranking"] <= 1 and \
        #     question_answer_data_of_all_pipelines[4].RAG_data[0]["SPLADE_local_ranking"] <= 1:
        #     chosen_data = question_answer_data_of_all_pipelines[4]
        #     use_RAG_counter += 1
            

        # use_rag = chosen_data != None
        # if use_rag:
        #     is_correct = chosen_data.is_correct
        # else:
        #     is_correct = no_rag_results[i]
        if not use_rag:
            is_correct = no_rag_results[i]

        # if best_score < 1.4:
        #     is_correct = no_rag_results[i]

        if i < 10:
            print(f"{i + 1}: {is_correct}")
        
        if is_correct:
            correct_counter += 1
            if use_rag:
                correct_counter_rag += 1
            else:
                correct_counter_no_rag += 1

    accuracy = correct_counter / QUESTION_COUNT

    print(f"use rag: {use_RAG_counter}/{QUESTION_COUNT}, rag accuracy: {0 if use_RAG_counter == 0 else correct_counter_rag / use_RAG_counter}, no rag accuracy: {correct_counter_no_rag / (QUESTION_COUNT - use_RAG_counter)}")
    # print(correct_counter_rag, correct_counter_no_rag)
    return accuracy

print("accuracy: ", get_accuracy_ranking_vote())