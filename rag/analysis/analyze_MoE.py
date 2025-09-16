import re
import itertools

QUESTION_COUNT = 1273

path = "rag/analysis/logs/"
# file_names = ["MedQA_810_noRAG.txt", "8-19-150_30_1.txt", "8-21-100_10_1.txt", "8-18-150_30_3.txt", "8-13-150_10_3.txt", "8-20-2_2_2.txt", "8-21-1_1_1.txt"]
# file_names = ["MedQA_810_noRAG.txt", "8-19-150_30_1.txt", "8-21-100_10_1.txt", "8-18-150_30_3.txt", "8-20-2_2_2.txt", "8-21-1_1_1.txt"]
# file_names = ["8-19-150_30_1.txt", "8-18-150_30_3.txt", "8-13-150_10_3.txt", "8-13-150_10_3.txt", "8-20-2_2_2.txt"]
file_names = ["MedQA_810_noRAG.txt", "8-19-150_30_1.txt", "8-21-100_10_1.txt", "8-18-150_30_3.txt", "8-13-150_10_3.txt", "8-20-2_2_2.txt", "8-21-1_1_1.txt", 
              "8-13-25_10_3.txt", "8-19-150_30_2.txt", "8-19-150_40_3.txt", "8-19-150_90_3.txt", "8-20-150_30_4.txt", "8-20-150_30_5.txt", "8-20-BM25_3.txt",
              "8-18-150_10_3_bge.txt", "8-16-175_10_3.txt", "8-25-150_30_2nd.txt", "8-25-150_30_3rd.txt", "9-2-150-30-0-1-Qwen-H100.txt", "8-31-150-30-3-0.7-A100.txt"]

# file_names = ["8-21-100_10_1.txt", "8-25-150_30_2nd.txt", "8-25-150_30_3rd.txt", "8-26-150_30_4th.txt", "8-25-150_30_5th.txt"]

full_file_names = [path + file_name for file_name in file_names]

answers_list = []
correct_answers = None
for i in range(len(full_file_names)):
    with open(full_file_names[i], "r") as f:
        answers = []
        curr_correct_answers = []
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
                    
                    answers.append(answer)
                    curr_correct_answers.append(correct_answer)
        
        answers_list.append(answers)
        if correct_answers == None:
            correct_answers = curr_correct_answers

# Check answers count == QUESTION_COUNT (1273)
for i in range(len(answers_list)):
    answers = answers_list[i]
    if len(answers) != QUESTION_COUNT:
        print(f"Error: {file_names[i]}: answers index {i} has {len(answers)} != {QUESTION_COUNT}")

# Individual accuracy
accuracy_list = []
for answers in answers_list:
    correct_counter = 0
    for i in range(len(correct_answers)):
        if answers[i] == correct_answers[i]:
            correct_counter += 1
    accuracy = float(correct_counter) / len(correct_answers)
    accuracy_list.append(accuracy)
print("Individual accuracy: ", accuracy_list)


# Mixture of Experts
def get_accuracy(chosen_index_list):
    chosen_answers_list = [answers_list[i] for i in chosen_index_list]

    correct_counter = 0
    for i in range(len(correct_answers)):
        votes = []
        for j in range(len(chosen_answers_list)):
            votes.append(chosen_answers_list[j][i])
        
        answer_to_count = {}
        for vote in votes:
            if vote not in answer_to_count.keys():
                answer_to_count[vote] = 1
            else:
                answer_to_count[vote] += 1
            
        voted_answer = max(answer_to_count, key = answer_to_count.get)
        # print(answer_to_count, voted_answer, correct_answers[i])
        if voted_answer == correct_answers[i]:
            correct_counter += 1

    accuracy = float(correct_counter) / len(correct_answers)
    return accuracy

# print(get_accuracy([0, 1, 2, 3, 4, 5, 6]))

min_experts_count = 3
max_experts_count = 3 # len(file_names)

all_indices = range(len(file_names))
all_possible_index_lists = []
for i in range(min_experts_count, max_experts_count + 1):
    all_possible_index_lists.extend(itertools.combinations(all_indices, i))

print (all_possible_index_lists)

print(f"Checking {len(all_possible_index_lists)} possible combinations...")

best_accuracy = 0
best_index_list = None
for i in range(len(all_possible_index_lists)):
    index_list = all_possible_index_lists[i]
    accuracy = get_accuracy(index_list)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_index_list = index_list
    
    if i % 10 == 0:
        print(f"finished {i + 1}/{len(all_possible_index_lists)}, current best accuracy: {best_accuracy}")

print(f"Mixture of Expert best accuracy: {best_accuracy}, index list: {best_index_list}")
print(f"Best file names: {[file_names[i] for i in best_index_list]}")

# print(answers_list)
# print(correct_answers)

# finished 65381/65382, current best accuracy: 0.7439120188531029
# Mixture of Expert best accuracy: 0.7439120188531029, index list: (0, 1, 4, 5, 8, 9, 11, 14)
# Best file names: ['rag/analysis/logs/MedQA_810_noRAG.txt', 'rag/analysis/logs/8-19-150_30_1.txt', 'rag/analysis/logs/8-13-150_10_3.txt', 'rag/analysis/logs/8-20-2_2_2.txt', 'rag/analysis/logs/8-19-150_30_2.txt', 'rag/analysis/logs/8-19-150_40_3.txt', 'rag/analysis/logs/8-20-150_30_4.txt', 'rag/analysis/logs/8-18-150_10_3_bge.txt']


# finished 6741/6748, current best accuracy: 0.7423409269442263
# Mixture of Expert best accuracy: 0.7423409269442263, index list: (0, 1, 9, 13, 14)
# Best file names: ['rag/analysis/logs/MedQA_810_noRAG.txt', 'rag/analysis/logs/8-19-150_30_1.txt', 'rag/analysis/logs/8-19-150_40_3.txt', 'rag/analysis/logs/8-20-BM25_3.txt', 'rag/analysis/logs/8-18-150_10_3_bge.txt']


# finished 3871/3876, current best accuracy: 0.7407698350353495
# Mixture of Expert best accuracy: 0.7407698350353495, index list: (0, 1, 3, 14)
# Best file names: ['rag/analysis/logs/MedQA_810_noRAG.txt', 'rag/analysis/logs/8-19-150_30_1.txt', 'rag/analysis/logs/8-18-150_30_3.txt', 'rag/analysis/logs/8-18-150_10_3_bge.txt']


# finished 811/816, current best accuracy: 0.7266300078554595
# Mixture of Expert best accuracy: 0.7266300078554595, index list: (0, 9, 14)
# Best file names: ['rag/analysis/logs/MedQA_810_noRAG.txt', 'rag/analysis/logs/8-19-150_40_3.txt', 'rag/analysis/logs/8-18-150_10_3_bge.txt']