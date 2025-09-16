import re

QUESTION_COUNT = 1273

path = "rag/analysis/logs/"
# file_names = ["MedQA_810_noRAG.txt", "8-19-150_30_1.txt", "8-25-150_30_2nd.txt", "8-25-150_30_3rd.txt"]
# file_names = ["8-19-150_30_1.txt", "8-25-150_30_2nd.txt", "8-25-150_30_3rd.txt", "8-26-150_30_4th.txt", "8-25-150_30_5th.txt"]
file_names = ["9-5-150_30_1_RAG_data_H100.txt", "9-5-150_30_2nd_RAG_data_H100.txt", "9-5-150_30_3rd_RAG_data_H100.txt"]

file_names = [path + file_name for file_name in file_names]


answers_list = []
correct_answers = None
for file_name in file_names:
    with open(file_name, "r") as f:
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
                    print(f"Error: {file_name}:", number, count)
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

individual_correct_counters = [0] * len(answers_list)
accumulated_correct_counters = [0] * len(answers_list) # only consider top n
correct_counter = 0
for i in range(len(correct_answers)):
    correct_answer = correct_answers[i]
    is_correct = False
    for j in range(len(answers_list)):
        individual_answer = answers_list[j][i]
        if correct_answer == individual_answer:
            individual_correct_counters[j] += 1
            is_correct = True
        
        if is_correct:
            accumulated_correct_counters[j] += 1


individual_accuracy_list = [float(correct_counter) / len(correct_answers) for correct_counter in individual_correct_counters]
print(f"Individual accuracy list: {individual_accuracy_list}")

accumulated_accuracy_list = [float(correct_counter) / len(correct_answers) for correct_counter in accumulated_correct_counters]
print(f"Accumulated accuracy list: {accumulated_accuracy_list}")

# Individual accuracy list: [0.6732128829536528, 0.6661429693637078, 0.6606441476826395, 0.6512175962293795, 0.6575019638648861]
# Accumulated accuracy list: [0.6732128829536528, 0.7941869599371564, 0.8413197172034564, 0.8695993715632364, 0.8900235663786331]
