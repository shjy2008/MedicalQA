import re
import os

QUESTION_COUNT = 1273

path = "rag/analysis/logs/"


file_names = os.listdir(path)
file_names = [file_name for file_name in file_names if file_name.endswith(".txt")]

# file_names = ["MedQA_810_noRAG.txt", "8-19-150_30_1.txt", "8-21-100_10_1.txt", "8-18-150_30_3.txt", "8-13-150_10_3.txt", "8-20-2_2_2.txt", "8-21-1_1_1.txt", 
#               "8-13-25_10_3.txt", "8-19-150_30_2.txt", "8-19-150_40_3.txt", "8-19-150_90_3.txt", "8-20-150_30_4.txt", "8-20-150_30_5.txt", "8-20-BM25_3.txt",
#               "8-18-150_10_3_bge.txt", "8-16-175_10_3.txt", "8-25-150_30_2nd.txt", "8-25-150_30_3rd.txt", "9-2-150-30-0-1-Qwen-H100.txt", "8-31-150-30-3-0.7-A100.txt",
#               "9-3-150_30_1_threshold_0.8_H100.txt", "9-3-150_30_2_threshold_0.8_H100.txt"]

file_names = ["MedQA_810_noRAG.txt", "8-19-150_30_1.txt", "8-25-150_30_2nd.txt", "8-25-150_30_3rd.txt", "8-26-150_30_4th.txt", "8-25-150_30_5th.txt"]

full_file_names = [path + file_name for file_name in file_names]

correct_indices_for_each_file = []

for i in range(len(full_file_names)):
    correct_indices = []
    with open(full_file_names[i], "r") as f:
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
                    
                    if answer == correct_answer:
                        correct_indices.append(count - 1)
    
    correct_indices_for_each_file.append(correct_indices)

print([len(correct_indices) / QUESTION_COUNT for correct_indices in correct_indices_for_each_file])


correct_files_for_each_index = []

for i in range(QUESTION_COUNT):
    correct_files = []
    for j in range(len(correct_indices_for_each_file)):
        if i in correct_indices_for_each_file[j]:
            correct_files.append(file_names[j])
    correct_files_for_each_index.append(correct_files)

# for i in range(len(correct_files_for_each_index)):
#     print(f"{i+1}: {correct_files_for_each_index[i]}")


upper_bound_count = 0
for correct_files in correct_files_for_each_index:
    if len(correct_files) > 0:
        upper_bound_count += 1
print(f"upper bound: {upper_bound_count / QUESTION_COUNT}")

