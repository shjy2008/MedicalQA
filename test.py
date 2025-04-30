from datetime import datetime
print(datetime.now())

count = 0
correct_count = 0
# file_name = "./train/test/test_MedQA_entire.txt"
file_name = "./train/test/notebook_log_test_fine_tuned.txt"
# file_name = "./train/test/test_MedQA_checkpoint_200000.txt"
with open(file_name, "r") as file:
    for line in file:
        if "question" in line: # line.startswith("question"):
            count += 1
            # number = line.strip().split(' ')[1].split('/')[0]
            # if int(number) != count:
            # 	print(number, count)
            # 	break
            result = line.strip().split(' ')[-1]
            if result == "True":
                correct_count += 1
            else:
                if result != "False":
                    print(result)

accuracy = correct_count / count
print(f"Total questions: {count}, correct: {correct_count}, accuracy: {accuracy}")