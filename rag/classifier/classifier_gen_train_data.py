import re
import os

QUESTION_COUNT = 10178

# log_file = "9-20-train_150_30_1st_H100.txt"
log_file = "9-20-train_150_30_2nd_H100.txt"

path = "rag/classifier/logs/"

file_name = path + log_file

class TrainingData:
    def __init__(self, context: str, question: str, is_correct: bool):
        self.context = context
        self.question = question
        self.is_correct = is_correct

training_data_list = []

with open(file_name) as f:
    is_reading_context = False
    context = ""

    is_reading_question = False
    question = ""

    for line in f:
        # if line == "\n":
        #     continue

        stripped_line = line.strip()
        if stripped_line == "Context:":
            is_reading_context = True
        elif stripped_line == "Question:":
            is_reading_context = False
            is_reading_question = True
        elif stripped_line.startswith("[output]"):
            is_correct = stripped_line[len("[output]"):] == "True"
            context = context.strip()
            question = question.strip()
            data = TrainingData(context, question, is_correct)
            # print("aaa", context, question, is_correct)

            training_data_list.append(data)

            if len(training_data_list) % 100 == 0:
                print(f"finished processing {len(training_data_list)} data")

            # clear
            is_reading_context = False
            context = ""
            is_reading_question = False
            question = ""
            # break
        else:
            if is_reading_context:
                context += line
            elif is_reading_question:
                question += line

print(len(training_data_list))