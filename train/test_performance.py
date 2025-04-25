from datasets import load_dataset
import torch
import re
import time

class TestPerformance():

    def __init__(self, model, tokenizer, device, temperature = 0.00001):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
    
    def test_MedQA_response(self):# Test apply_chat_template // https://huggingface.co/docs/transformers/main/en/chat_templating
        print(self.model.name_or_path)

        def format_choices(choices):
            a = zip(list(choices.keys()), choices.values())
            final_answers = []
            for x,y in a:
                final_answers.append(f'[{x}] : {y}')
            return "\n".join(final_answers)


        def run_inference(content, model, tokenizer, max_new_tokens, temperature):
            messages = [{"role": "user", "content": f"{content}"}]
            # add_generation_prompt indicates the start of a response
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt = True, return_tensors = "pt").to(self.device)
            # print("inputs:", tokenizer.apply_chat_template(messages, add_generation_prompt = True, tokenize = False))
            outputs = model.generate(inputs, max_new_tokens = max_new_tokens, do_sample = True, temperature = temperature)
            text = tokenizer.batch_decode(outputs)[0]
            return text

        prompt = f'''
        {{question}} \n
        {{choices}}
        '''

        examples = [
        # Training data index 0 (GBaker/MedQA-USMLE-4-options)
        # correct: D
        {"question": "A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient?",
        "choices": {
        "A": "Ampicillin",
        "B": "Ceftriaxone",
        "C": "Doxycycline",
        "D": "Nitrofurantoin"
        },
        "correct": "D",
        "source": "Training data index 0 (GBaker/MedQA-USMLE-4-options)"
        },

        # Training data index 1
        # correct: A
        {"question": "A 3-month-old baby died suddenly at night while asleep. His mother noticed that he had died only after she awoke in the morning. No cause of death was determined based on the autopsy. Which of the following precautions could have prevented the death of the baby?",
        "choices": {
        "A": "Placing the infant in a supine position on a firm mattress while sleeping",
        "B": "Keeping the infant covered and maintaining a high room temperature",
        "C": "Application of a device to maintain the sleeping position",
        "D": "Avoiding pacifier use during sleep"
        },
        "correct": "A",
        "source": "Training data index 1 (GBaker/MedQA-USMLE-4-options)"
        },

        # Training data index 2
        # correct: A
        {"question": "A mother brings her 3-week-old infant to the pediatrician's office because she is concerned about his feeding habits. He was born without complications and has not had any medical problems up until this time. However, for the past 4 days, he has been fussy, is regurgitating all of his feeds, and his vomit is yellow in color. On physical exam, the child's abdomen is minimally distended but no other abnormalities are appreciated. Which of the following embryologic errors could account for this presentation?",
        "choices": {
        "A": "Abnormal migration of ventral pancreatic bud",
        "B": "Complete failure of proximal duodenum to recanalize",
        "C": "Abnormal hypertrophy of the pylorus",
        "D": "Failure of lateral body folds to move ventrally and fuse in the midline"
        },
        "correct": "A",
        "source": "Training data index 2 (GBaker/MedQA-USMLE-4-options)"
        },

        # Test data index 0
        # correct: B
        {"question": "A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?",
        "choices":{
        "A": "Disclose the error to the patient and put it in the operative report",
        "B": "Tell the attending that he cannot fail to disclose this mistake",
        "C": "Report the physician to the ethics committee",
        "D": "Refuse to dictate the operative report"
        },
        "correct": "B",
        "source": "Test data index 0 (GBaker/MedQA-USMLE-4-options)"
        },

        ]

        for example in examples:
            formated_choices = format_choices(example["choices"])
    
            model_prompt = prompt.format(question = example["question"], choices = formated_choices)
    
            # print(model_prompt)
    
            output_text = run_inference(model_prompt, self.model, self.tokenizer, max_new_tokens = 1024, temperature = self.temperature)
            # output_text = output_text.split("<|assistant|>")[-1]
            print("model output:", output_text)
            print("\nsource: ", example["source"])
            print("correct: ", example["correct"])
            print("\n\n")

    def test_MedQA_test_data_accuracy(self, is_ensemble = False):

        MAX_TOKEN_OUTPUT = 1024
        SPLIT = "test"
        DATA_RANGE = None #range(0, 100)

        regex_pattern=r"[\(\[]([A-Z])[\)\]]"
        regex = re.compile(regex_pattern)

        def format_choices(choices):
            a = zip(list(choices.keys()), choices.values())
            final_answers = []
            for x,y in a:
                final_answers.append(f'[{x}] : {y}')
            return "\n".join(final_answers)

        def find_match(regex, resp, convert_dict={}):
            match = regex.findall(resp)
            if match:
                match = match[-1]
                if isinstance(match, tuple):
                    match = [m for m in match if m][0]
                match = match.strip()
                if match and match in convert_dict: 
                    match = convert_dict[match]
            return match
                
        def extract_answer(response):
            matchFirst = re.search(r'the answer is .(\w).', response)
            if matchFirst:
                return f"({matchFirst.group(1)})"
            match = find_match(regex, response) 
            if match:
                return f"({match})"
            return "[invalid]"

        def run_inference_get_answer_letter(inputs, temperature):
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens = MAX_TOKEN_OUTPUT,
                    do_sample=True,
                    temperature = temperature,
                )

            # print("temperature:", temperature)
            # print("inputs", inputs)
            
            text = self.tokenizer.batch_decode(outputs)[0]
            # print("outputs text:", text)
            text = text.split("<|assistant|>")[-1]
            # answer = tokenizer.decode(output[0], skip_special_tokens = True)

            answer = extract_answer(text).strip("()")
            
            print(f"answer: {answer}")
            
            return answer
                
        def get_medqa_accuracy():
            # Load MedQA dataset
            # med_qa = load_dataset("bigbio/med_qa", trust_remote_code = True)
            med_qa = load_dataset("GBaker/MedQA-USMLE-4-options", trust_remote_code = True)
            keys = med_qa.keys()
            # print(len(med_qa["train"]), len(med_qa["validation"]), len(med_qa["test"]))

            print(f"model: {self.model.name_or_path}")
            
            start_time = time.time()
            
            data_list = med_qa[SPLIT]
            if DATA_RANGE != None:
                data_list = data_list.select(DATA_RANGE)
            count = 0
            correct_count = 0
            for data in data_list:
                question = data["question"]
                answer_idx = data["answer_idx"]
                choices = data["options"]

                prompt = f'''\n{{question}}\n{{choices}}\n'''

                formated_choices = format_choices(choices)
                
                model_prompt = prompt.format(question = question, choices = formated_choices)
                
                messages = [{"role": "user", "content": f"{model_prompt}"}]
                inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
                
                # print("messages: ", messages)
                # break
                
                # print("correct answer: ", answer_idx)

                if is_ensemble:
                    answer_dict = {}
                    for i in range(0, 5):
                        # inputs = tokenizer(model_prompt, return_tensors = "pt", padding=False).to(device)
                        current_answer = run_inference_get_answer_letter(inputs, temperature = 0.7)
                        if current_answer in answer_dict:
                            answer_dict[current_answer] += 1
                        else:
                            answer_dict[current_answer] = 1
                    answer = max(answer_dict, key = answer_dict.get)
                else:
                    answer = run_inference_get_answer_letter(inputs, temperature = self.temperature)
                
                correct_answer = answer_idx
            
                is_correct = (answer == correct_answer)
                # print("Correct!!!" if is_correct else "Wrong")
            
            
                if is_correct:
                    correct_count += 1
            
                count += 1

                print(f"question {count}/{len(data_list)} answer:{answer} correct_answer:{correct_answer} {is_correct}")
            
            accuracy = correct_count / count
            print(f"Total questions: {count}, correct: {correct_count}, accuracy: {accuracy}")
            
            finish_time = time.time()
            elapse_time = finish_time - start_time
            print(f"elapse_time: {elapse_time}")

            return accuracy

        get_medqa_accuracy()