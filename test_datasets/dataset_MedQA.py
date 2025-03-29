# pip install transformers datasets

# datasets: hugging face

from datasets import load_dataset
import torch

def main():
    # medqa = load_dataset("bigbio/med_qa", trust_remote_code=True)
    medqa = load_dataset("GBaker/MedQA-USMLE-4-options", trust_remote_code=True)

    print("Dataset cache directory:", medqa.cache_files)

    print(medqa.keys()) # dict_keys(['train', 'test', 'validation'])

    print("-------------")

    print(medqa["train"][0])
    # {'meta_info': 'step2&3', 'question': 'A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient?', 'answer_idx': 'E', 'answer': 'Nitrofurantoin', 'options': [{'key': 'A', 'value': 'Ampicillin'}, {'key': 'B', 'value': 'Ceftriaxone'}, {'key': 'C', 'value': 'Ciprofloxacin'}, {'key': 'D', 'value': 'Doxycycline'}, {'key': 'E', 'value': 'Nitrofurantoin'}]}

    print("-------------")

    # print(medqa["validation"][0])
    # # {'meta_info': 'step1', 'question': 'A 21-year-old sexually active male complains of fever, pain during urination, and inflammation and pain in the right knee. A culture of the joint fluid shows a bacteria that does not ferment maltose and has no polysaccharide capsule. The physician orders antibiotic therapy for the patient. The mechanism of action of action of the medication given blocks cell wall synthesis, which of the following was given?', 'answer_idx': 'D', 'answer': 'Ceftriaxone', 'options': [{'key': 'A', 'value': 'Chloramphenicol'}, {'key': 'B', 'value': 'Gentamicin'}, {'key': 'C', 'value': 'Ciprofloxacin'}, {'key': 'D', 'value': 'Ceftriaxone'}, {'key': 'E', 'value': 'Trimethoprim'}]}

    # print("-------------")

    print(medqa["test"][0])
    # {'meta_info': 'step1', 'question': 'A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?', 'answer_idx': 'C', 'answer': 'Tell the attending that he cannot fail to disclose this mistake', 'options': [{'key': 'A', 'value': 'Disclose the error to the patient but leave it out of the operative report'}, {'key': 'B', 'value': 'Disclose the error to the patient and put it in the operative report'}, {'key': 'C', 'value': 'Tell the attending that he cannot fail to disclose this mistake'}, {'key': 'D', 'value': 'Report the physician to the ethics committee'}, {'key': 'E', 'value': 'Refuse to dictate the operative report'}]}

    print("-------------")
    
    # print(len(medqa["train"]), len(medqa["validation"]), len(medqa["test"])) # 10178 1272 1273
    print(len(medqa["train"]), len(medqa["test"])) # 10178 1273

    print("-------------------")
    print("-------------------")
    print("-------------------")

    # for i in range(0, 10):
    #     print(medqa["test"][i]["answer_idx"])

    print(medqa["train"][0])

if __name__ == "__main__":
    main()