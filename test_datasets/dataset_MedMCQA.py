# pip install transformers datasets

# datasets: hugging face

from datasets import load_dataset
import torch

def main():
    medmc_qa = load_dataset("openlifescienceai/medmcqa")

    print("Dataset cache directory:", medmc_qa.cache_files)

    print(medmc_qa.keys()) # dict_keys(['train', 'validation'])

    print("-------------")
    
    print(medmc_qa["train"][0])
    # {'id': 'e9ad821a-c438-4965-9f77-760819dfa155', 'question': 'Chronic urethral obstruction due to benign prismatic hyperplasia can lead to the following change in kidney parenchyma', 'opa': 'Hyperplasia', 'opb': 'Hyperophy', 'opc': 'Atrophy', 'opd': 'Dyplasia', 'cop': 2, 'choice_type': 'single', 'exp': 'Chronic urethral obstruction because of urinary calculi, prostatic hyperophy, tumors, normal pregnancy, tumors, uterine prolapse or functional disorders cause hydronephrosis which by definition is used to describe dilatation of renal pelvis and calculus associated with progressive atrophy of the kidney due to obstruction to the outflow of urine Refer Robbins 7yh/9,1012,9/e. P950', 'subject_name': 'Anatomy', 'topic_name': 'Urinary tract'}
    print("-------------")

    print(medmc_qa["validation"][0])
    # {'id': '45258d3d-b974-44dd-a161-c3fccbdadd88', 'question': 'Which of the following is not true for myelinated nerve fibers:', 'opa': 'Impulse through myelinated fibers is slower than non-myelinated fibers', 'opb': 'Membrane currents are generated at nodes of Ranvier', 'opc': 'Saltatory conduction of impulses is seen', 'opd': 'Local anesthesia is effective only when the nerve is not covered by myelin sheath', 'cop': 0, 'choice_type': 'multi', 'exp': None, 'subject_name': 'Physiology', 'topic_name': None}

    print("-------------")

    print(medmc_qa["test"][0])
    # {'id': '84f328d3-fca4-422d-8fb2-19d55eb31503', 'question': 'Which of the following is derived from fibroblast cells ?', 'opa': 'TGF-13', 'opb': 'MMP2', 'opc': 'Collagen', 'opd': 'Angiopoietin', 'cop': -1, 'choice_type': 'single', 'exp': '', 'subject_name': 'Pathology', 'topic_name': None}

    print("-------------")
    
    print(len(medmc_qa["train"]), len(medmc_qa["validation"]), len(medmc_qa["test"])) # 182822 4183 6150

    counter = 0
    counter2 = 0
    for data in medmc_qa["test"]:
        #if (isinstance(data["cop"], int)):
        if (data["cop"] > 3):
            counter += 1
        else:
            counter2 += 1
    print(counter, counter2)


if __name__ == "__main__":
    main()