import numpy as np

# a = [0.1, 0.2, 0.3, 0.4]


def get_95_confidence_interval(accuracy_list):
    mean = np.mean(accuracy_list)
    standard_deviation = np.std(accuracy_list, ddof = 1)
    standard_error_of_mean = standard_deviation / np.sqrt(len(accuracy_list))
    z_score = 1.96 # for 95% confidence interval
    margin_of_error = z_score * standard_error_of_mean

    confidence_interval = (mean - margin_of_error, mean + margin_of_error)
    return confidence_interval

accuracy_list = [0.5302435192458759, 0.5294579732914375, 0.5365278868813825, 0.5326001571091908, 0.5365278868813825, 0.5318146111547526, 0.5365278868813825, 0.5271013354281225, 0.5333857030636292, 0.5310290652003142, 0.5357423409269442, 0.5192458758837392, 0.5404556166535742, 0.5388845247446976, 0.5310290652003142, 0.5380989787902593, 0.5349567949725059, 0.5294579732914375, 0.5247446975648076, 0.5412411626080126, 0.5349567949725059]

print(get_95_confidence_interval(accuracy_list))


# percentile_1 = np.percentile(accuracy_list, 2.5)
# percentile_2 = np.percentile(accuracy_list, 97.5)

# print((percentile_1, percentile_2))


def extract_answer(input_text):
    # Find the position of 'target:'
    target_string = "target: the answer to the question given the context is"
    start_idx = input_text.lower().find(target_string)
    
    if start_idx == -1:
        return None  # Handle the case where 'target:' is not found
    
    # Extract everything after the 'target:'
    answer = input_text[start_idx + len(target_string):].strip()
    
    # Remove extra spaces or newlines that may exist in the answer part
    clean_answer = ' '.join(answer.split())
    
    return clean_answer

input_text = """
source: question: Double balloon enteroscopy: is it efficacious and safe in a community setting? 
context: From March 2007 to January 2011, 88 DBE procedures were performed on 66 patients. Indications included evaluation anemia / gastrointestinal bleed, small bowel IBD and dilation of strictures. 
...
target: the answer to the question given the context is yes.
"""

answer = extract_answer(input_text)
print(answer)  # Output: yes