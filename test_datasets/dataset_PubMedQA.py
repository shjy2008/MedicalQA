# pip install transformers datasets

# datasets: hugging face

from datasets import load_dataset
import torch

def main():
    pubmed_qa = load_dataset("bigbio/pubmed_qa", trust_remote_code=True)

    print("Dataset cache directory:", pubmed_qa.cache_files)

    print(pubmed_qa.keys()) # dict_keys(['train', 'validation'])

    print("-------------")
    
    print(pubmed_qa["train"][0])
    # {'QUESTION': 'Does neurobehavioral disinhibition predict initiation of substance use in children with prenatal cocaine exposure?', 'CONTEXTS': ['In previous work we (Fisher et al., 2011) examined the emergence of neurobehavioral disinhibition (ND) in adolescents with prenatal substance exposure. We computed ND factor scores at three age points (8/9, 11 and 13/14 years) and found that both prenatal substance exposure and early adversity predicted ND. The purpose of the current study was to determine the association between these ND scores and initiation of substance use between ages 8 and 16 in this cohort as early initiation of substance use has been related to later substance use disorders. Our hypothesis was that prenatal cocaine exposure predisposes the child to ND, which, in turn, is associated with initiation of substance use by age 16.', "We studied 386 cocaine exposed and 517 unexposed children followed since birth in a longitudinal study. Five dichotomous variables were computed based on the subject's report of substance use: alcohol only; tobacco only; marijuana only; illicit substances and any substance.", 'Cox proportional hazard regression showed that the 8/9 year ND score was related to initiation of alcohol, tobacco, illicit and any substance use but not marijuana use. The trajectory of ND across the three age periods was related to substance use initiation in all five substance use categories. Prenatal cocaine exposure, although initially related to tobacco, marijuana and illicit substance initiation, was no longer significant with ND scores in the models.'], 'LABELS': ['BACKGROUND', 'METHODS', 'RESULTS'], 'MESHES': ['Adolescent', 'Adult', 'Caregivers', 'Child', 'Child Abuse', 'Child Behavior Disorders', 'Cocaine', 'Depression', 'Domestic Violence', 'Female', 'Humans', 'Inhibition (Psychology)', 'Longitudinal Studies', 'Marijuana Abuse', 'Pregnancy', 'Prenatal Exposure Delayed Effects', 'Proportional Hazards Models', 'Puberty', 'Smoking', 'Social Environment', 'Socioeconomic Factors', 'Stress, Psychological', 'Substance-Related Disorders', 'Violence'], 'YEAR': None, 'reasoning_required_pred': None, 'reasoning_free_pred': None, 'final_decision': 'yes', 'LONG_ANSWER': 'Prenatal drug exposure appears to be a risk pathway to ND, which by 8/9 years portends substance use initiation.'}

    print("-------------")

    print(pubmed_qa["validation"][0])
    # {'QUESTION': 'Do posterior fossa and spinal gangliogliomas form two distinct clinicopathologic and molecular subgroups?', 'CONTEXTS': ['Gangliogliomas are low-grade glioneuronal tumors of the central nervous system and the commonest cause of chronic intractable epilepsy. Most gangliogliomas (>70%) arise in the temporal lobe, and infratentorial tumors account for less than 10%. Posterior fossa gangliogliomas can have the features of a classic supratentorial tumor or a pilocytic astrocytoma with focal gangliocytic differentiation, and this observation led to the hypothesis tested in this study - gangliogliomas of the posterior fossa and spinal cord consist of two morphologic types that can be distinguished by specific genetic alterations.', 'Histological review of 27 pediatric gangliogliomas from the posterior fossa and spinal cord indicated that they could be readily placed into two groups: classic gangliogliomas (group I; n\u2009=\u200916) and tumors that appeared largely as a pilocytic astrocytoma, but with foci of gangliocytic differentiation (group II; n\u2009=\u200911). Detailed radiological review, which was blind to morphologic assignment, identified a triad of features, hemorrhage, midline location, and the presence of cysts or necrosis, that distinguished the two morphological groups with a sensitivity of 91% and specificity of 100%. Molecular genetic analysis revealed BRAF duplication and a KIAA1549-BRAF fusion gene in 82% of group II tumors, but in none of the group I tumors, and a BRAF:p.V600E mutation in 43% of group I tumors, but in none of the group II tumors.'], 'LABELS': ['BACKGROUND', 'RESULTS'], 'MESHES': ['Adolescent', 'Child', 'Child, Preschool', 'Female', 'Ganglioglioma', 'Genetic Testing', 'Humans', 'Infant', 'Infratentorial Neoplasms', 'Male', 'Mutation', 'Proto-Oncogene Proteins B-raf', 'Recombinant Fusion Proteins', 'Spinal Cord Neoplasms', 'Young Adult'], 'YEAR': None, 'reasoning_required_pred': None, 'reasoning_free_pred': None, 'final_decision': 'yes', 'LONG_ANSWER': 'Our study provides support for a classification that would divide infratentorial gangliogliomas into two categories, (classic) gangliogliomas and pilocytic astrocytomas with gangliocytic differentiation, which have distinct morphological, radiological, and molecular characteristics.'}

    print("-------------")

    print(len(pubmed_qa["train"]), len(pubmed_qa["validation"])) # 200000 11269

if __name__ == "__main__":
    main()