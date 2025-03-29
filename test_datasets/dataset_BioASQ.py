# pip install transformers datasets

# datasets: hugging face

from datasets import load_dataset
import torch

def main():
    bio_asq = load_dataset("kroshan/BioASQ", trust_remote_code=True)

    print("Dataset cache directory:", bio_asq.cache_files)

    print(bio_asq.keys()) # dict_keys(['train', 'test', 'validation'])

    print("-------------")

    print(bio_asq["train"][0])
    # {'question': 'What is the inheritance pattern of Li–Fraumeni syndrome?', 'text': '<answer> autosomal dominant <context> Balanced t(11;15)(q23;q15) in a TP53+/+ breast cancer patient from a Li-Fraumeni syndrome family. Li-Fraumeni Syndrome (LFS) is characterized by early-onset carcinogenesis involving multiple tumor types and shows autosomal dominant inheritance. Approximately 70% of LFS cases are due to germline mutations in the TP53 gene on chromosome 17p13.1. Mutations have also been found in the CHEK2 gene on chromosome 22q11, and others have been mapped to chromosome 11q23. While characterizing an LFS family with a documented defect in TP53, we found one family member who developed bilateral breast cancer at age 37 yet was homozygous for wild-type TP53. Her mother also developed early-onset primary bilateral breast cancer, and a sister had unilateral breast cancer and a soft tissue sarcoma. Cytogenetic analysis using fluorescence in situ hybridization of a primary skin fibroblast cell line revealed that the patient had a novel balanced reciprocal translocation between the long arms of chromosomes 11 and 15: t(11;15)(q23;q15). This translocation was not present in a primary skin fibroblast cell line from a brother with neuroblastoma, who was heterozygous for the TP53 mutation. There was no evidence of acute lymphoblastic leukemia in either the patient or her mother, although a nephew did develop leukemia and died in childhood. These data may implicate the region at breakpoint 11q23 and/or 15q15 as playing a significant role in predisposition to breast cancer development.'}

    print("-------------")

    print(bio_asq["validation"][0])
    # {'question': 'Name synonym of Acrokeratosis paraneoplastica.', 'text': '<answer> Bazex syndrome <context> Acrokeratosis paraneoplastica (Bazex syndrome): report of a case associated with small cell lung carcinoma and review of the literature. Acrokeratosis paraneoplastic (Bazex syndrome) is a rare, but distinctive paraneoplastic dermatosis characterized by erythematosquamous lesions located at the acral sites and is most commonly associated with carcinomas of the upper aerodigestive tract. We report a 58-year-old female with a history of a pigmented rash on her extremities, thick keratotic plaques on her hands, and brittle nails. Chest imaging revealed a right upper lobe mass that was proven to be small cell lung carcinoma. While Bazex syndrome has been described in the dermatology literature, it is also important for the radiologist to be aware of this entity and its common presentations.'}

    print("-------------")

    print(len(bio_asq["train"]), len(bio_asq["validation"])) # 3266 4950

if __name__ == "__main__":
    main()