from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import shutil
from test_performance import TestPerformance

class ModelTester():
    def __init__(self):

        self.model = None
        self.tokenizer = None
        self.device = None

        self.check_gpu()

    # Check GPU name and status
    def check_gpu(self):
        print(f"---------- start checking GPU -----------")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print("torch.cuda.is_bf16_supported(): ", torch.cuda.is_bf16_supported())
        else:
            print("GPU not available.")
        print(f"---------- finish checking GPU -----------")

    # Load model
    def load_model(self, model_name, tokenizer_name):
        print(f"---------- start loading model:{model_name}, tokenizer:{tokenizer_name} -----------")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code = False)
        print("finish loading tokenizer")
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code = False, 
                                                    #  torch_dtype= torch.float16 # Sometimes RuntimeError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'
                                                     torch_dtype = torch.bfloat16
                                                     )
        print("finish loading model")
        print("torch_dtype:", model.config.torch_dtype)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print("cuda available:", torch.cuda.is_available())
        print("device:", device)

        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        print(f"---------- finish loading model:{self.model.name_or_path} -----------")

    # Test the response format of current model (MedQA question 0)
    def test_model_MedQA_response(self):
        print(f"----------- start testing model {self.model.name_or_path} performance ------------")
        test = TestPerformance(self.model, self.tokenizer, self.device)
        test.test_MedQA_response()
        print(f"----------- finish testing model {self.model.name_or_path} performance ------------")
    
    # Test the performance on MedQA test dataset of current model
    def test_model_MedQA_accuracy(self):
        print(f"----------- start testing model {self.model.name_or_path} accuracy ------------")
        test = TestPerformance(self.model, self.tokenizer, self.device)
        test.test_MedQA_test_data_accuracy()
        print(f"----------- finish testing model {self.model.name_or_path} accuracy ------------")


if __name__ == "__main__":
    print("time:", datetime.now())

    tester = ModelTester()

    # model_name = "./fine_tuned_model_checkpoints/checkpoint-20000/"
    model_name = "./fine_tuned_model_entire_UltraMedical_batch_32"
    tokenizer_name = model_name #"microsoft/Phi-3-mini-4k-instruct"
    tester.load_model(model_name, tokenizer_name)
    tester.test_model_MedQA_response()
    tester.test_model_MedQA_accuracy()

    print("All done.")

