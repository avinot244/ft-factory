from services.huggingface.CLM.model_loader import get_model_and_tokenizer_CLM
from services.huggingface.CLM.data_loader import data_loader_CLM, data_loader_eli5
from services.huggingface.CLM.trainer_loader import trainer_CLM

from services.huggingface.MLM.model_loader import get_model_and_tokenizer_MLM
from services.huggingface.MLM.data_loader import data_loader_MLM
from services.huggingface.MLM.trainer_loader import trainer_MLM
from utils.globals import EPOCHS
from utils.token_manager import get_hf_token
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def main():
    for epochs in [5, 6, 7]:
        ft_mode = "lora"
        model, tokenizer = get_model_and_tokenizer_CLM(ft_mode)
        dataset_train = data_loader_CLM(tokenizer, "train")
        # dataset = data_loader_eli5(tokenizer)
        dataset_validation = data_loader_CLM(tokenizer, "validation")
        model_name = f"LoLlama3.2-1B-{ft_mode}-{epochs}ep-v3"
        
        # trainer = trainer_hf(model_name, model, tokenizer, dataset["train"], dataset["test"])
        trainer = trainer_CLM(model_name, model, tokenizer, dataset_train, dataset_validation, epochs)
        trainer.train()
        trainer.push_to_hub()
    
    
    # for epochs in [2]:
    #     model, tokenizer = get_model_and_tokenizer_MLM("distilroberta-base")
    #     dataset_train = data_loader_MLM(tokenizer, "train")
    #     # dataset = data_loader_eli5(tokenizer)
    #     dataset_validation = data_loader_MLM(tokenizer, "validation")
    #     model_name = f"distilolroberta-MLM-{epochs}ep-v2"
        
    #     # trainer = trainer_hf(model_name, model, tokenizer, dataset["train"], dataset["test"])
    #     trainer = trainer_MLM(model_name, model, tokenizer, dataset_train, dataset_validation, epochs)
    #     trainer.train()
    #     trainer.push_to_hub()
    #     tokenizer.push_to_hub(f"{model_name}-tok", token=get_hf_token("write"))
    

if __name__ == "__main__":
    main()