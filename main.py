from services.huggingface.model_loader import get_model_and_tokenizer_hf
from services.huggingface.data_loader import data_loader, data_loader_eli5
from services.huggingface.trainer_loader import trainer_hf
from utils.globals import EPOCHS

def main():
    ft_mode = "lora"
    model, tokenizer = get_model_and_tokenizer_hf("lora")
    dataset_train = data_loader(tokenizer, "train")
    # dataset = data_loader_eli5(tokenizer)
    dataset_validation = data_loader(tokenizer, "validation")
    model_name = f"LoLlama3.2-1B-{ft_mode}-{EPOCHS}ep"
    
    # trainer = trainer_hf(model_name, model, tokenizer, dataset["train"], dataset["test"])
    trainer = trainer_hf(model_name, model, tokenizer, dataset_train, dataset_validation)
    trainer.train()
    trainer.push_to_hub()
    

if __name__ == "__main__":
    main()