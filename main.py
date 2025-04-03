from services.huggingface.model_loader import get_model_and_tokenizer_hf
from services.huggingface.data_loader import data_loader, data_loader_eli5
from services.huggingface.trainer_loader import trainer_hf

def main():
    ft_mode = "lora"
    model, tokenizer = get_model_and_tokenizer_hf("lora")
    # dataset_train = data_loader(tokenizer, "train")
    dataset = data_loader_eli5(tokenizer)
    # dataset_validation = data_loader(tokenizer, "validation")
    trainer = trainer_hf(model, tokenizer, dataset["train"], dataset["test"], ft_mode)
    trainer.train()
    # trainer.push_to_hub()
    

if __name__ == "__main__":
    main()