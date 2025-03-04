from services.huggingface.model_loader import get_model_and_tokenizer_hf
from services.huggingface.data_loader import data_loader
from services.huggingface.trainer_loader import trainer_hf

def main():
    model, tokenizer = get_model_and_tokenizer_hf("lora")
    dataset_train = data_loader(tokenizer, "train")
    dataset_validation = data_loader(tokenizer, "validation")
    trainer = trainer_hf(model, tokenizer, dataset_train, dataset_validation)
    trainer.train()
    trainer.push_to_hub()
    

if __name__ == "__main__":
    main()