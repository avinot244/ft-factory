from model_loader import get_model_and_tokenizer_hf
from data_loader import data_loader
from trainer import trainer_hf

model, tokenizer = get_model_and_tokenizer_hf()
dataset_train = data_loader(tokenizer, "train")
dataset_validation = data_loader(tokenizer, "validation")
trainer = trainer_hf(model, dataset_train, dataset_validation)
trainer.train()