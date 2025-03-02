from model_loader import get_model_and_tokenizer
from data_loader import data_loader
from trainer import trainer

model, tokenizer = get_model_and_tokenizer()
dataset = data_loader(tokenizer)
trainer(model, tokenizer, dataset)