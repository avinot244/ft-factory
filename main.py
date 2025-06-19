import torch
from services.huggingface.contrastive.model_loader import PredictionHead
from services.huggingface.contrastive.data_loader import TripletDataset
from services.huggingface.contrastive.trainer_loader import train
from utils.types.TrainingArgs import ContrastiveTrainingArgs
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def main():
    model = PredictionHead(input_dim=2048, output_dim=2048)

    dataset_train = TripletDataset(split="train")
    dataset_validation = TripletDataset(split="validation")
    
    training_args = ContrastiveTrainingArgs(
        output_dir="output/",
        logging_path="logs/training_log.jsonl",
        epochs=3,
        train_batch_size=16,
        eval_batch_size=32,
        learning_rate=0.001,
        logging_steps=50,
        save_steps=2605,
        eval_steps=2605,
        max_grad_norm=1.0,
        margin=0.5,
        p=2
    )
    
    train(
        model,
        optimizer=torch.optim.Adam(model.parameters(), lr=training_args.learning_rate),
        dataset_train=dataset_train,
        dataset_validation=dataset_validation,
        training_args=training_args
    )
    

if __name__ == "__main__":
    main()