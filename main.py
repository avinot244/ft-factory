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
        output_dir="output/champion_embedding_v6/",
        logging_path="output/champion_embedding_v6/training_log_v13.jsonl",
        epochs=10,
        train_batch_size=8,
        eval_batch_size=8,
        learning_rate=5e-5,
        logging_steps=10,
        eval_steps=200,
        weight_decay=1e-2,
        margin=0.5,
        p=2,
        temperature=0.02,
        use_sigreg=True
    )
    
    train(
        model,
        optimizer=torch.optim.Adam(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay),
        dataset_train=dataset_train,
        dataset_validation=dataset_validation,
        training_args=training_args
    )
    

if __name__ == "__main__":
    main()