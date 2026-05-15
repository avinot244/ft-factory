import torch
from services.sentence_transformer.contrastive.model_loader import PredictionHead
from services.sentence_transformer.contrastive.data_loader import ChampionSimilarityDataset
from services.sentence_transformer.contrastive.trainer_loader import train, ContrastiveTrainingArgs
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def main():
    model = PredictionHead()
    
    training_args = ContrastiveTrainingArgs(
        # Paths
        output_dir = "output/c2cp_bge",
        logging_path = "logs/c2cp_bge/c2cp_bge_metrics.jsonl",

        # Training hyper-parameters
        epochs = 10,
        train_batch_size = 8,
        eval_batch_size = 8,
        learning_rate = 3e-4,
        weight_decay = 1e-2,

        # Loss hyper-parameters
        temperature = 0.07,   # NCE temperature
        margin = 0.5,    # triplet margin
        p = 2,      # Lp distance for triplet
        use_sigreg = True,   # add SIGReg uniformity term

        # Logging / checkpointing
        logging_steps = 20,   # train log frequency (steps)
        eval_steps = 100,  # validation frequency (steps)

        # Model architecture
        output_dim = 256,
        hidden_dim = 512,
        bottleneck_dim = 256,
        dropout = 0.1,
        freeze_backbone = True,

        # HuggingFace dataset cache
        hf_cache_dir = None,

        # Device
        device = "cuda"
    )
    train(
        model=model,
        dataset_train=ChampionSimilarityDataset(split="train"),
        dataset_validation=ChampionSimilarityDataset(split="validation"),
        args=training_args,
    )
    

if __name__ == "__main__":
    main()