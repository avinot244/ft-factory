import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
import torch.nn.functional as F
from typing import Optional

from utils.types.TrainingArgs import ContrastiveTrainingArgs
from services.huggingface.contrastive.loss import NCELoss

def eval_model(
    dataloader_train : torch.utils.data.DataLoader,
    dataloader_validation : torch.utils.data.DataLoader,
    model : torch.nn.Module,
    loss_fn : NCELoss,
    training_args : ContrastiveTrainingArgs,
    epoch : int,
    step: int,
):
    for (anchor_val, positive_val, negative_val) in tqdm(dataloader_validation):
        with torch.no_grad():
            anchor_embeddings_val = model(anchor_val)
            positive_embeddings_val = model(positive_val)
            negative_embeddings_val = model(negative_val)
            
            triplet_loss = torch.nn.functional.relu(
                F.pairwise_distance(anchor_embeddings_val, positive_embeddings_val, p=training_args.p) -
                F.pairwise_distance(anchor_embeddings_val, negative_embeddings_val, p=training_args.p) +
                training_args.margin
            ).mean()
            
            val_cost = loss_fn(anchor_embeddings_val, positive_embeddings_val, negative_embeddings_val) + 0.5 * triplet_loss

    tqdm.write(f"Validation Loss: {val_cost.item():.4f}")
    with open(training_args.logging_path, "a") as log_file:
        json.dump({
            "epoch": epoch,
            "step": step + epoch*len(dataloader_train),
            "validation_loss": val_cost.item()
        }, log_file)
        log_file.write("\n")
    return val_cost.item()

def train(
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    dataset_train : torch.utils.data.Dataset,
    dataset_validation : torch.utils.data.Dataset,
    training_args : ContrastiveTrainingArgs,
):
    # loss_fn = torch.nn.TripletMarginLoss(margin=training_args.margin, p=training_args.p)
    loss_fn = NCELoss(temperature=training_args.temperature)
    dataloader_train = DataLoader(dataset_train, batch_size=training_args.train_batch_size, shuffle=True)
    dataloader_validation = DataLoader(dataset_validation, batch_size=training_args.eval_batch_size, shuffle=True)
    best_validation_loss : float = 1
    for epoch in range(training_args.epochs):
        for (step, (anchor_batch, positive_batch, negative_batch)) in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
            # Forward pass
            anchor_embeddings = model(anchor_batch)
            positive_embeddings = model(positive_batch)
            negative_embeddings = model(negative_batch)
            
            # Compute the loss
            triplet_loss = torch.nn.functional.relu(
                F.pairwise_distance(anchor_embeddings, positive_embeddings, p=training_args.p) -
                F.pairwise_distance(anchor_embeddings, negative_embeddings, p=training_args.p) +
                training_args.margin
            ).mean()
            cost = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings) + 0.5 * triplet_loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
            if step % training_args.logging_steps == 0:
                tqdm.write(f"Epoch {epoch}, Step {step}, Loss: {cost.item():.4f}")
                if not os.path.exists(training_args.logging_path):
                    with open(training_args.logging_path, "w") as log_file:
                        json.dump({
                            "epoch": epoch ,
                            "step": step + epoch*len(dataloader_train),
                            "loss": cost.item()
                        }, log_file)
                        log_file.write("\n")
                else:
                    with open(training_args.logging_path, "a") as log_file:
                        json.dump({
                            "epoch": epoch,
                            "step": step + epoch*len(dataloader_train),
                            "loss": cost.item()
                        }, log_file)
                        log_file.write("\n")
            if step % 10 == 0:
                with torch.no_grad():
                    cos_sim_pos = F.cosine_similarity(anchor_embeddings, positive_embeddings, dim=1).mean().item()
                    cos_sim_neg = F.cosine_similarity(anchor_embeddings, negative_embeddings, dim=1).mean().item()
                    if not(os.path.exists("./temp.csv")):
                        with open("./temp.csv", "w") as f:
                            f.write("step,cos_sim_pos,cos_sim_neg\n")
                            f.write(f"{step},{cos_sim_pos:.4f},{cos_sim_neg:.4f}\n")
                            
                    else:
                        with open("./temp.csv", "a") as f:
                            f.write(f"{step},{cos_sim_pos:.4f},{cos_sim_neg:.4f}\n")
                
            if step % training_args.eval_steps == 0:
                # Evaluating the model on the validation set
                tqdm.write("Evaluating on validation set...")
                val_loss = eval_model(dataloader_train, dataloader_validation, model, loss_fn, training_args, epoch, step)
                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    torch.save(model.state_dict(), f"{training_args.output_dir}model_epoch_{epoch+1}_step_{step+1}_{val_loss:.4f}.pth")
                    
        
        print(f"Epoch {epoch+1}/{training_args.epochs}, Loss: {cost.item():.4f}")
        val_loss = eval_model(dataloader_train, dataloader_validation, model, loss_fn, training_args, epoch, step)
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            torch.save(model.state_dict(), f"{training_args.output_dir}model_epoch_{epoch+1}_step_{step+1}_{val_loss:.4f}.pth")