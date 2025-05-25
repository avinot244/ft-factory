import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os

from utils.types.TrainingArgs import ContrastiveTrainingArgs

def train(
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    dataset_train : torch.utils.data.Dataset,
    dataset_validation : torch.utils.data.Dataset,
    training_args : ContrastiveTrainingArgs,
):
    loss_fn = torch.nn.TripletMarginLoss(margin=training_args.margin, p=training_args.p)
    dataloader_train = DataLoader(dataset_train, batch_size=training_args.train_batch_size, shuffle=True)
    dataloader_validation = DataLoader(dataset_validation, batch_size=training_args.eval_batch_size, shuffle=True)
    for epoch in range(training_args.epochs):
        for (step, (anchor_batch, positive_batch, negative_batch)) in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
            # Forward pass
            anchor_embeddings = model(anchor_batch)
            positive_embeddings = model(positive_batch)
            negative_embeddings = model(negative_batch)
            
            # Compute the loss
            cost = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            
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
            
            
            if step % training_args.save_steps == 0:
                # Save the model checkpoint
                torch.save(model.state_dict(), f"{training_args.output_dir}model_epoch_{epoch+1}_step_{step+1}.pth")
                
            if step % training_args.eval_steps == 0:
                # Evaluating the model on the validation set
                tqdm.write("Evaluating on validation set...")
                for (anchor_val, positive_val, negative_val) in tqdm(dataloader_validation):
                    with torch.no_grad():
                        anchor_embeddings_val = model(anchor_val)
                        positive_embeddings_val = model(positive_val)
                        negative_embeddings_val = model(negative_val)
                        val_cost = loss_fn(anchor_embeddings_val, positive_embeddings_val, negative_embeddings_val)

                tqdm.write(f"Validation Loss: {val_cost.item():.4f}")
                with open(training_args.logging_path, "a") as log_file:
                    json.dump({
                        "epoch": epoch,
                        "step": step + epoch*len(dataloader_train),
                        "validation_loss": val_cost.item()
                    }, log_file)
                    log_file.write("\n")
        
        print(f"Epoch {epoch+1}/{training_args.epochs}, Loss: {cost.item():.4f}")
        torch.save(model.state_dict(), f"{training_args.output_dir}model_epoch_{epoch+1}_step_{step+1}.pth")