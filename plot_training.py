import matplotlib.pyplot as plt
import numpy as np
import os
import json


def load_history_from_model(model_path : str) -> dict:
    history : dict = []
    dir_list_index : list = list()
    # Traverse to the last directory in the model_path
    for dir in os.listdir(model_path):
        if os.path.isdir(os.path.join(model_path, dir)):
            dir_list_index.append(int(dir.split("-")[-1]))
    
    last_dir_index : int = sorted(dir_list_index)[-1]
    
    last_dir = os.path.join(model_path, f"checkpoint-{last_dir_index}")
    trainer_state_path = os.path.join(last_dir, "trainer_state.json")
    
    # Load the trainer_state.json file
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path, "r") as file:
            history = json.load(file)
    return history

def plot_training_history(model : str, history : list):
    loss : list[float] = []
    eval_loss : list[float] = []
    epochs : list[float] = []
    eval_epochs : list[int] = []
    for entry in history["log_history"]:
        if "loss" in entry:
            loss.append(entry["loss"])
            epochs.append(entry["epoch"])
            
        if "eval_loss" in entry:
            eval_loss.append(entry["eval_loss"])
            eval_epochs.append(entry["epoch"])

        
    
    plt.plot(epochs, loss, label="Loss")
    plt.plot(eval_epochs, eval_loss, label="Eval Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid()
    plt.savefig(f"training_loss-{model}.png")

    

history = load_history_from_model("results/LoLlama3.2-1B-lora-70ep")
plot_training_history("LoLlama3.2-1B-lora-70ep", history)