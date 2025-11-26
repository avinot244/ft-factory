import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
import os

version : str = "v11"
log_path : str = f"./logs/training_log_{version}.jsonl"
out_path : str = f"./output/{version}/training_validation_loss"
x_interval = 10

with open(log_path, "r") as f:
    lines = f.readlines()
    
    validation_loss = []
    training_loss = []
    steps_validation = []
    steps_training = []
    
    for i, line in enumerate(lines):
        data : dict = json.loads(line)
        if "loss" in list(data.keys()):
            training_loss.append(data["loss"])
            steps_training.append(data["step"])
        if "validation_loss" in list(data.keys()):
            validation_loss.append(data["validation_loss"])
            steps_validation.append(data["step"])
    
    window = max(1, len(training_loss) // 100)
    smoothed = np.convolve(training_loss, np.ones(window) / window, mode='valid')
    
    sns.lineplot(x=[i*x_interval for i in range(len(training_loss))], y=training_loss, label="Training Loss", color=(207/255, 207/255, 207/255))
    sns.lineplot(x=[i*x_interval for i in range(len(smoothed))], y=smoothed, label="Smoothed Training Loss", color="blue")
    sns.lineplot(x=steps_validation, y=validation_loss, label="Validation Loss", marker='o')
    plt.title("Training and Validation Loss Over Steps")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(out_path, dpi=200)

pos_sims = []
neg_sims = []
with open("temp.csv", "r") as f:
    reader = csv.reader(f)
    for i, line in enumerate(reader):
        if i == 0:
            continue
        step = i * x_interval
        cos_sim_pos = float(line[1])
        cos_sim_neg = float(line[2])
        neg_sims.append(cos_sim_neg)
        pos_sims.append(cos_sim_pos)
        


# plot similarities
plt.clf()
sns.lineplot(x=[i * x_interval for i in range(len(pos_sims))], y=pos_sims, label="Positive (raw)", color="tab:green", alpha=0.35)
sns.lineplot(x=[i * x_interval for i in range(len(neg_sims))], y=neg_sims, label="Negative (raw)", color="tab:red", alpha=0.35)
smoothed_pos = np.convolve(pos_sims, np.ones(window) / window, mode='valid')
smoothed_neg = np.convolve(neg_sims, np.ones(window) / window, mode='valid')
sns.lineplot(x=[i * x_interval for i in range(len(smoothed_pos))], y=smoothed_pos, label="Positive (smoothed)", color="darkgreen")
sns.lineplot(x=[i * x_interval for i in range(len(smoothed_neg))], y=smoothed_neg, label="Negative (smoothed)", color="darkred")

plt.xlabel("Steps")
plt.ylabel("Cosine Similarity")
plt.title("Evolution of Positive vs Negative Cosine Similarities")
plt.ylim(-1.05, 1.05)
plt.legend()
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path + "_cosine.png", dpi=200)