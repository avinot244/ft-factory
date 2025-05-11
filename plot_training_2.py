import json
import matplotlib.pyplot as plt

with open("./instruct-ft-1.jsonl", "r") as f:
    lines = f.readlines()

eval_history : list[dict] = list()
epochs : list[float] = list()
train_history : list[dict] = list()

for line in lines:
    data : dict = json.loads(line)
    epochs.append(data["epoch"])
    if "eval_loss" in list(data.keys()):
        eval_history.append(data)
    else:
        train_history.append(data)
        
plt.plot([d["epoch"] for d in train_history], [d["mean_token_accuracy"] for d in train_history], "-b", label="train_mean_token_accuracy")
plt.plot([d["epoch"] for d in eval_history], [d["eval_mean_token_accuracy"] for d in eval_history], "-r", label="eval_mean_token_accuracy")
plt.xlabel("epoch")
plt.ylabel("mean_token_accuracy")
plt.legend()
plt.title("Training History from base model llama3.2:1B")
plt.savefig("temp.png")