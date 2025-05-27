import json
import matplotlib.pyplot as plt
import seaborn as sns

with open("./logs/training_log.jsonl", "r") as f:
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
        if "validation_loss" in list(data.keys()) and i%2 == 0:
            validation_loss.append(data["validation_loss"])
            steps_validation.append(data["step"])
    
    sns.lineplot(x=steps_training, y=training_loss, label="Training Loss")
    sns.lineplot(x=steps_validation, y=validation_loss, label="Validation Loss", marker='o')
    plt.title("Training and Validation Loss Over Steps")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    
    # plt.plot(steps_training, training_loss, label="Training Loss")
    # plt.plot(steps_validation, validation_loss, "ro", label="Validation Loss")
    # plt.xlabel("Steps")
    # plt.ylabel("Loss")
    # plt.title("Training and Validation Loss Over Steps")
    # plt.legend()
    plt.savefig("training_validation_loss.png")