import json
import matplotlib.pyplot as plt

with open("./logs/training_log.jsonl", "r") as f:
    lines = f.readlines()
    
    validation_loss = []
    training_loss = []
    steps_validation = []
    steps_training = []
    
    for line in lines:
        data : dict = json.loads(line)
        if "loss" in list(data.keys()):
            training_loss.append(data["loss"])
            steps_training.append(data["epoch"] * data["step"])
        if "validation_loss" in list(data.keys()):
            validation_loss.append(data["validation_loss"])
            steps_validation.append(data["epoch"] * data["step"])
    
    print(steps_training)
    
    plt.plot(steps_training, training_loss, label="Training Loss")
    plt.plot(steps_validation, validation_loss, label="Validation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Steps")
    plt.legend()
    plt.savefig("training_validation_loss.png")