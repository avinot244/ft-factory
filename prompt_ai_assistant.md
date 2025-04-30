Here is an overview of a project that I'm building. I'm seeking your advice and critical mind.
Please everytime try to be critical and only validate facts and claims if you deem it relevant and strongly founded

My goal is to build an embedding model for League of Legends champions.
As I was browing through the MTEB leaderboard I noticed that all top models were decoder-based model, i.e. model that "extract" only the embedding layers of decoder-based LLMs

# Corpus building
So in order to have a strong embedding model about League of Legends I've built a corpus containing a total of 1,907,871 tokens (1,526,364 for training and 381,507 for validation). This dataset contains the following : 
- Almost all the content of the official League of Legend Wiki (Champions, items, runes, epic monsters, etc...)
- Match up and gameplay strategies tailored on each champions from a coaching website (mobalytics)
- Content about genral League of Legends rationales from a League of Legends specific forum
- Data Augmentation
    - Paraphrasing of all the previous entries
    - Champion card : descrip each champions kit and its strategical synergies, strengths and weaknesses
    - Champion match up : from the data of mobalytics I've generated some match up strategies to specifically play against said champion strenghts/weaknesses
    - Champion role : from the data of the wiki, I generated a detailed analysis of the champions kit in order to dertemine where the champion supposed to be played

# Causal Language Modeling fine-tuning
In order to train an LLM on this task I've chosen to fine-tune using LoRA the llama-3.2-1B model from meta-ai on the Causal Language Modeling task (CLM). Please note that the original model is **not an instruct model**. After some trials and errors I came up with this training configuration : 
For LoRA :
```python
lora_config = LoraConfig(
    r = 8,
    lora_alpha=16,
    # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
```

For the training :
```python
trainings_args = TrainingArguments(
    output_dir=f"./results/{model_name}",
    eval_strategy="epoch",
    learning_rate=1e-5,
    num_train_epochs=epochs,
    weight_decay=0.01,
    push_to_hub=True,
    hub_token=get_hf_token("write"),
    # Memory optimization settings
    per_device_train_batch_size=2,  # Reduce batch size to save memory
    per_device_eval_batch_size=2,
    fp16=True,  # Enable mixed precision training
    gradient_accumulation_steps=4  # Accumulate gradients to simulate larger batch
)
```

# Follow-up plan
After a session of brainstorming with an AI assistant I came up with the following plan : 
1. Generate a QA instruct dataset about the knowledge of all the 1.9 million tokens of my corpus
2. Instruct fine tune my pre-trained model on this QA dataset
3. Extract the embedding layers
4. Fine-tune the embedding layers on a triplet dataset (anchor, positive, negative) where the anchor and positive are both champions name that play the same role (Toplane, Jungle, Midlane, ADC or Support) or have the same class (Marksman, Assassin, Juggernaut, Warder, ...). This final training is inspired by the way that "traditional" sentence-transformers models are trained