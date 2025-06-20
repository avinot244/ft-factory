Here is an overview of a project that I'm building. I'm seeking your advice and critical mind.
Please everytime try to be critical and only validate facts and claims if you deem it relevant and strongly founded on data and relevant observations

My goal is to build an embedding model for League of Legends champions (i.e. champion names).
As I was browing through the MTEB leaderboard I noticed that a vast majority of the top models were decoder-based model, i.e. model that "extract" only the embedding layers of decoder-based LLMs

# Corpus building
So in order to have a strong embedding model about League of Legends I've built a corpus containing a total of 1,907,871 tokens (1,526,364 for training and 381,507 for validation). This dataset contains the following : 
- Almost all the content of the official League of Legend Wiki (Champions, items, runes, epic monsters, etc...)
- Match up and gameplay strategies tailored on each champions from a coaching website (mobalytics)
- Content about genral League of Legends rationales from a League of Legends specific forum. Only some specific content.
- Data Augmentation
    - Paraphrasing of all the previous entries
    - Champion card : descrip each champions kit and its strategical synergies, strengths and weaknesses
    - Champion match up : from the data of mobalytics I've generated some match up strategies to specifically play against said champion strenghts/weaknesses
    - Champion role : from the data of the wiki, I generated a detailed analysis of the champions kit in order to dertemine where the champion supposed to be played
When building this corpus I made sure to take only non patch dependent data. For example I focus and the key mechanics of champions rather than knowing the exact mana cost of the abilities that might be suggest to change.
Also when performing paraphrasing I made sure to only do it once for each data entry and using a much more bigger model (Claude-3.5-Haiku) so the phrasing would be much more different and some better knowledge would be outputed.

# Causal Language Modeling fine-tuning
In order to train an LLM on this task I've chosen to fine-tune using LoRA the llama-3.2-1B model and tokenizer from meta-ai on the Causal Language Modeling task (CLM). Please note that the original model is **not an instruct model**. After some trials and errors I came up with this training configuration : 
For LoRA :
```python
lora_config = LoraConfig(
    r = 8,
    lora_alpha=16,
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
    num_train_epochs=3, #After testing it was the best epoch value before overfitting
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
# Instruct tuning corpus building
Afterward I then created an instruct corpus of 1.7 million traning tokens (2.1 millions totals) that is made to instruct the fine-tuned model to better retrieve data from his knowledge accumulated during CLM fine-tuning, better chatting behavior and better reasoning behavior. These QA pairs were generated using a way bigger model (Claude-3.5-Haiku) so that the reasoning could add additional knowledge.
I've constructed the following dataset with 3 different type of instruct tuning :
- factual instruct : basic factual question answer ranging from easy to hard difficulty. The whole 1.9 million CLM pre-training corpus is there covered
- strategic instruct : some QA pairs that requires reasoning on the strategic aspects of the game (champion mechanics/counter strategies, items and runes)
- role istruct : some QA pairs that requires reasoning on the role aspec and the purpose of the champion playstyle
Here is a sample from each type there : 
factual :
```json
[ { "role": "user", "content": "How does Teemo's Toxic Shot ability modify his basic attacks?" }, { "role": "assistant", "content": "Teemo's Toxic Shot transforms his basic attacks with poison mechanics:\n\n- Adds bonus magic damage on hit\n- Inflicts continuous magic damage over time\n- Subsequent attacks can refresh the poison duration\n\nThis creates sustained damage potential through each attack." } ]
```

strategic :
```json
[ { "role": "user", "content": "As a top lane player facing a Riven with high mobility, what specific strategies can you employ to consistently interrupt her engage potential and minimize her trading effectiveness?" }, { "role": "assistant", "content": "To counter Riven's mobility and trading potential, implement these strategic approaches:\n\n1. Champion Selection: Choose champions with reliable point-and-click crowd control like Pantheon or Renekton. Their targeted abilities can consistently interrupt Riven's Broken Wings and Valor mobility skills.\n\n2. Positioning: Maintain a positioning just outside her Broken Wings range, forcing her to overextend if she wants to engage. This creates opportunities to punish her aggressive attempts.\n\n3. Ability Tracking: Carefully monitor her Valor (E) shield cooldown. When it's unavailable, she becomes significantly more vulnerable to trades and all-ins.\n\n4. Itemization: Rush defensive items like Plated Steelcaps to reduce her auto-attack damage and consider Silvermere Dawn to cleanse her crowd control, neutralizing her burst potential.\n\nBy combining these approaches, you can systematically reduce Riven's lane dominance and create consistent counterplay opportunities." } ]
```

role :
```json
[ { "role": "user", "content": "How do different mythic item choices and secondary item builds fundamentally alter Heimerdinger's mid-lane effectiveness, and what situational factors should players consider when constructing their optimal item path?" }, { "role": "assistant", "content": "Mythic Item Optimization Strategies:\n\n1. Liandry's Anguish:\n- Maximizes turret damage over time\n- Ideal against tanky compositions\n- Synergizes with area control playstyle\n\n2. Night Harvester:\n- Provides burst potential\n- Enhances pick-off capabilities\n- Better against squishy, mobile compositions\n\n3. Cosmic Drive:\n- Increases mobility and ability haste\n- Supports kiting and repositioning\n- Beneficial in extended team fights\n\nSecondary Considerations:\n- Zhonya's Hourglass for survivability\n- Morellonomicon against healing-heavy teams\n- Void Staff for magic resistance penetration\n\nSituational Adaptation Factors:\n- Enemy team composition\n- Current game state\n- Personal playstyle preferences\n\nOptimal itemization requires dynamic decision-making, transforming Heimerdinger from a lane specialist to a versatile team fight controller." } ]
```

# Instruct tuning fine-tuning
With this fresh new dataset I then perform full fine-tuning on the merged CLM LoRA fine-tuned model with the base model. I've also tried to directly fine-tune the base llama3.2:1B model just to do some test.
Here is the training set-up : 
```py
training_args = SFTConfig(
    output_dir="./results/LoLlama-3.2-1B-lora-3ep-v3-instruct",
    num_train_epochs=3,
    per_device_train_batch_size=2, 
    per_device_eval_batch_size=2, 
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    logging_steps=5,
    packing=False,
    gradient_checkpointing=True,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
)
```
And the tokenizer here is the tokenizer from the llama3.2:1B **instruct** model to have the chat template.

When comparing the loss and the mean_token_accuracy when training the base_model and the pre-trained one, I've noticed that the pre-trained one had way better loss stability and performance over few thousands step than the base_model

Here are some samples of interactions that I had with this instruct model :
```markdown
**Question :**
What positioning strategies should players use against Gragas's Body Slam?

**Answer :**
Players should:

- Maintain distance when Gragas attempts Body Slam
- Use terrain and choke points to limit his engagement
- Avoid direct confrontations that enable his all-in potential
- Prevent him from isolating and eliminating targets
- Coordinate team positioning to neutralize his engage potential

**Key Insight**: Proper positioning can significantly reduce Gragas's ability to initiate and control team fights.

---

**Question :**
Explore the theoretical maximum defensive potential of Sterak's Gage when combined with champions possessing inherent health-scaling or shield-amplifying abilities, detailing complex interaction mechanics.

**Answer :**
Theoretical Maximum Sterak's Gage Interaction:

1. Champion Health Scaling Mechanics:
- Champions like Malphite, Leona, Nautilus
- Ability to generate shield-based health pools
- Potential for exponential defensive scaling

2. Shield Amplification Interactions:
- Champions with inherent shield generation
- Potential for multi-layered defensive scaling
- Revealing complex defensive optimization strategies

3. Theoretical Maximum Potential:
- 100% shield amplification
- 100% health scaling
- Potential for 100-200% defensive transformation

4. Complex Interaction Scenarios:
- Champions with hybrid shield/heal mechanics
- Potential for precise defensive optimization
- Revealing advanced mechanical skill levels

Theoretical maximum Sterak's Gage represents a pinnacle of defensive design, capable of dramatically shifting traditional defensive scaling paradigms. It represents a higher order of defensive design, transcending simple statistical optimization. The item's true value lies not just in raw defensive stats, but in its ability to fundamentally reshape how defensive capabilities are perceived and exploited in team compositions.

```

As you can see the model is behaving quite well but lacks deep knowledge, it's only "on the surface" reasoning

# Benchmarking on champion embedding
In order to assess if these 2 previous training step where efficient I generated a dataset (available here : [avinot/Champion-Similarity](https://huggingface.co/datasets/avinot/Champion-Similarity)) of champion triplets where the anchor and positive are champions that share a the same role and class (and sub-class) and the negative doesn't share a role and class with the anchor. But comparing the cosine similarities of the embedding using the last embedding layer of the instruct tuned model, I only had a 0.5 of accuracy (good prediction = cos_sim(anchor, positive) > cos_sim(anchor, negative)). Also I've taken into account that the same champion could be flexed on different roles/classes. (ex: Gragas can be Top, Jungle, Mid and Support, and could be a burst and Vanguard)

I also did a top_k_retrieval where I get for every champion the k nearest champions based on the cosine similarity of their embedding, but still the results weren't that great.

# Extracting embeding layers of decoder and perform contrastive fine-tuning
After that I've take the mean-pooling of the last 10 hidden layers of my backbone LLM and appended on top of it a linear prediction head. In the process I've frozen the parameters of the backbone LLM (instruct fine-tuned one) and only trained the linear prediction head.
During training I took the TripletMarginLoss from pytorch. And the prediction head was trained on the previously generated triplet dataset : [avinot/Champion-Similarity](https://huggingface.co/datasets/avinot/Champion-Similarity) that has 236k training triplets and 59k validation triplets.
The training and validation loss showed a steady mitigation until reaching approximately 0 and not spiking afterward on 3 epochs.
After that I've benchmarked my embedding model on wether he could accuractly tell if the anchor is closer to the positive than the negative. And it had 100% accuracy on the validation set

However I've tested only fine-tuning a sentence transformer model on the same dataset but on the MultipleRerankingLoss loss function and managed to have similar results.

Here is some overview of the cosine-similarity between champions :
```
- Thresh-Lulu : 0.80 (same role different class, not in training or validation set)
- Thresh-Janna : 0.93 (same role different class, not in training or validation set)
- Renekton-Aatrox : 0.91 (same role and class, in train set)
- Renekton-Orianna : 0.16 (not same role and class, in train set)
```

My intuition is that the dataset is not "complex enough" so that this task requires the representation of the backbone insstruct tuned LLM. For example we could have :
A : Karthus, P : Swain and N : Lux. This triplets tends to show similarity between sustained AoE dps mages.
A : Quinn, P : Bard and N : Caitlyn. This triplets shows champions that have roaming capabilities.