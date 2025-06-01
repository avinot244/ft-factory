from datasets import load_dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainingArguments, SentenceTransformerTrainer
from sentence_transformers.evaluation import TripletEvaluator, SimilarityFunction
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import TripletEvaluator

from utils.token_manager import get_hf_token

model = SentenceTransformer("all-mpnet-base-v2")
loss = MultipleNegativesRankingLoss(model)

train_dataset = load_dataset("avinot/Champion-Similarity", token=get_hf_token("read"), split="train")
eval_dataset = load_dataset("avinot/Champion-Similarity", token=get_hf_token("read"), split="validation")

args = SentenceTransformerTrainingArguments(
    output_dir="results/all-mpnet-base-v2-lolchamps",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    bf16=True,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    logging_steps=100,
)

dev_evaluator = TripletEvaluator(
    anchors=eval_dataset["anchor"],
    positives=eval_dataset["positive"],
    negatives=eval_dataset["negative"],
    main_similarity_function=SimilarityFunction.COSINE,
    name="Champion-Similarity-dev",
    show_progress_bar=True
)
print("Evaluating model")
dev_evaluator(model)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator
)

print("Training model")
trainer.train(resume_from_checkpoint=True)

test_evaluator = TripletEvaluator(
    anchors=eval_dataset["anchor"],
    positives=eval_dataset["positive"],
    negatives=eval_dataset["negative"],
    name="Champion-Similarity-test",
)
test_evaluator(model)

model.save_pretrained("results/all-mpnet-base-v2-lolchamps")

model.push_to_hub("all-mpnet-base-v2-lolchamps", token=get_hf_token("write"))