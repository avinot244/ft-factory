from pydantic import BaseModel

class ContrastiveTrainingArgs(BaseModel):
    """
    Base class for training arguments for cotnrastive learning.
    """
    output_dir: str
    logging_path: str
    epochs: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    logging_steps: int
    save_steps: int
    eval_steps: int
    max_grad_norm: float
    epochs: int
    margin: float
    p: int