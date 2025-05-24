from pydantic import BaseModel

class ContrastiveTrainingArgs(BaseModel):
    """
    Base class for training arguments for cotnrastive learning.
    """
    output_dir: str
    logging_path: str
    epochs: int=10
    batch_size: int=10
    learning_rate: float=0.001
    logging_steps: int=100
    save_steps: int=500
    eval_steps: int=500
    max_grad_norm: float=1.0
    epochs: int=3
    margin: float=0.1
    p: int=2