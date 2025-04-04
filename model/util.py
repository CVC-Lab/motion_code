import torch
import functools
from torch.utils.tensorboard import SummaryWriter

def tensorboard_logger(log_dir="runs/default", meta="loss"):
    """Decorator that logs loss values without modifying the train loop.
        Assuming the result is given in result in train methods, it is supposed to be 
        a dictionary or list
    
    """
    def decorator(train_func):
        @functools.wraps(train_func)
        def wrapper(self, *args, **kwargs):
            writer = SummaryWriter(log_dir)  # Initialize TensorBoard writer
            
            result = train_func(self, *args, **kwargs)  # Run original train method

            # Expect train method to return loss values
            if isinstance(result, list):
                for step, loss in enumerate(result):
                    if isinstance(loss, dict):
                        writer.add_scalars(f"{meta}/train", loss, step)
                    else:
                        writer.add_scalar(f"{meta}/train", loss, step)
            if isinstance(result, dict):
                for loss_name, loss_list in result.items():
                    for step, loss in enumerate(loss_list):
                        writer.add_scalar(f"{meta}/{loss_name}", loss, step)
            writer.close()  # Close TensorBoard writer
            return result  # Return loss data for further use

        return wrapper