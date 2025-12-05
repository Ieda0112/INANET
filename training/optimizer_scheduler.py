import torch

from concern.config import Configurable, State


class OptimizerScheduler(Configurable):
    optimizer = State()
    optimizer_args = State(default={})
    learning_rate = State(autoload=False)

    def __init__(self, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.load('learning_rate', cmd=cmd, **kwargs)
        if 'lr' in cmd:
            self.optimizer_args['lr'] = cmd['lr']

    def create_optimizer(self, parameters):
        # Convert optimizer_args values to appropriate numeric types
        # (YAML may parse scientific notation like 3e-4 as strings)
        cleaned_args = {}
        for key, value in self.optimizer_args.items():
            if isinstance(value, str):
                try:
                    # Try to convert string to float
                    cleaned_args[key] = float(value)
                except ValueError:
                    # If conversion fails, keep original value
                    cleaned_args[key] = value
            else:
                cleaned_args[key] = value
        
        optimizer = getattr(torch.optim, self.optimizer)(
                parameters, **cleaned_args)
        if hasattr(self.learning_rate, 'prepare'):
            self.learning_rate.prepare(optimizer)
        return optimizer
