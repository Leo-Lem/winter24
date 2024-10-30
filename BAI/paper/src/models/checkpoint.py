from torch import save, load
from torch.nn import Module


class CheckpointModule(Module):
    """ A module that can save and load checkpoints. """

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def save_checkpoint(self, checkpoint: int):
        """ Save the model at a checkpoint. """
        save({
            "checkpoint": checkpoint,
            "state": self.state_dict()
        }, self.path)

        print(f"[Save] To {self.path} at checkpoint {checkpoint}.")

    def load_checkpoint(self) -> int:
        """ Load the model from the last checkpoint. """
        try:
            checkpoint = load(self.path, weights_only=True)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['state'].items()
                               if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)

            self.load_state_dict(model_dict)

            print(
                f"[Load] From {self.path} at checkpoint {checkpoint['checkpoint']}.")

            return checkpoint['checkpoint']
        except FileNotFoundError:
            return 0
