# import packages
import math
import os
import torch
"""
Handles:
- saving best model checkpoint
- saving last checkpoint every epoch
- early stopping
"""

class ModelCheckpoint:
    def __init__(
        self,
        checkpoint_dir,
        monitor="val_loss",
        mode="min",
        patience=10,
        min_delta=0.0,
        save_best_after_epoch=0,
        verbose=True,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.save_best_after_epoch = save_best_after_epoch
        self.verbose = verbose

        # create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.mode not in ["min", "max"]:
            raise ValueError("mode must be 'min' or 'max'")

        # intialise best score to worse value 
        self.best_score = math.inf if mode == "min" else -math.inf
        self.best_epoch = -1        # track each epoch to identify the best score
        self.num_bad_epochs = 0     # counts consecutive epochs without improvement
        self.should_stop = False    # flag to signal early stoppping

    # check if there is improvement over the best
    def is_improvement(self, current):
        if self.mode == "min":
            return current < (self.best_score - self.min_delta)
        else:
            return current > (self.best_score + self.min_delta)

    # evaluate the monitored metrics, for saving and checking for early stop
    def step(self, epoch, metrics, model, optimiser=None):
        if self.monitor not in metrics:
            raise KeyError(f"Monitored metric '{self.monitor}' not found in metrics")

        current = metrics[self.monitor]
        improved = self.is_improvement(current)

        # always save last checkpoint
        last_path = os.path.join(self.checkpoint_dir, "last_model.pt")
        self._save_checkpoint(
            path=last_path,
            epoch=epoch,
            model=model,
            optimiser=optimiser,
            metrics=metrics,
            is_best=False,
        )

        if improved:
            # update best tracking sttae
            self.best_score = current
            self.best_epoch = epoch
            self.num_bad_epochs = 0     # reset patience counter on improvement

            if epoch >= self.save_best_after_epoch:
                # save best model only after warmup 
                best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                self._save_checkpoint(
                    path=best_path,
                    epoch=epoch,
                    model=model,
                    optimiser=optimiser,
                    metrics=metrics,
                    is_best=True,
                )
                if self.verbose:
                    print(f"[Checkpoint] New best model saved at epoch {epoch} | {self.monitor} = {current:.6f}")
            else:
                # impovement detected 
                if self.verbose:
                    print(f"[Checkpoint] Metric improved at epoch {epoch}, but best saving starts from epoch {self.save_best_after_epoch}")
        else:
            # no improvement
            self.num_bad_epochs += 1
            if self.verbose:
                print(f"[Checkpoint] No improvement in {self.monitor}. Patience: {self.num_bad_epochs}/{self.patience}")
        
        # trigger early stopping
        if self.num_bad_epochs >= self.patience:
            self.should_stop = True
            if self.verbose:
                print(f"[EarlyStopping] Stopping triggered at epoch {epoch}")

        return improved

    # save mdoel state and training metadata to pt file
    def save_checkpoint(self, path, epoch, model, optimiser, metrics, is_best=False):
        checkpoint = {
            "epoch": epoch,                            
            "model_state_dict": model.state_dict(),      # learned model weights
            "metrics": metrics,                          # metric values at this epoch
            "is_best": is_best,                          # flag indicateing best checkpoint
            "monitor": self.monitor,                     # metric to be monitored
            "best_score": self.best_score,               # best score seen
            "best_epoch": self.best_epoch,               # epoch that produced best score
        }

        # save optimiser state for full training resuming
        if optimiser is not None:
            checkpoint["optimiser_state_dict"] = optimiser.state_dict()

        torch.save(checkpoint, path)