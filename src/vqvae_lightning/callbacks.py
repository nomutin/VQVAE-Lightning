"""Callbacks for Logging Training/Testing Results."""

from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger


class LogReconsrtuctions(Callback):
    """Callback to log image reconstructions to WandbLogger."""

    def __init__(self, every_n_epochs: int, num_samples: int) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log image reconstructions."""
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        if not isinstance(logger := trainer.logger, WandbLogger):
            return

        for stage in ("train", "val"):
            dataloader = getattr(trainer.datamodule, f"{stage}_dataloader")()
            image, *_ = next(iter(dataloader))
            image = image[: self.num_samples].to(pl_module.device)
            recon = pl_module.decode(pl_module.encode(image))
            logger.log_image(f"{stage}_inputs", list(image))
            logger.log_image(f"{stage}_reconstructions", list(recon))
