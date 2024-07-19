"""VQ-VAE model implementation."""

from lightning import LightningModule
from torch import Tensor
from vector_quantize_pytorch import FSQ, VectorQuantize

from vqvae_lightning.networks import Decoder, Encoder


class AEBase(LightningModule):
    """Base class for autoencoder models."""

    def encode(self, x: Tensor) -> Tensor:
        """Encode image tensor into latent space."""
        raise NotImplementedError

    def decode(self, x: Tensor) -> Tensor:
        """Reconstruct image tensor from latent space."""
        raise NotImplementedError

    def shared_step(self, batch: tuple[Tensor, ...]) -> dict[str, Tensor]:
        """Shared training/validation step."""
        raise NotImplementedError

    def training_step(self, batch: tuple[Tensor, ...], **_: str) -> dict[str, Tensor]:
        """Rollout training step."""
        loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict

    def validation_step(self, batch: tuple[Tensor, ...], _: int) -> dict[str, Tensor]:
        """Rollout validation step."""
        loss_dict = self.shared_step(batch)
        loss_dict = {"val_" + k: v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict


class VQVAE(AEBase):
    """Vector Quantized Variational Autoencoder(VQ-VAE) model."""

    def __init__(
        self,
        *,
        codebook_size: int,
        alpha: float,
        embedding_dim: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        num_downsampling_layers: int,
        num_upsampling_layers: int,
        activation_name: str,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.codebook_size = codebook_size
        self.alpha = alpha
        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            num_downsampling_layers=num_downsampling_layers,
            activation_name=activation_name,
        )
        self.decoder = Decoder(
            embedding_dim=embedding_dim,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            num_upsampling_layers=num_upsampling_layers,
            activation_name=activation_name,
        )
        self.vq = VectorQuantize(
            dim=embedding_dim,
            accept_image_fmap=True,
            codebook_size=codebook_size,
        )

    def encode(self, x: Tensor) -> Tensor:
        """Encode image tensor into latent space."""
        feature_map = self.encoder.forward(x)
        return self.vq.forward(feature_map)[0]

    def decode(self, x: Tensor) -> Tensor:
        """Reconstruct image tensor from latent space."""
        return self.decoder.forward(x)

    def shared_step(self, batch: tuple[Tensor, ...]) -> dict[str, Tensor]:
        """Shared training/validation step."""
        image, *_ = batch
        feature_map = self.encoder.forward(image)
        quantized, _, commitment_loss = self.vq.forward(feature_map)
        commitment_loss = self.alpha * commitment_loss
        reconstruction = self.decoder.forward(quantized)
        reconstruction_loss = (reconstruction - image).abs().mean()
        return {
            "loss": reconstruction_loss + commitment_loss,
            "reconstruction_loss": reconstruction_loss,
            "commitment_loss": commitment_loss,
        }


class FSQVQVAE(AEBase):
    """Vector Quantized Variational Autoencoder(VQ-VAE) with FSQ."""

    def __init__(
        self,
        *,
        levels: tuple[int, ...],
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        num_downsampling_layers: int,
        num_upsampling_layers: int,
        activation_name: str,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(
            embedding_dim=len(levels),
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            num_downsampling_layers=num_downsampling_layers,
            activation_name=activation_name,
        )
        self.decoder = Decoder(
            embedding_dim=len(levels),
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            num_upsampling_layers=num_upsampling_layers,
            activation_name=activation_name,
        )
        self.vq = FSQ(levels=list(levels))

    def encode(self, x: Tensor) -> Tensor:
        """Encode image tensor into latent space."""
        feature_map = self.encoder.forward(x)
        return self.vq.forward(feature_map)[0]

    def decode(self, x: Tensor) -> Tensor:
        """Reconstruct image tensor from latent space."""
        return self.decoder.forward(x)

    def shared_step(self, batch: tuple[Tensor, ...]) -> dict[str, Tensor]:
        """Shared training/validation step."""
        image, *_ = batch
        feature_map = self.encode(image)
        reconstruction = self.decode(feature_map)
        reconstruction_loss = (reconstruction - image).abs().mean()
        return {"loss": reconstruction_loss}
