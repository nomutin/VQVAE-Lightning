"""
Networks for the VQ-VAE model.

References
----------
- https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb.
"""

from torch import Tensor, nn


def conv_3x3(in_channels: int, out_channels: int) -> nn.Conv2d:
    """Convolutional layer with keeping the spatial resolution."""
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        padding=1,
    )


def conv_1x1(in_channels: int, out_channels: int) -> nn.Conv2d:
    """Convolutional layer with reducing the spatial resolution."""
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
    )


def downsample(in_channels: int, out_channels: int) -> nn.Conv2d:
    """Downsample layer."""
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
    )


def upsample(in_channels: int, out_channels: int) -> nn.ConvTranspose2d:
    """Upsample layer."""
    return nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
    )


class ResidualStack(nn.Module):
    """Residual stack module."""

    def __init__(
        self,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        activation_name: str,
    ) -> None:
        super().__init__()
        activation = getattr(nn, activation_name)
        layers: list[nn.Sequential] = []
        for _ in range(num_residual_layers):
            layer = nn.Sequential(
                activation(),
                conv_3x3(num_hiddens, num_residual_hiddens),
                nn.BatchNorm2d(num_residual_hiddens),
                activation(),
                conv_1x1(num_residual_hiddens, num_hiddens),
            )
            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        self.last_activation: nn.Module = getattr(nn, activation_name)()

    def forward(self, x: Tensor) -> Tensor:
        """Forward method."""
        h = x
        for layer in self.layers:
            h = h.add(layer(h))
        return self.last_activation.forward(h)


class Encoder(nn.Module):
    """Encoder module."""

    def __init__(
        self,
        *,
        num_hiddens: int,
        num_downsampling_layers: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        activation_name: str,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        activation = getattr(nn, activation_name)
        layers: list[nn.Module] = []
        layers.extend((downsample(3, num_hiddens // 2), nn.BatchNorm2d(num_hiddens // 2), activation()))
        layers.extend((downsample(num_hiddens // 2, num_hiddens), nn.BatchNorm2d(num_hiddens), activation()))
        for _ in range(num_downsampling_layers - 2):
            layers.extend((downsample(num_hiddens, num_hiddens), nn.BatchNorm2d(num_hiddens), activation()))
        layers.append(conv_3x3(num_hiddens, num_hiddens))

        self.conv = nn.Sequential(*layers)
        self.residual_stack = ResidualStack(
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            activation_name=activation_name,
        )
        self.last_activation = getattr(nn, activation_name)()
        self.embedding = conv_3x3(num_hiddens, embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward method."""
        h = self.conv(x)
        h = self.residual_stack(h)
        h = self.last_activation(h)
        return self.embedding.forward(h)


class Decoder(nn.Module):
    """Decoder module."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        num_hiddens: int,
        num_upsampling_layers: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        activation_name: str,
    ) -> None:
        super().__init__()
        activation = getattr(nn, activation_name)
        self.conv = conv_3x3(embedding_dim, num_hiddens)
        self.residual_stack = ResidualStack(
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            activation_name=activation_name,
        )
        layers: list[nn.Module] = []
        for _ in range(num_upsampling_layers - 2):
            layers.extend((upsample(num_hiddens, num_hiddens), nn.BatchNorm2d(num_hiddens), activation()))
        layers.extend((upsample(num_hiddens, num_hiddens // 2), nn.BatchNorm2d(num_hiddens // 2), activation()))
        layers.append(upsample(num_hiddens // 2, 3))
        self.upconv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward method."""
        h = self.conv.forward(x)
        h = self.residual_stack.forward(h)
        return self.upconv(h)
