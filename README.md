# VQVAE-Lightning

Lightning implementation of [VQ-VAE](https://arxiv.org/abs/1711.00937).

## Usage

### Training

[LightningCLI](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html) is used for learning.

- Dry-Run

    ```bash
    python scripts/cli.py fit \
        --config configs/example.yaml \
        --trainer.devices [0] \
        --trainer.fast_dev_run true
    ```

- Training(CIFAR100)

    ```bash
    python scripts/cli.py fit \
        --config configs/example.yaml \
        --trainer.devices [0]
    ```

## Reference

```bibtex
@article{van2017neural,
  title={Neural discrete representation learning},
  author={Van Den Oord, Aaron and Vinyals, Oriol and others},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```
