This repository borrows most of the code from the paper
> Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer and Tom Goldstein. [*Visualizing the Loss Landscape of Neural Nets*](https://arxiv.org/abs/1712.09913). NIPS, 2018.

Given a network architecture and its pre-trained parameters, this tool memorize and visualizes the optimization trajectory.
The calculation is set to be done on single GPU, with multiple process.

## Setup

Ensure the following software/libraries installed:
- [Python 3.10](https://python.org/)
- [PyTorch 2.0](https://pytorch.org/)

You can install other dependencies using the following command:

```
pip install -r requirements.txt
```

Then, launch the backend simply run:
```
python api.py
```

## Reference

[EfficientNetV2 implementation](https://github.com/d-li14/efficientnetv2.pytorch)

[Other CNNs](https://github.com/fdraxler/PyTorch-AutoNEB)
