# `pytorch-focalloss`

The `pytorch-focalloss` package contains the python package `torch_focalloss`, which provides PyTorch implementations of binary and multi-class focal loss functions.

### Installation

`pytorch-focalloss` is installable from PyPI.

```
pip install pytorch-focalloss
```

### Usage

The python package is importable as `torch_focalloss`. The only components in the package are the `BinaryFocalLoss` and `MultiClassFocalLoss` classes, which have interfaces that allow them to be drop-in replacements for PyTorch's `BCEWithLogitsLoss` and `CrossEntropyLoss` classes, respectively. All of the same keyword arguments are supported, as well as the focusing parameter $\gamma$ (gamma), and they function just like any other PyTorch loss function.

Benchmarks for comparing run times and memory usage of the focal loss implementations compared to their standard counterparts can be run using `python ./benchmarking/benchmark_X.py` from the repository's root directory.

### About

Focal loss was first described in Lin et al.'s "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002).

#### Binary focal loss

This implementation of binary focal loss extends the original slightly, allowing for multi-label classification with no modification needed, including support for using a different value of $\alpha$ for each label by supplying a tensor of values.

It is built on top of PyTorch's `BCEWithLogitsLoss` class, and supports all of the same arguments. The `pos_weight` argument is given as `alpha` (but can alternatively be given as `pos_weight` for drop-in compatability with `BCEWithLogitsLoss`), and the `reduction` and `weight` arguments are the same as for `BCEWithLogitsLoss`.

#### Multi-class focal loss

The multi-class focal loss is a logical extension of the original binary focal loss to the N-class case. It similarly accepts a tensor of weights, with one for each class, as $\alpha$.

It is built on top of PyTorch's `CrossEntropyLoss` class, and supports all of the same arguments. The `weight` argument is given as `alpha` (but can alternatively be given as `weight` for drop-in compatability with `CrossEntropyLoss`), and the `reduction`, `ignore_index`, and `label_smoothing` arguments are the same as for `CrossEntropyLoss`.

Note that one difference from `CrossEntropyLoss` is that if all samples have target value `ignore_index`, then `MultiClassFocalLoss` returns 0 where `CrossEntropyLoss` would return `nan`.

## Demo

See below or check out `DEMO.ipynb` above for a demonstration of how the binary and multi-class focal losses work and compare to the standard cross entropy versions.

There are also benchmarks available to run using `python ./benchmarking/benchmark_X.py` from the repository's root directory that can compare the run times and memory usage of the focal loss implementations compared to their standard counterparts.


```python
from torch import cuda, float32, ones, randint, randn, tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from torch_focalloss import BinaryFocalLoss, MultiClassFocalLoss
```


```python
device = "cuda" if cuda.is_available() else "cpu"
```

## BinaryFocalLoss

### BinaryFocalLoss for binary classification

We'll use the same inputs for the whole example to demonstrate how changes in parameters changes the loss value.

First we create our simulated batch of 5 binary labels and raw logits.


```python
preds = randn(5, device=device)
target = randint(2, size=(5,), dtype=float32, device=device)
print("Logits: ", preds)
print("Target: ", target)
```

    Logits:  tensor([ 0.5257,  0.9124, -0.9304,  1.0868,  1.2109], device='cuda:0')
    Target:  tensor([1., 1., 1., 0., 1.], device='cuda:0')


The normal binary cross entropy loss is the same as focal loss when $\gamma$ (gamma), which determines the strength of focus on difficult samples, is equal to $0$.


```python
gamma = 0

bce = BCEWithLogitsLoss()
bfl = BinaryFocalLoss(gamma=gamma)

print(f"BCE Loss: {bce(preds, target).item():.5f}")
print(f"Focal Loss: {bfl(preds, target).item():.5f}")
```

    BCE Loss: 0.74063
    Focal Loss: 0.74063


This is also true when the weight applied to the positive class (1) relative to the negative class (0) is not 1. This parameter is called $\alpha$ (alpha) and is identical to the `pos_weight` parameter of the `BCEWithLogits` class, which is used to help manage class imbalance.


```python
gamma = 0
alpha = tensor(1.5, device=device)

bce = BCEWithLogitsLoss(pos_weight=alpha)
bfl = BinaryFocalLoss(gamma=gamma, alpha=alpha)

print(f"BCE Loss: {bce(preds, target).item():.5f}")
print(f"Focal Loss: {bfl(preds, target).item():.5f}")
```

    BCE Loss: 0.97320
    Focal Loss: 0.97320


Note that our $\alpha$ is similar, but not identical, to the one in Lin et al.'s "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002). Both implementations use $\alpha$ as the weight for the positive class, but Lin et al. uses $(1-\alpha)$ as the weight for the negative class, whereas our implementation implicitly uses $1$ as the weight for the negative class. This means that Lin et al.'s $\alpha$ is constrained to $[0,1]$, but ours is unbounded.

The formula $\alpha = L / (1-L)$, where $L$ is the $\alpha$ from Lin et al., converts between the two. However, to eliminate balancing and replicate the behavior for $\alpha=1$ using the Lin et al. implementation, we must set $L=0.5$ and multiply the final loss by 2, which demonstrates that the conversion is not 1-to-1 when it comes to training behavior. Notably, it requires reevaluation of the learning rate in particular, generally requiring lower learning rates in our implementation compared to Lin et al. for $\alpha>1$ and higher learning rates for $\alpha<1$.

Focal loss differs from binary cross entropy loss when $\gamma\neq0$. Technically, $\gamma$ can be less than $0$, but this would increase focus on easy samples and defocus hard samples, which is the opposite of why focal loss is effective. Thus, we will show what happens when $\gamma>0$.


```python
gamma = 2

bce = BCEWithLogitsLoss()
bfl = BinaryFocalLoss(gamma=gamma)

print(f"BCE Loss: {bce(preds, target).item():.5f}")
print(f"Focal Loss: {bfl(preds, target).item():.5f}")
```

    BCE Loss: 0.74063
    Focal Loss: 0.30507


### BinaryFocalLoss for multi-label classification.

Just like binary cross entropy loss, we can use our binary focal loss for multi-label classification without modification.

We will simulate a batch of 5 samples, each with 3 binary labels.


```python
preds = randn(5, 3, device=device)
target = randint(2, size=(5, 3), dtype=float32, device=device)
print("Logits: \n", preds)
print("Target: \n", target)
```

    Logits:
     tensor([[-0.8072,  0.0658,  1.5409],
            [-1.1151,  0.9102,  0.3073],
            [ 2.3941,  2.0975, -0.3208],
            [ 0.2687,  0.0528,  0.5680],
            [-1.3618, -0.4430, -1.3281]], device='cuda:0')
    Target:
     tensor([[1., 0., 1.],
            [1., 1., 0.],
            [0., 1., 1.],
            [1., 1., 1.],
            [0., 0., 0.]], device='cuda:0')



```python
gamma = 2
alpha = tensor(1.5, device=device)

bce = BCEWithLogitsLoss(pos_weight=alpha)
bfl = BinaryFocalLoss(gamma=gamma, alpha=alpha)

print(f"BCE Loss: {bce(preds, target).item():.5f}")
print(f"Focal Loss: {bfl(preds, target).item():.5f}")
```

    BCE Loss: 0.91235
    Focal Loss: 0.37774


When doing multi-label classification, you can also specify a value of $\alpha$ for each label by combining them in a tensor.


```python
gamma = 2
alpha = tensor([0.5, 1, 1.5], device=device)

bce = BCEWithLogitsLoss(pos_weight=alpha)
bfl = BinaryFocalLoss(gamma=gamma, alpha=alpha)

print(f"BCE Loss: {bce(preds, target).item():.5f}")
print(f"Focal Loss: {bfl(preds, target).item():.5f}")
```

    BCE Loss: 0.66547
    Focal Loss: 0.27402


## MultiClassFocalLoss

We also extended Lin et al.'s focal loss, which they only defined for the binary case, to the multiclass case.

Our example input will be for a 4-class classification problem, so we will create a sample of 5 labels and 5 sets of logits.


```python
preds = randn(5, 4, device=device)
target = randint(4, size=(5,), device=device)
print("Logits: \n", preds)
print("Target: \n", target)
```

    Logits:
     tensor([[ 0.6680, -0.9365,  0.1303, -0.6680],
            [-0.0752,  1.0425, -0.1543, -0.7228],
            [-1.1970,  0.5895,  0.3956,  1.9686],
            [-0.0353,  1.0202,  0.6165, -1.0623],
            [-1.9054, -0.4874, -1.2124,  0.5739]], device='cuda:0')
    Target:
     tensor([3, 1, 0, 0, 0], device='cuda:0')


Like binary focal loss and binary cross entropy loss, multi-class focal loss and cross entropy loss are the same when $\gamma=0$.


```python
gamma = 0

cel = CrossEntropyLoss()
mcfl = MultiClassFocalLoss(gamma=gamma)

print(f"Cross Entropy Loss: {cel(preds, target).item():.5f}")
print(f"Multi-Class Focal Loss: {mcfl(preds, target).item():.5f}")
```

    Cross Entropy Loss: 2.19540
    Multi-Class Focal Loss: 2.19540


This is also true when we apply class balancing weights. We also call these $\alpha$, and they are identical to the "weight" argument of the `CrossEntropyLoss` class. Note that when using the reduction option `"mean"`, the weighted mean is taken, which means that the sum is divided by the effective number of samples according to the class weights. This is the same behavior as for the standard `CrossEntropyLoss` class.


```python
gamma = 0
alpha = (ones(4) + randn(4)).abs().to(device=device)
print(f"Alpha: {alpha}\n")

cel = CrossEntropyLoss(weight=alpha)
mcfl = MultiClassFocalLoss(gamma=gamma, alpha=alpha)

print(f"Cross Entropy Loss: {cel(preds, target).item():.5f}")
print(f"Multi-Class Focal Loss: {mcfl(preds, target).item():.5f}")
```

    Alpha: tensor([1.0490, 0.1758, 0.8946, 1.3543], device='cuda:0')

    Cross Entropy Loss: 2.48619
    Multi-Class Focal Loss: 2.48619


As in the binary case, multi-class focal loss differs from cross entropy loss when $\gamma\neq0$. Again, we will only show what happens when $\gamma>0$.


```python
gamma = 2

cel = CrossEntropyLoss(weight=alpha)
mcfl = MultiClassFocalLoss(gamma=gamma, alpha=alpha)

print(f"Cross Entropy Loss: {cel(preds, target).item():.5f}")
print(f"Multi-Class Focal Loss: {mcfl(preds, target).item():.5f}")
```

    Cross Entropy Loss: 2.48619
    Multi-Class Focal Loss: 2.09198


## Info

Use the Issues section for questions, feedback, and concerns, or create a Pull Request if you want to contribute!

### Thank you for checking out the `torch_focalloss` package!
