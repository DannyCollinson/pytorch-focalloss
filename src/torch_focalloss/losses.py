"""PyTorch focal loss function implementations for torch_focalloss"""

# pylint: disable=no-name-in-module
from torch import Tensor, arange, topk, zeros, int64
from torch import bool as torch_bool
from torch.nn import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    Module,
    Sigmoid,
    Softmax,
)
from torch.nn.functional import one_hot


class BinaryFocalLoss(Module):
    """
    Binary focal loss implementation based on "Focal Loss for
    Dense Object Detection" (https://arxiv.org/abs/1708.02002)

    In addition to working for standard binary classification, this loss
    function works without modification for multi-label classification.

    Note that the alpha parameter differs slightly
    from Lin et al.'s implementation.

    This implementation also supports the `weight` argument
    from PyTorch's `BCEWithLogitsLoss` class.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Tensor | None = None,
        reduction: str = "mean",
        weight: Tensor | None = None,
        pos_weight: Tensor | None = None,
    ) -> None:
        """
        Initialize binary focal loss function

        Args:
            gamma (float, optional): focusing parameter that determines
                importance of hard samples vs easy samples. If set to
                `0`, focal loss is identical to `BCEWithLogitsLoss`.
                As `gamma` grows above 0, focusing strength increases
                exponentially with `gamma`. Defaults to `2`.
            alpha (Tensor | None, optional): class balancing factor(s).
                Identical to the `pos_weight` argument to
                `BCEWithLogitsLoss`. If tensor with just one element,
                it will be used to scale the loss of all positive
                examples. If tensor with more than one element,
                will be broadcast against any input tensors to scale
                corresponding positive samples. Note that this alpha is
                not identical to Lin et al.'s; see below for details.
                If tensor, should be on correct device before training.
                Defaults to `None` (no balancing).
            reduction (str): identical to the `reduction` arg to
                `BCEWithLogitsLoss`: "Specifies the reduction to apply
                to the output". Defaults to `"mean"`.
            weight (Tensor | None, optional): identical to the `weight`
                argument to `BCEWithLogitsLoss`: "a manual rescaling
                weight given to the loss of each batch element.
                If given, has to be a Tensor of size `nbatch`."
                Defaults to `None`.
            pos_weight (Tensor | None, optional): alternate name for
                specifying `alpha`. Included for drop-in compatibility
                with `BCEWithLogitsLoss`. Ignored if `alpha` is not
                `None`. Defaults to `None`.

        Note on alpha:
            Note that our alpha is similar, but not identical, to the
            one in Lin et al.'s "Focal Loss for Dense Object Detection"
            (https://arxiv.org/abs/1708.02002). Both implementations use
            alpha as the weight for the positive class, but Lin et al.
            uses (1-alpha) as the weight for the negative class, whereas
            our implementation implicitly uses 1 as the weight for the
            negative class. This means that Lin et al.'s alpha is
            constrained to [0,1], but ours is unbounded. The formula
            alpha = alpha* / (1-alpha*), where alpha* is the alpha from
            Lin et al., converts between the two. However, to eliminate
            balancing and replicate the behavior for alpha = 1 using the
            Lin et al. implementation, we must set alpha* = 1/2 and
            multiply the final loss by 2, which demonstrates that the
            conversion is not 1-to-1 when it comes to training behavior
            and requires reevaluation of the learning rate in
            particular, generally requiring lower learning rates in our
            implementation compared to Lin et al. for alpha > 1 and
            greater for alpha < 1.
        """
        super().__init__()  # type: ignore

        # check reduction mode
        assert reduction in [
            "none",
            "sum",
            "mean",
        ], (
            "Valid reduction modes are 'none', 'sum', and 'mean', "
            f"got {reduction}"
        )

        # make sure gamma is numeric
        assert isinstance(
            gamma, (int, float)
        ), f"Gamma must be a float, got {gamma} of type {type(gamma)}"

        # check alpha
        if alpha is not None:
            assert isinstance(alpha, Tensor), (
                "Alpha must be float or tensor, "
                f"got {alpha} of type {type(alpha)}"
            )
        elif pos_weight is not None:
            assert isinstance(pos_weight, Tensor), (
                "pos_weight/alpha must be tensor or None, "
                f"got {pos_weight} of type {type(pos_weight)}"
            )
            # use pos_weight in place of alpha for drop-in compatability
            alpha = pos_weight

        # components
        self.binary_cross_entropy = BCEWithLogitsLoss(
            reduction="none", pos_weight=alpha, weight=weight
        )
        self.sigmoid = Sigmoid()

        # focusing parameters
        self.gamma = gamma
        self.alpha = alpha

        # other parameters
        self.weight = weight

        # settings
        self.reduction = reduction

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        """
        Calculate binary focal loss

        Args:
            inputs (Tensor): (unnormalized) prediction logits
                of any shape
            target (Tensor): true labels with same shape as inputs

        Returns:
            Tensor: loss tensor with reduction applied
        """
        # calculate regular binary cross entropy loss
        binary_cross_entropy_loss = self.binary_cross_entropy(inputs, target)

        # calculate predicted class probabilities for correct class
        probabilities = self.sigmoid(inputs)

        # calculate focusing if needed
        if self.gamma != 0:
            focus = (target - probabilities).abs() ** self.gamma
        else:
            focus = 1  # dummy

        # apply focusing
        loss = focus * binary_cross_entropy_loss

        # apply reduction option to loss and return
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MultiClassFocalLoss(Module):
    """
    Multi-class focal loss implementation inspired by "Focal Loss for
    Dense Object Detection" (https://arxiv.org/abs/1708.02002) and
    addapted to support classification tasks with more than two classes

    Class index targets should be prefered whenever possible to class
    probability targets. This does not include using the
    `label_smoothing` parameter or the `alpha`/`weight` parameter.

    Note that the alpha parameter's meaning differs somewhat from its
    meaning in Lin et al.'s original binary focal loss

    This implementation also supports the `ignore_index` and
    `label_smoothing` arguments from PyTorch's `CrossEntropyLoss` class

    Note that one difference from `CrossEntropyLoss` is that if all
    samples have target value `ignore_index`, then `MultiClassFocalLoss`
    returns 0 where `CrossEntropyLoss` would return `nan`.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Tensor | None = None,
        reduction: str = "mean",
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        weight: Tensor | None = None,
        focus_on: int | float = 1,
        sum_first: bool = True,
    ) -> None:
        """
        Initialize multi-class focal loss function

        Args:
            gamma (float, optional): focusing parameter that determines
                importance of hard samples vs easy samples. If set to
                `0`, focal loss is identical to `CrossEntropyLoss`.
                As `gamma` grows above 0, focusing strength increases
                exponentially with `gamma`. Defaults to `2`.
            alpha (Tensor | None, optional): tensor of class balancing
                factors of shape `[num_classes,]`. Identical to the
                `weight` argument of `CrossEntropyLoss`. Should be on
                correct device before training. Defaults to `None`.
            reduction (str): identical to the `reduction` argument to
                `CrossEntropyLoss`: "Specifies the reduction to apply to
                the output". Defaults to `"mean"`.
                Values `"mean"`, `"sum"`, and `"none"` are valid.
            ignore_index (int): identical to the `ignore_index` argument
                to `CrossEntropyLoss`: "Specifies a target value that is
                ignored and does not contribute to the input gradient.
                When `reduction` is `"mean"`, the loss is averaged over
                non-ignored targets. Note that `ignore_index` is only
                applicable when the target contains class indices."
                Defaults to `-100`.
            label_smoothing (float): identical to the `label_smoothing`
                argument to `CrossEntropyLoss`: "A float in [0.0, 1.0].
                Specifies the amount of smoothing when computing the
                loss, where 0.0 means no smoothing. The targets become a
                mixture of the original ground truth and a uniform
                distribution as described in 'Rethinking the Inception
                Architecture for Computer Vision'." Defaults to 0.0.
            weight (Tensor | None, optional): alternate name for
                specifying `alpha`. Included for drop-in compatibility
                with `CrossEntropyLoss`. Ignored if `alpha` is not
                `None`. Defaults to `None`.
            focus_on (int | float): if using class probabilities as
                targets, determines which classes focusing is applied to
                for each sample. If a positive `int`, then the
                `focus_on` top classes according to probability for the
                sample will be considered. If `focus_on=-1`, then all
                classes will be considered. If a `float` in the range
                `(0, 1)`, then any class with a probability of above
                `focus_on` will be considered. Defaults to 1.
            sum_first (bool): if using class probabilities as targets
                with `focus_max_only=False`, determines if summation of
                errors occurs before or after gamma is applied. Ignored
                if `focus_max_only=True`. Defaults to True.
        """
        super().__init__()  # type: ignore

        # check reduction mode
        assert reduction in [
            "none",
            "sum",
            "mean",
        ], (
            "Valid reduction modes are 'none', 'sum', and 'mean', "
            f"got {reduction}"
        )

        # make sure gamma is numeric
        assert isinstance(
            gamma, (int, float)
        ), f"Gamma must be a float, got {gamma} of type {type(gamma)}"

        # check alpha
        if alpha is not None:
            assert isinstance(alpha, Tensor), (
                "Alpha must be a tensor or None, "
                f"got {alpha} of type {type(alpha)}"
            )
        elif weight is not None:
            assert isinstance(weight, Tensor), (
                "Weight/alpha must be a tensor or None, "
                f"got {weight} of type {type(weight)}"
            )
            # use weight in place of alpha for drop-in compatability
            alpha = weight

        # make sure focus_on is valid
        if isinstance(focus_on, int):
            assert (
                focus_on == -1 or focus_on >= 1
            ), f"If an int, focus_on must be -1 or >= 1, got {focus_on}"
        else:
            assert (
                0 < focus_on < 1
            ), f"If a float, focus_on must be in (0, 1), got {focus_on}"

        # components
        self.cross_entropy = CrossEntropyLoss(
            reduction="none",
            weight=alpha,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
        self.softmax = Softmax(dim=1)

        # focusing parameters
        self.gamma = float(gamma)
        self.alpha = alpha

        # other parameters
        self.label_smoothing = label_smoothing
        self.focus_on = focus_on

        # settings
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.sum_first = sum_first

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        """
        Calculate multi-class focal loss

        Args:
            inputs (Tensor): (unnormalized) prediction logits
                of shape `[batch_size, num_classes]`
            target (Tensor): true labels of shape `[batch_size,]`
                if targets are class indices
                or of shape `[batch_size, num_classes]`
                if targets are class probabilities

        Returns:
            Tensor: loss tensor with reduction applied
        """
        # calculate regular cross entropy loss
        cross_entropy_loss = self.cross_entropy(inputs, target)

        # calculate focusing
        if self.gamma == 0:
            # no focusing needed
            focus = 1  # dummy
        elif len(target.shape) == 1:
            # target is class indices
            probabilities = self.softmax(inputs)[
                arange(0, target.shape[0]), target.type(int64)
            ]
            focus = (1 - probabilities) ** self.gamma
        else:
            # target is class probabilities
            probabilities = self.softmax(inputs)

            # compute focus
            if self.focus_on == -1:
                focus = (target - probabilities).abs()

                # reduce dimension of focus according to sum_first
                if self.sum_first:
                    focus = focus.sum(dim=1) ** self.gamma  # type: ignore
                else:
                    focus = (focus**self.gamma).sum(dim=1)

            elif isinstance(self.focus_on, int):
                # create mask for which entries to focus on
                focus_mask = one_hot(  # pylint: disable=not-callable
                    topk(target, self.focus_on, dim=1).indices,
                    num_classes=target.shape[1],
                )
                focus_mask = focus_mask.sum(dim=1).type(torch_bool).squeeze()

                # compute focus and apply mask
                focus = (target - probabilities).abs()[focus_mask]
                focus = focus.view(  # type: ignore
                    target.shape[0], self.focus_on
                )

                # reduce dimension of focus according to sum_first
                if self.sum_first:
                    focus = focus.sum(dim=1) ** self.gamma  # type: ignore
                else:
                    focus = (focus**self.gamma).sum(dim=1)

            else:
                focus_mask = target > self.focus_on
                # don't know how many per row are focused, so calculate by row
                diff = (target - probabilities).abs()
                focus = zeros((target.shape[0],))  # type: ignore
                for i, row_focus_mask in enumerate(focus_mask):
                    if self.sum_first:
                        focus[i] += (  # type: ignore
                            diff[i, row_focus_mask].sum() ** self.gamma
                        )
                    else:
                        focus[i] += (  # type: ignore
                            diff[i, row_focus_mask] ** self.gamma
                        ).sum()

        # apply focusing
        loss = focus * cross_entropy_loss

        # apply reduction option to loss and return
        if self.reduction == "mean":
            # CrossEntropyLoss "mean" divides by effective number of samples
            if len(target.shape) == 1:
                # target is class indices, so ignore_index is checked
                if self.alpha is not None:
                    weighted_sample_num = Tensor(
                        [
                            self.alpha[val] if val != self.ignore_index else 0
                            for val in target
                        ]
                    ).sum()
                else:
                    weighted_sample_num = Tensor(
                        [1 for val in target if val != self.ignore_index]
                    ).sum()
                if weighted_sample_num.item() == 0:
                    # if all targets ignored, we want to return 0, not nan
                    return Tensor([0.0])
            else:
                # target is class probabilities, so no ignore_index
                if self.alpha is not None:
                    weighted_sample_num = (
                        self.alpha * target.sum(dim=0)
                    ).sum()
                else:
                    weighted_sample_num = target.shape[0]  # type: ignore
            return loss.sum() / weighted_sample_num
        if self.reduction == "sum":
            return loss.sum()
        return loss
