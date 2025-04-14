"""Tests of focal loss function implementations in torch_focalloss"""

# pylint: disable=no-name-in-module
from torch import (
    Tensor,
    equal,
    float32,
    full,
    isclose,
    isnan,
    rand,
    randint,
    randn,
    tensor,
    zeros,
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.nn.functional import one_hot

from torch_focalloss import BinaryFocalLoss, MultiClassFocalLoss


class TestBinaryFocalLoss:
    """Tests for BinaryFocalLoss"""

    def test_initialization(self) -> None:
        """Test initialization with various parameters"""
        # Default initialization
        bfl = BinaryFocalLoss()
        assert bfl.gamma == 2.0
        assert bfl.alpha is None
        assert bfl.reduction == "mean"
        assert bfl.weight is None

        # Custom parameters
        gamma = 3.0
        alpha = tensor(0.75)
        bfl = BinaryFocalLoss(gamma=gamma, alpha=alpha, reduction="sum")
        assert bfl.gamma == gamma
        assert isinstance(bfl.alpha, Tensor)
        assert bfl.alpha.item() == alpha
        assert bfl.reduction == "sum"

        # Using pos_weight instead of alpha
        pos_weight = tensor([1.5])
        bfl = BinaryFocalLoss(pos_weight=pos_weight)
        assert bfl.alpha is not None
        assert equal(bfl.alpha, pos_weight)

        # Using alpha tensor
        alpha_tensor = tensor([0.5, 1.0, 1.5])
        bfl = BinaryFocalLoss(alpha=alpha_tensor)
        assert equal(bfl.alpha, alpha_tensor)  # type: ignore

    def test_equals_bce_when_gamma_zero(self) -> None:
        """Test equivalence to BCE when gamma=0"""
        # Binary classification
        batch_size = 10
        preds = randn(batch_size)
        target = randint(2, size=(batch_size,), dtype=float32)

        # No weighting
        bce = BCEWithLogitsLoss()
        bfl = BinaryFocalLoss(gamma=0)

        bce_loss = bce(preds, target)
        focal_loss = bfl(preds, target)

        assert isclose(bce_loss, focal_loss)

        # With alpha weighting
        alpha = tensor(1.5)
        bce = BCEWithLogitsLoss(pos_weight=alpha)
        bfl = BinaryFocalLoss(gamma=0, alpha=alpha)

        bce_loss = bce(preds, target)
        focal_loss = bfl(preds, target)

        assert isclose(bce_loss, focal_loss)

    def test_multi_label_classification(self) -> None:
        """Test multi-label classification"""
        batch_size = 10
        num_classes = 3

        preds = randn(batch_size, num_classes)
        target = randint(2, size=(batch_size, num_classes), dtype=float32)

        # Test with different alphas for each label
        alpha = tensor([0.5, 1.0, 1.5])
        bce = BCEWithLogitsLoss(pos_weight=alpha)
        bfl = BinaryFocalLoss(gamma=0, alpha=alpha)

        bce_loss = bce(preds, target)
        focal_loss = bfl(preds, target)

        assert isclose(bce_loss, focal_loss)

        # Test with focal effect (gamma > 0)
        gamma = 2.0
        bfl = BinaryFocalLoss(gamma=gamma, alpha=alpha)
        focal_loss = bfl(preds, target)

        # Loss should be lower with gamma > 0
        assert focal_loss < bce_loss

    def test_reduction_options(self) -> None:
        """Test different reduction options"""
        batch_size = 5
        preds = randn(batch_size)
        target = randint(2, size=(batch_size,), dtype=float32)

        # Test 'none' reduction
        bfl = BinaryFocalLoss(reduction="none")
        loss = bfl(preds, target)
        assert loss.shape == (batch_size,)

        # Test 'sum' reduction
        bfl = BinaryFocalLoss(reduction="sum")
        loss = bfl(preds, target)
        assert loss.shape == ()  # scalar

        # Test 'mean' reduction
        bfl = BinaryFocalLoss(reduction="mean")
        loss = bfl(preds, target)
        assert loss.shape == ()  # scalar

    def test_weight_parameter(self) -> None:
        """Test the weight parameter works correctly"""
        batch_size = 5
        preds = randn(batch_size)
        target = randint(2, size=(batch_size,), dtype=float32)
        weight = rand(batch_size)

        bfl = BinaryFocalLoss(gamma=0, weight=weight)
        bce = BCEWithLogitsLoss(weight=weight)

        bfl_loss = bfl(preds, target)
        bce_loss = bce(preds, target)

        assert isclose(bfl_loss, bce_loss)

    def test_gamma_effect(self) -> None:
        """Test that increasing gamma decreases the loss value"""
        batch_size = 10
        preds = randn(batch_size)
        target = randint(2, size=(batch_size,), dtype=float32)

        # Create a sequence of losses with increasing gamma
        gammas = [0, 1, 2, 3]
        losses: list[float] = []

        for gamma in gammas:
            bfl = BinaryFocalLoss(gamma=gamma)
            losses.append(bfl(preds, target).item())

        # Check that loss decreases as gamma increases
        for i in range(1, len(losses)):
            assert losses[i] < losses[i - 1]

    def test_all_correct_predictions(self) -> None:
        """Test behavior when all predictions are correct"""
        # batch_size = 5

        # Create perfect predictions
        # (large positive logits for positive examples,
        # large negative logits for negative examples)
        target = tensor([0.0, 1.0, 0.0, 1.0, 0.0])
        preds = tensor([-10.0, 10.0, -10.0, 10.0, -10.0])

        bfl = BinaryFocalLoss()
        loss = bfl(preds, target)

        # Loss should be very small, possibly zero due to numerical precision
        assert loss >= 0
        assert loss < 0.01

    def test_all_incorrect_predictions(self) -> None:
        """Test behavior when all predictions are incorrect"""
        # batch_size = 5

        # Create completely wrong predictions
        target = tensor([0.0, 1.0, 0.0, 1.0, 0.0])
        preds = tensor([10.0, -10.0, 10.0, -10.0, 10.0])

        bfl = BinaryFocalLoss()
        loss = bfl(preds, target)

        # Loss should be large
        assert loss > 5


class TestMultiClassFocalLoss:
    """Tests for MultiClassFocalLoss"""

    def test_initialization(self) -> None:
        """Test initialization with various parameters"""
        # Default initialization
        mcfl = MultiClassFocalLoss()
        assert mcfl.gamma == 2.0
        assert mcfl.alpha is None
        assert mcfl.reduction == "mean"
        assert mcfl.ignore_index == -100
        assert mcfl.label_smoothing == 0.0
        assert mcfl.sum_first is True
        assert mcfl.focus_on == 1  # Default focus_on value from implementation

        # Custom parameters with integer focus_on
        gamma = 3.0
        num_classes = 4
        alpha = rand(num_classes)
        focus_on: int | float = 2
        mcfl = MultiClassFocalLoss(
            gamma=gamma,
            alpha=alpha,
            reduction="sum",
            ignore_index=-1,
            label_smoothing=0.1,
            focus_on=focus_on,
            sum_first=False,
        )
        assert mcfl.gamma == gamma
        assert equal(mcfl.alpha, alpha)  # type: ignore
        assert mcfl.reduction == "sum"
        assert mcfl.ignore_index == -1
        assert mcfl.label_smoothing == 0.1
        assert mcfl.sum_first is False
        assert mcfl.focus_on == focus_on

        # Custom parameters with float focus_on (threshold)
        focus_on = 0.3
        mcfl = MultiClassFocalLoss(focus_on=focus_on)
        assert mcfl.focus_on == focus_on

        # Custom parameters with all classes focus
        mcfl = MultiClassFocalLoss(focus_on=-1)
        assert mcfl.focus_on == -1

        # Using weight instead of alpha
        weight = rand(num_classes)
        mcfl = MultiClassFocalLoss(weight=weight)
        assert equal(mcfl.alpha, weight)  # type: ignore

    def test_equals_ce_when_gamma_zero(self) -> None:
        """Test equivalence to CrossEntropy when gamma=0"""
        batch_size = 10
        num_classes = 4

        preds = randn(batch_size, num_classes)
        target = randint(num_classes, size=(batch_size,))

        # No weighting
        ce = CrossEntropyLoss()
        mcfl = MultiClassFocalLoss(gamma=0)

        ce_loss = ce(preds, target)
        focal_loss = mcfl(preds, target)

        assert isclose(ce_loss, focal_loss)

        # With class weighting
        alpha = rand(num_classes)
        ce = CrossEntropyLoss(weight=alpha)
        mcfl = MultiClassFocalLoss(gamma=0, alpha=alpha)

        ce_loss = ce(preds, target)
        focal_loss = mcfl(preds, target)

        assert isclose(ce_loss, focal_loss)

    def test_reduction_options(self) -> None:
        """Test different reduction options"""
        batch_size = 5
        num_classes = 3

        preds = randn(batch_size, num_classes)
        target = randint(num_classes, size=(batch_size,))

        # Test 'none' reduction
        mcfl = MultiClassFocalLoss(reduction="none")
        loss = mcfl(preds, target)
        assert loss.shape == (batch_size,)

        # Test 'sum' reduction
        mcfl = MultiClassFocalLoss(reduction="sum")
        loss = mcfl(preds, target)
        assert loss.shape == ()  # scalar

        # Test 'mean' reduction
        mcfl = MultiClassFocalLoss(reduction="mean")
        loss = mcfl(preds, target)
        assert loss.shape == ()  # scalar

    def test_gamma_effect(self) -> None:
        """Test that increasing gamma decreases the loss value"""
        batch_size = 10
        num_classes = 4

        preds = randn(batch_size, num_classes)
        target = randint(num_classes, size=(batch_size,))

        # Create a sequence of losses with increasing gamma
        gammas = [0, 1, 2, 3]
        losses: list[float] = []

        for gamma in gammas:
            mcfl = MultiClassFocalLoss(gamma=gamma)
            losses.append(mcfl(preds, target).item())

        # Check that loss decreases as gamma increases
        for i in range(1, len(losses)):
            assert losses[i] < losses[i - 1]

    def test_ignore_index(self) -> None:
        """Test ignore_index parameter works correctly"""
        batch_size = 5
        num_classes = 3
        ignore_idx = 2

        preds = randn(batch_size, num_classes)
        # Create targets with some elements equal to ignore_idx
        target = tensor([0, 1, ignore_idx, 0, 1])

        # Standard CE loss with ignore_index
        ce = CrossEntropyLoss(ignore_index=ignore_idx)
        # Focal loss with same ignore_index
        mcfl = MultiClassFocalLoss(gamma=0, ignore_index=ignore_idx)

        ce_loss = ce(preds, target)
        focal_loss = mcfl(preds, target)

        assert isclose(ce_loss, focal_loss)

        # Test all-ignored case
        # Create a target with all elements equal to ignore_idx
        all_ignored_target = full((batch_size,), ignore_idx)

        ce_loss = ce(preds, all_ignored_target)
        focal_loss = mcfl(preds, all_ignored_target)

        # PyTorch behavior returns nan when all targets are ignored
        assert isnan(ce_loss)
        # Our improved implementation returns 0 instead of nan
        assert focal_loss.item() == 0.0

    def test_label_smoothing(self) -> None:
        """Test label smoothing parameter works correctly"""
        batch_size = 5
        num_classes = 3
        smoothing = 0.1

        preds = randn(batch_size, num_classes)
        target = randint(num_classes, size=(batch_size,))

        # Standard CE with label smoothing
        ce = CrossEntropyLoss(label_smoothing=smoothing)
        # Focal loss with same label smoothing
        mcfl = MultiClassFocalLoss(gamma=0, label_smoothing=smoothing)

        ce_loss = ce(preds, target)
        focal_loss = mcfl(preds, target)

        assert isclose(ce_loss, focal_loss)

    def test_all_correct_predictions(self) -> None:
        """Test behavior when all predictions are correct"""
        batch_size = 5
        num_classes = 3

        # Create one-hot encoded target
        target = tensor([0, 1, 2, 0, 1])

        # Create perfect predictions (very large values at target positions)
        preds = full((batch_size, num_classes), -100.0)
        for i, t in enumerate(target):
            preds[i, t] = 100.0

        mcfl = MultiClassFocalLoss()
        loss = mcfl(preds, target)

        # Loss should be very small, possibly 0 due to numerical precision
        assert loss >= 0
        assert loss < 0.01

    def test_all_incorrect_predictions(self) -> None:
        """Test behavior when all predictions are incorrect"""
        batch_size = 5
        num_classes = 3

        # Create targets
        target = tensor([0, 1, 2, 0, 1])

        # Create completely wrong predictions
        # (very negative values at target positions)
        preds = full((batch_size, num_classes), 100.0)
        for i, t in enumerate(target):
            preds[i, t] = -100.0

        mcfl = MultiClassFocalLoss()
        loss = mcfl(preds, target)

        # Loss should be large
        assert loss > 5

    def test_weighted_loss_mean_reduction(self) -> None:
        """Test that mean reduction respects class weights"""
        batch_size = 6
        num_classes = 3

        # Create a balanced dataset with 2 examples of each class
        target = tensor([0, 0, 1, 1, 2, 2])
        preds = randn(batch_size, num_classes)

        # Create weights that heavily favor class 0
        alpha = tensor([10.0, 1.0, 1.0])

        # Create loss with these weights
        mcfl = MultiClassFocalLoss(gamma=0, alpha=alpha)
        weighted_loss = mcfl(preds, target)

        # Create a separate loss for each class
        class0_loss = CrossEntropyLoss()(
            preds[target == 0], target[target == 0]
        )
        class1_loss = CrossEntropyLoss()(
            preds[target == 1], target[target == 1]
        )
        class2_loss = CrossEntropyLoss()(
            preds[target == 2], target[target == 2]
        )

        # weighted mean should be closer to class0_loss due to higher weight
        manual_weighted_mean = (
            10.0 * class0_loss + 1.0 * class1_loss + 1.0 * class2_loss
        ) / 12.0

        assert isclose(weighted_loss, manual_weighted_mean)

    def test_with_probability_targets(self) -> None:
        """Test using class probability targets instead of class indices"""
        batch_size = 5
        num_classes = 4

        # Create prediction logits
        preds = randn(batch_size, num_classes)

        # Create class index targets
        idx_targets = randint(num_classes, size=(batch_size,))

        # Convert to one-hot probability targets
        # pylint: disable=not-callable
        prob_targets = one_hot(idx_targets, num_classes=num_classes).float()
        # pylint: enable=not-callable

        # Test with gamma=0 to compare with standard CrossEntropyLoss
        mcfl_idx = MultiClassFocalLoss(gamma=0)
        loss_idx = mcfl_idx(preds, idx_targets)

        mcfl_prob = MultiClassFocalLoss(gamma=0)
        loss_prob = mcfl_prob(preds, prob_targets)

        # Losses should be equivalent
        assert isclose(loss_idx, loss_prob)

    def test_focus_on_top_k(self) -> None:
        """Test focus_on parameter with integer value (top-k)"""
        batch_size = 5
        num_classes = 4

        # Create prediction logits
        preds = randn(batch_size, num_classes)

        # Create probability targets with different confidence levels
        prob_targets = rand(batch_size, num_classes)
        # Normalize to make them sum to 1
        prob_targets = prob_targets / prob_targets.sum(dim=1, keepdim=True)

        # Test with different top-k values
        gamma = 2.0
        for k in [1, 2, 3]:
            mcfl = MultiClassFocalLoss(gamma=gamma, focus_on=k)
            loss = mcfl(preds, prob_targets)

            # Just verify we get a valid loss (no errors)
            assert loss > 0

        # Test that different k values give different results
        loss_top1 = MultiClassFocalLoss(gamma=gamma, focus_on=1)(
            preds, prob_targets
        )
        loss_top2 = MultiClassFocalLoss(gamma=gamma, focus_on=2)(
            preds, prob_targets
        )

        # Loss with more focused classes should be different
        assert not isclose(loss_top1, loss_top2)

        # Test with both sum_first=True and sum_first=False
        loss_top2_sum_first = MultiClassFocalLoss(
            gamma=gamma, focus_on=2, sum_first=True
        )(preds, prob_targets)
        loss_top2_not_sum_first = MultiClassFocalLoss(
            gamma=gamma, focus_on=2, sum_first=False
        )(preds, prob_targets)

        # These should give different results
        assert not isclose(loss_top2_sum_first, loss_top2_not_sum_first)

    def test_focus_on_threshold(self) -> None:
        """Test focus_on parameter with float value (threshold)"""
        batch_size = 5
        num_classes = 4

        # Create prediction logits
        preds = randn(batch_size, num_classes)

        # Create probability targets with clear differences
        prob_targets = zeros(batch_size, num_classes)
        # Set primary class with 0.6 probability and secondary with 0.3
        for i in range(batch_size):
            primary = i % num_classes
            secondary = (i + 1) % num_classes
            prob_targets[i, primary] = 0.6
            prob_targets[i, secondary] = 0.3
            # Distribute remaining 0.1 evenly
            remaining = [
                j for j in range(num_classes) if j not in [primary, secondary]
            ]
            for j in remaining:
                prob_targets[i, j] = 0.05

        # Test with different thresholds
        gamma = 2.0

        # Threshold 0.2 should include primary and secondary classes
        loss_th_0_2 = MultiClassFocalLoss(gamma=gamma, focus_on=0.2)(
            preds, prob_targets
        )

        # Threshold 0.4 should include only primary class
        loss_th_0_4 = MultiClassFocalLoss(gamma=gamma, focus_on=0.4)(
            preds, prob_targets
        )

        # Threshold 0.7 should include no classes - test extreme case
        loss_th_0_7 = MultiClassFocalLoss(gamma=gamma, focus_on=0.7)(
            preds, prob_targets
        )

        # Losses should be different
        assert not isclose(loss_th_0_2, loss_th_0_4)
        assert not isclose(loss_th_0_4, loss_th_0_7)

        # Test with both sum_first=True and sum_first=False
        loss_th_0_2_sum_first = MultiClassFocalLoss(
            gamma=gamma, focus_on=0.2, sum_first=True
        )(preds, prob_targets)
        loss_th_0_2_not_sum_first = MultiClassFocalLoss(
            gamma=gamma, focus_on=0.2, sum_first=False
        )(preds, prob_targets)

        # These should give different results
        assert not isclose(loss_th_0_2_sum_first, loss_th_0_2_not_sum_first)

    def test_focus_on_all_classes(self) -> None:
        """Test focus_on=-1 to focus on all classes"""
        batch_size = 5
        num_classes = 4

        # Create prediction logits
        preds = randn(batch_size, num_classes)

        # Create probability targets
        prob_targets = rand(batch_size, num_classes)
        # Normalize to make them sum to 1
        prob_targets = prob_targets / prob_targets.sum(dim=1, keepdim=True)

        gamma = 2.0

        # Test focus_on=-1 (all classes)
        loss_all = MultiClassFocalLoss(gamma=gamma, focus_on=-1)(
            preds, prob_targets
        )

        # Test focus_on=1 (just the top class)
        loss_top1 = MultiClassFocalLoss(gamma=gamma, focus_on=1)(
            preds, prob_targets
        )

        # Loss with all classes should be different from top1
        assert not isclose(loss_all, loss_top1)

        # Using a very low threshold should be similar to using all classes
        loss_low_th = MultiClassFocalLoss(gamma=gamma, focus_on=0.01)(
            preds, prob_targets
        )
        # These might not be exactly equal but should be closer to each other
        # than to the top1 loss
        diff_all_vs_low = (loss_all - loss_low_th).abs()
        diff_all_vs_top1 = (loss_all - loss_top1).abs()
        assert diff_all_vs_low < diff_all_vs_top1

        # Test with both sum_first=True and sum_first=False
        loss_all_sum_first = MultiClassFocalLoss(
            gamma=gamma, focus_on=-1, sum_first=True
        )(preds, prob_targets)
        loss_all_not_sum_first = MultiClassFocalLoss(
            gamma=gamma, focus_on=-1, sum_first=False
        )(preds, prob_targets)

        # These should give different results
        assert not isclose(loss_all_sum_first, loss_all_not_sum_first)

    def test_sum_first_parameter(self) -> None:
        """
        Test that sum_first parameter works correctly with probability targets
        """
        batch_size = 5
        num_classes = 3

        # Create prediction logits
        preds = randn(batch_size, num_classes)

        # Create probability targets (smooth targets, not just one-hot)
        prob_targets = rand(batch_size, num_classes)
        # Normalize to make them sum to 1
        prob_targets = prob_targets / prob_targets.sum(dim=1, keepdim=True)

        # Create loss with sum_first=True and sum_first=False
        mcfl_sum_first = MultiClassFocalLoss(
            gamma=2.0, sum_first=True, focus_on=-1
        )
        mcfl_not_sum_first = MultiClassFocalLoss(
            gamma=2.0, sum_first=False, focus_on=-1
        )

        loss_sum_first = mcfl_sum_first(preds, prob_targets)
        loss_not_sum_first = mcfl_not_sum_first(preds, prob_targets)

        # Losses should be different
        assert not isclose(loss_sum_first, loss_not_sum_first)

        # Both losses should be positive
        assert loss_sum_first > 0
        assert loss_not_sum_first > 0

    def test_weighted_probability_target_reduction(self) -> None:
        """Test mean reduction with alpha and probability targets"""
        batch_size = 5
        num_classes = 4

        # Create prediction logits
        preds = randn(batch_size, num_classes)

        # Create probability targets
        prob_targets = rand(batch_size, num_classes)
        # Normalize to make them sum to 1
        prob_targets = prob_targets / prob_targets.sum(dim=1, keepdim=True)

        # Create alpha weights
        alpha = rand(num_classes)

        # Test with mean reduction
        mcfl = MultiClassFocalLoss(gamma=2.0, alpha=alpha, reduction="mean")
        loss = mcfl(preds, prob_targets)

        # Verify loss is computed correctly
        assert loss > 0

        # Test sum reduction as well
        mcfl_sum = MultiClassFocalLoss(gamma=2.0, alpha=alpha, reduction="sum")
        loss_sum = mcfl_sum(preds, prob_targets)

        # Sum should be larger than mean
        assert loss_sum > loss

    def test_empty_focus_mask(self) -> None:
        """Test case where no classes meet the threshold criteria"""
        batch_size = 5
        num_classes = 4

        # Create prediction logits
        preds = randn(batch_size, num_classes)

        # Create probability targets with low max values
        prob_targets = (
            rand(batch_size, num_classes) * 0.1
        )  # All values below 0.1
        # Normalize to make them sum to 1
        prob_targets = prob_targets / prob_targets.sum(dim=1, keepdim=True)

        # Set a threshold higher than any probability
        mcfl = MultiClassFocalLoss(gamma=2.0, focus_on=0.5)

        # This should still compute without errors
        loss = mcfl(preds, prob_targets)
        assert loss >= 0
