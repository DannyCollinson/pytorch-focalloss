"""
Benchmark for torch_focalloss comparing
performance against standard PyTorch losses.

This script compares the execution time of:
- BinaryFocalLoss vs BCEWithLogitsLoss
- MultiClassFocalLoss vs CrossEntropyLoss

Benchmarks are run with different batch sizes
and with varying values of gamma.
"""

from time import time
from typing import Any

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.benchmark import Timer

from torch_focalloss import BinaryFocalLoss, MultiClassFocalLoss


def benchmark_binary_losses(
    batch_size: int = 1024,
    feature_dim: int = 1,
    num_classes: int = 1,
    device: str = "cpu",
) -> tuple[Timer, list[Any]]:
    """
    Benchmark BinaryFocalLoss against BCEWithLogitsLoss.

    Args:
        batch_size: Number of samples in each batch
        feature_dim: For multi-label, number of features per sample
            (1 for binary classification)
        num_classes: For multi-label, number of classes
            (1 for binary classification)
        device: Device to run benchmark on ("cpu" or "cuda")
    """
    # Shape handling for both binary and multi-label cases
    if num_classes == 1 and feature_dim == 1:
        # Binary classification (batch,)
        shape = (batch_size,)
    else:
        # Multi-label classification (batch, classes)
        shape = (batch_size, feature_dim)  # type: ignore

    # Create random inputs
    inputs = torch.randn(shape, device=device)
    targets = torch.randint(0, 2, shape, device=device, dtype=torch.float32)

    # Create alpha/pos_weight if needed for multi-label
    if feature_dim > 1:
        alpha = torch.rand(feature_dim, device=device)  # type: ignore
    else:
        alpha = torch.tensor(1.5, device=device)  # type: ignore

    # Standard BCE
    bce_timer = Timer(
        stmt="loss_fn(inputs, targets)",
        globals={
            "loss_fn": BCEWithLogitsLoss(pos_weight=alpha),
            "inputs": inputs,
            "targets": targets,
        },
    )

    # Focal loss with different gamma values
    focal_timers = []
    for gamma in [0, 1, 2, 5]:
        focal_timer = Timer(
            stmt="loss_fn(inputs, targets)",
            globals={
                "loss_fn": BinaryFocalLoss(gamma=gamma, alpha=alpha),
                "inputs": inputs,
                "targets": targets,
            },
        )
        focal_timers.append((gamma, focal_timer))  # type: ignore

    return bce_timer, focal_timers  # type: ignore


def benchmark_multiclass_losses(
    batch_size: int = 1024, num_classes: int = 10, device: str = "cpu"
) -> tuple[Timer, list[Any]]:
    """
    Benchmark MultiClassFocalLoss against CrossEntropyLoss.

    Args:
        batch_size: Number of samples in each batch
        num_classes: Number of classes
        device: Device to run benchmark on ("cpu" or "cuda")
    """
    # Create random inputs
    inputs = torch.randn(batch_size, num_classes, device=device)
    targets = torch.randint(0, num_classes, (batch_size,), device=device)

    # Create class weights
    alpha = torch.rand(num_classes, device=device)

    # Standard CrossEntropy
    ce_timer = Timer(
        stmt="loss_fn(inputs, targets)",
        globals={
            "loss_fn": CrossEntropyLoss(weight=alpha),
            "inputs": inputs,
            "targets": targets,
        },
    )

    # Focal loss with different gamma values
    focal_timers = []
    for gamma in [0, 1, 2, 5]:
        focal_timer = Timer(
            stmt="loss_fn(inputs, targets)",
            globals={
                "loss_fn": MultiClassFocalLoss(
                    gamma=gamma, alpha=alpha.to(device=device)
                ),
                "inputs": inputs,
                "targets": targets,
            },
        )
        focal_timers.append((gamma, focal_timer))  # type: ignore

    return ce_timer, focal_timers  # type: ignore


def run_benchmarks():
    """Run all benchmarks and display results"""
    # start clock
    time0 = time()

    print("=" * 80)
    print("BENCHMARKING TORCH_FOCALLOSS")
    print("=" * 80)

    # Determine if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmarks on: {device}\n")

    # Binary classification benchmarks
    print(
        "\nBENCHMARK: BINARY CLASSIFICATION "
        "(BinaryFocalLoss vs BCEWithLogitsLoss)"
    )
    print("-" * 80)

    batch_sizes = [32, 128, 512, 2048]
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        bce_timer, focal_timers = benchmark_binary_losses(
            batch_size=batch_size, device=device
        )

        # Run benchmarks
        bce_result = bce_timer.blocked_autorange()
        focal_results = [
            (gamma, timer.blocked_autorange()) for gamma, timer in focal_timers
        ]

        # Display results
        print(f"BCEWithLogitsLoss: {bce_result.mean * 1000:.3f} ms")
        for gamma, result in focal_results:
            print(
                f"BinaryFocalLoss (gamma={gamma}): "
                f"{result.mean * 1000:.3f} ms "
                f"({result.mean / bce_result.mean:.2f}x slower)"
            )

    time1 = time()
    print(f"\nBenchmark took {(time1 - time0):.2f} s.")

    # Multi-label classification benchmarks
    print(
        "\nBENCHMARK: MULTI-LABEL CLASSIFICATION "
        "(BinaryFocalLoss vs BCEWithLogitsLoss)"
    )
    print("-" * 80)

    feature_dims = [3, 10, 20]
    batch_size = 512
    for feature_dim in feature_dims:
        print(f"\nBatch size: {batch_size}, Features: {feature_dim}")
        bce_timer, focal_timers = benchmark_binary_losses(
            batch_size=batch_size,
            feature_dim=feature_dim,
            num_classes=feature_dim,
            device=device,
        )

        # Run benchmarks
        bce_result = bce_timer.blocked_autorange()
        focal_results = [
            (gamma, timer.blocked_autorange()) for gamma, timer in focal_timers
        ]

        # Display results
        print(f"BCEWithLogitsLoss: {bce_result.mean * 1000:.3f} ms")
        for gamma, result in focal_results:
            print(
                f"BinaryFocalLoss (gamma={gamma}): "
                f"{result.mean * 1000:.3f} ms "
                f"({result.mean / bce_result.mean:.2f}x slower)"
            )

    time2 = time()
    print(f"\nBenchmark took {(time2 - time1):.2f} s.")

    # Multi-class classification benchmarks
    print(
        "\nBENCHMARK: MULTI-CLASS CLASSIFICATION "
        "(MultiClassFocalLoss vs CrossEntropyLoss)"
    )
    print("-" * 80)

    batch_sizes = [32, 128, 512, 2048]
    num_classes = 10
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}, Classes: {num_classes}")
        ce_timer, focal_timers = benchmark_multiclass_losses(
            batch_size=batch_size, num_classes=num_classes, device=device
        )

        # Run benchmarks
        ce_result = ce_timer.blocked_autorange()
        focal_results = [
            (gamma, timer.blocked_autorange()) for gamma, timer in focal_timers
        ]

        # Display results
        print(f"CrossEntropyLoss: {ce_result.mean * 1000:.3f} ms")
        for gamma, result in focal_results:
            print(
                f"MultiClassFocalLoss (gamma={gamma}): "
                f"{result.mean * 1000:.3f} ms "
                f"({result.mean / ce_result.mean:.2f}x slower)"
            )

    time3 = time()
    print(f"\nBenchmark took {(time3 - time2):.2f} s.")

    # Different class counts for multi-class
    print(
        "\nBENCHMARK: VARYING CLASS COUNTS "
        "(MultiClassFocalLoss vs CrossEntropyLoss)"
    )
    print("-" * 80)

    batch_size = 512
    class_counts = [2, 5, 10, 20, 100]
    for num_classes in class_counts:
        print(f"\nBatch size: {batch_size}, Classes: {num_classes}")
        ce_timer, focal_timers = benchmark_multiclass_losses(
            batch_size=batch_size, num_classes=num_classes, device=device
        )

        # Run benchmarks
        ce_result = ce_timer.blocked_autorange()
        focal_results = [
            (gamma, timer.blocked_autorange()) for gamma, timer in focal_timers
        ]

        # Display results
        print(f"CrossEntropyLoss: {ce_result.mean * 1000:.3f} ms")
        for gamma, result in focal_results:
            print(
                f"MultiClassFocalLoss (gamma={gamma}): "
                f"{result.mean * 1000:.3f} ms "
                f"({result.mean / ce_result.mean:.2f}x slower)"
            )

    time4 = time()
    print(f"\nBenchmark took {(time4 - time3):.2f} s.")

    print("\n", "-" * 80)
    print(f"Benchmarking took {(time4 - time0):.2f} s total.")


if __name__ == "__main__":
    run_benchmarks()
