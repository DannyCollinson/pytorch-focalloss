"""
Advanced benchmark for torch_focalloss
with detailed analysis and visualization.

This script provides:
1. Detailed performance analysis across different parameters
2. Memory usage comparisons
3. Visual plots of benchmark results
4. Comparative time analysis with varying hyperparameters
"""

import gc
from collections.abc import Callable, Sequence
from time import time
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.figure import Figure
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.benchmark import Timer

from torch_focalloss import BinaryFocalLoss, MultiClassFocalLoss


def measure_memory_usage(
    func: Callable[..., Any], *args: Sequence[Any], **kwargs: dict[str, Any]
) -> tuple[Any, int]:
    """Measure the peak memory usage of a function"""
    # Make sure CUDA is synchronized before measurement
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.max_memory_allocated()

    result = func(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        end_mem = torch.cuda.max_memory_allocated()
        memory_usage = end_mem - start_mem  # type: ignore
    else:
        memory_usage = 0  # Can't measure memory usage on CPU

    return result, memory_usage


def benchmark_binary_focal_loss(
    batch_sizes: list[int],
    gammas: list[float],
    feature_dims: list[int] | None = None,
    device: str = "cpu",
):
    """
    Comprehensive benchmark for BinaryFocalLoss

    Args:
        batch_sizes: List of batch sizes to test
        gammas: List of gamma values to test
        feature_dims: List of feature dimensions for multi-label case
            (None for binary)
        device: Device to run on

    Returns:
        DataFrame with benchmark results
    """
    results = []

    # Default to binary classification if no feature_dims provided
    if feature_dims is None:
        feature_dims = [1]

    for batch_size in batch_sizes:
        for feature_dim in feature_dims:
            # Shape handling
            if feature_dim == 1:
                shape = (batch_size,)
                alpha = torch.tensor(1.5, device=device)
            else:
                shape = (batch_size, feature_dim)  # type: ignore
                alpha = torch.rand(feature_dim, device=device)  # type: ignore

            # Create inputs
            inputs = torch.randn(shape, device=device)
            targets = torch.randint(
                0, 2, shape, device=device, dtype=torch.float32
            )

            # Test standard BCE first
            bce_loss = BCEWithLogitsLoss(pos_weight=alpha)

            # Measure execution time
            bce_timer = Timer(
                stmt="loss_fn(inputs, targets)",
                globals={
                    "loss_fn": bce_loss,
                    "inputs": inputs,
                    "targets": targets,
                },
            )
            bce_measurements = bce_timer.blocked_autorange()

            # Measure memory for BCE (only on CUDA)
            if device == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
                _, bce_memory = measure_memory_usage(
                    bce_loss, inputs, targets  # type: ignore
                )
            else:
                bce_memory = 0

            results.append(  # type: ignore
                {
                    "loss_type": "BCEWithLogitsLoss",
                    "batch_size": batch_size,
                    "feature_dim": feature_dim,
                    "gamma": 0,  # Not applicable
                    "time_mean_ms": bce_measurements.mean * 1000,
                    "time_std_ms": bce_measurements.iqr / 1.349 * 1000,
                    "memory_bytes": bce_memory,
                    "relative_time": 1.0,  # Reference
                    "is_focal": False,
                }
            )

            # Now test focal loss with different gammas
            for gamma in gammas:
                focal_loss = BinaryFocalLoss(
                    gamma=gamma, alpha=alpha  # type: ignore
                )

                # Measure execution time
                focal_timer = Timer(
                    stmt="loss_fn(inputs, targets)",
                    globals={
                        "loss_fn": focal_loss,
                        "inputs": inputs,
                        "targets": targets,
                    },
                )
                focal_measurements = focal_timer.blocked_autorange()

                # Measure memory for focal loss (only on CUDA)
                if device == "cuda":
                    gc.collect()
                    torch.cuda.empty_cache()
                    _, focal_memory = measure_memory_usage(
                        focal_loss, inputs, targets  # type: ignore
                    )
                else:
                    focal_memory = 0

                results.append(  # type: ignore
                    {
                        "loss_type": "BinaryFocalLoss",
                        "batch_size": batch_size,
                        "feature_dim": feature_dim,
                        "gamma": gamma,
                        "time_mean_ms": focal_measurements.mean * 1000,
                        "time_std_ms": focal_measurements.iqr / 1.349 * 1000,
                        "memory_bytes": focal_memory,
                        "relative_time": focal_measurements.mean
                        / bce_measurements.mean,
                        "is_focal": True,
                    }
                )

    return pd.DataFrame(results)


def benchmark_multiclass_focal_loss(
    batch_sizes: list[int],
    gammas: list[float],
    class_counts: list[int],
    device: str = "cpu",
):
    """
    Comprehensive benchmark for MultiClassFocalLoss

    Args:
        batch_sizes: List of batch sizes to test
        gammas: List of gamma values to test
        class_counts: List of number of classes to test
        device: Device to run on

    Returns:
        DataFrame with benchmark results
    """
    results = []

    for batch_size in batch_sizes:
        for num_classes in class_counts:
            # Create inputs
            inputs = torch.randn(batch_size, num_classes, device=device)
            targets = torch.randint(
                0, num_classes, (batch_size,), device=device
            )

            # Create class weights
            alpha = torch.rand(num_classes, device=device)

            # Test standard CrossEntropy first
            ce_loss = CrossEntropyLoss(weight=alpha)

            # Measure execution time
            ce_timer = Timer(
                stmt="loss_fn(inputs, targets)",
                globals={
                    "loss_fn": ce_loss,
                    "inputs": inputs,
                    "targets": targets,
                },
            )
            ce_measurements = ce_timer.blocked_autorange()

            # Measure memory for CE (only on CUDA)
            if device == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
                _, ce_memory = measure_memory_usage(
                    ce_loss, inputs, targets  # type: ignore
                )
            else:
                ce_memory = 0

            results.append(  # type: ignore
                {
                    "loss_type": "CrossEntropyLoss",
                    "batch_size": batch_size,
                    "num_classes": num_classes,
                    "gamma": 0,  # Not applicable
                    "time_mean_ms": ce_measurements.mean * 1000,
                    "time_std_ms": ce_measurements.iqr / 1.349 * 1000,
                    "memory_bytes": ce_memory,
                    "relative_time": 1.0,  # Reference
                    "is_focal": False,
                }
            )

            # Now test focal loss with different gammas
            for gamma in gammas:
                focal_loss = MultiClassFocalLoss(gamma=gamma, alpha=alpha)

                # Measure execution time
                focal_timer = Timer(
                    stmt="loss_fn(inputs, targets)",
                    globals={
                        "loss_fn": focal_loss,
                        "inputs": inputs,
                        "targets": targets,
                    },
                )
                focal_measurements = focal_timer.blocked_autorange()

                # Measure memory for focal loss (only on CUDA)
                if device == "cuda":
                    gc.collect()
                    torch.cuda.empty_cache()
                    _, focal_memory = measure_memory_usage(
                        focal_loss, inputs, targets  # type: ignore
                    )
                else:
                    focal_memory = 0

                results.append(  # type: ignore
                    {
                        "loss_type": "MultiClassFocalLoss",
                        "batch_size": batch_size,
                        "num_classes": num_classes,
                        "gamma": gamma,
                        "time_mean_ms": focal_measurements.mean * 1000,
                        "time_std_ms": focal_measurements.iqr / 1.349 * 1000,
                        "memory_bytes": focal_memory,
                        "relative_time": focal_measurements.mean
                        / ce_measurements.mean,
                        "is_focal": True,
                    }
                )

    return pd.DataFrame(results)


def plot_benchmark_results(
    binary_results: Any, multiclass_results: Any, save_path: str | None = None
) -> Figure:
    """Create plots from benchmark results"""
    # Set up the figure layout
    fig = plt.figure(figsize=(15, 12))  # type: ignore
    grid = plt.GridSpec(  # type: ignore
        3, 2, figure=fig, hspace=0.4, wspace=0.3
    )

    # 1. Binary Loss - Batch Size vs Time plot
    ax1 = fig.add_subplot(grid[0, 0])  # type: ignore
    for gamma in binary_results["gamma"].unique():
        if gamma == 0:
            df_subset = binary_results[
                (binary_results["gamma"] == gamma)
                & (binary_results["loss_type"] == "BCEWithLogitsLoss")
                & (binary_results["feature_dim"] == 1)
            ]
            ax1.plot(  # type: ignore
                df_subset["batch_size"],
                df_subset["time_mean_ms"],
                "o-",
                label="BCE",
            )
        else:
            df_subset = binary_results[
                (binary_results["gamma"] == gamma)
                & (binary_results["loss_type"] == "BinaryFocalLoss")
                & (binary_results["feature_dim"] == 1)
            ]
            ax1.plot(  # type: ignore
                df_subset["batch_size"],
                df_subset["time_mean_ms"],
                "o-",
                label=f"BFL γ={gamma}",
            )

    ax1.set_title("Binary Loss Performance vs Batch Size")  # type: ignore
    ax1.set_xlabel("Batch Size")  # type: ignore
    ax1.set_ylabel("Time (ms)")  # type: ignore
    ax1.set_xscale("log")  # type: ignore
    ax1.set_yscale("log")  # type: ignore
    ax1.legend()  # type: ignore
    ax1.grid(True, which="both", linestyle="--", alpha=0.6)  # type: ignore

    # 2. MultiClass Loss - Batch Size vs Time plot
    ax2 = fig.add_subplot(grid[0, 1])  # type: ignore
    num_classes_ref = multiclass_results[
        "num_classes"
    ].min()  # Get a reference class count

    for gamma in multiclass_results["gamma"].unique():
        if gamma == 0:
            df_subset = multiclass_results[
                (multiclass_results["gamma"] == gamma)
                & (multiclass_results["loss_type"] == "CrossEntropyLoss")
                & (multiclass_results["num_classes"] == num_classes_ref)
            ]
            ax2.plot(  # type: ignore
                df_subset["batch_size"],
                df_subset["time_mean_ms"],
                "o-",
                label="CE",
            )
        else:
            df_subset = multiclass_results[
                (multiclass_results["gamma"] == gamma)
                & (multiclass_results["loss_type"] == "MultiClassFocalLoss")
                & (multiclass_results["num_classes"] == num_classes_ref)
            ]
            ax2.plot(  # type: ignore
                df_subset["batch_size"],
                df_subset["time_mean_ms"],
                "o-",
                label=f"MCFL γ={gamma}",
            )

    ax2.set_title(  # type: ignore
        "MultiClass Loss Performance vs Batch Size\n"
        f"(Classes={num_classes_ref})"
    )
    ax2.set_xlabel("Batch Size")  # type: ignore
    ax2.set_ylabel("Time (ms)")  # type: ignore
    ax2.set_xscale("log")  # type: ignore
    ax2.set_yscale("log")  # type: ignore
    ax2.legend()  # type: ignore
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)  # type: ignore

    # 3. Binary Loss - Effect of Feature Dimensions
    ax3 = fig.add_subplot(grid[1, 0])  # type: ignore
    batch_size_ref = binary_results[
        "batch_size"
    ].median()  # Get a reference batch size

    # Group by feature_dim and loss_type, then calculate mean of relative_time
    feature_dim_data = (
        binary_results[binary_results["batch_size"] == batch_size_ref]
        .groupby(["feature_dim", "loss_type", "gamma"])["time_mean_ms"]
        .mean()
        .reset_index()
    )

    # Plot for each gamma value
    for gamma in feature_dim_data["gamma"].unique():
        if gamma == 0:
            df_subset = feature_dim_data[
                (feature_dim_data["gamma"] == gamma)
                & (feature_dim_data["loss_type"] == "BCEWithLogitsLoss")
            ]
            ax3.plot(  # type: ignore
                df_subset["feature_dim"],
                df_subset["time_mean_ms"],
                "o-",
                label="BCE",
            )
        else:
            df_subset = feature_dim_data[
                (feature_dim_data["gamma"] == gamma)
                & (feature_dim_data["loss_type"] == "BinaryFocalLoss")
            ]
            ax3.plot(  # type: ignore
                df_subset["feature_dim"],
                df_subset["time_mean_ms"],
                "o-",
                label=f"BFL γ={gamma}",
            )

    ax3.set_title(  # type: ignore
        "Binary Loss Performance vs Feature Dimensions\n"
        f"(Batch Size={batch_size_ref})"
    )
    ax3.set_xlabel("Feature Dimensions")  # type: ignore
    ax3.set_ylabel("Time (ms)")  # type: ignore
    ax3.legend()  # type: ignore
    ax3.grid(True, which="both", linestyle="--", alpha=0.6)  # type: ignore

    # 4. MultiClass Loss - Effect of Number of Classes
    ax4 = fig.add_subplot(grid[1, 1])  # type: ignore
    batch_size_ref = multiclass_results[
        "batch_size"
    ].median()  # Get a reference batch size

    # Group by num_classes and loss_type, gamma, then calculate mean time
    class_count_data = (
        multiclass_results[multiclass_results["batch_size"] == batch_size_ref]
        .groupby(["num_classes", "loss_type", "gamma"])["time_mean_ms"]
        .mean()
        .reset_index()
    )

    # Plot for each gamma value
    for gamma in class_count_data["gamma"].unique():
        if gamma == 0:
            df_subset = class_count_data[
                (class_count_data["gamma"] == gamma)
                & (class_count_data["loss_type"] == "CrossEntropyLoss")
            ]
            ax4.plot(  # type: ignore
                df_subset["num_classes"],
                df_subset["time_mean_ms"],
                "o-",
                label="CE",
            )
        else:
            df_subset = class_count_data[
                (class_count_data["gamma"] == gamma)
                & (class_count_data["loss_type"] == "MultiClassFocalLoss")
            ]
            ax4.plot(  # type: ignore
                df_subset["num_classes"],
                df_subset["time_mean_ms"],
                "o-",
                label=f"MCFL γ={gamma}",
            )

    ax4.set_title(  # type: ignore
        "MultiClass Loss Performance vs Number of Classes\n"
        f"(Batch Size={batch_size_ref})"
    )
    ax4.set_xlabel("Number of Classes")  # type: ignore
    ax4.set_ylabel("Time (ms)")  # type: ignore
    ax4.legend()  # type: ignore
    ax4.grid(True, which="both", linestyle="--", alpha=0.6)  # type: ignore

    # 5. Effect of gamma parameter on relative performance
    ax5 = fig.add_subplot(grid[2, 0])  # type: ignore

    # For binary focal loss
    gamma_effect_binary = (
        binary_results[
            (binary_results["feature_dim"] == 1)
            & (binary_results["loss_type"] == "BinaryFocalLoss")
        ]
        .groupby("gamma")["relative_time"]
        .mean()
        .reset_index()
    )

    ax5.plot(  # type: ignore
        gamma_effect_binary["gamma"],
        gamma_effect_binary["relative_time"],
        "o-b",
        label="BinaryFocalLoss",
    )

    # For multiclass focal loss
    gamma_effect_multi = (
        multiclass_results[
            multiclass_results["loss_type"] == "MultiClassFocalLoss"
        ]
        .groupby("gamma")["relative_time"]
        .mean()
        .reset_index()
    )

    ax5.plot(  # type: ignore
        gamma_effect_multi["gamma"],
        gamma_effect_multi["relative_time"],
        "o-r",
        label="MultiClassFocalLoss",
    )

    ax5.set_title(  # type: ignore
        "Effect of γ Parameter on Relative Performance"
    )
    ax5.set_xlabel("γ Value")  # type: ignore
    ax5.set_ylabel("Relative Time (compared to standard loss)")  # type: ignore
    ax5.axhline(y=1.0, color="k", linestyle="--", alpha=0.7)  # type: ignore
    ax5.legend()  # type: ignore
    ax5.grid(True)  # type: ignore

    # 6. Memory usage comparison if available
    ax6 = fig.add_subplot(grid[2, 1])  # type: ignore

    # Filter results with valid memory measurements
    binary_mem = binary_results[binary_results["memory_bytes"] > 0]
    multiclass_mem = multiclass_results[multiclass_results["memory_bytes"] > 0]

    if len(binary_mem) > 0 and len(multiclass_mem) > 0:
        # For binary focal loss
        binary_mem_summary = (
            binary_mem.groupby(["loss_type", "gamma"])["memory_bytes"]
            .mean()
            .reset_index()
        )

        # Calculate relative memory usage
        binary_ref_mem = binary_mem_summary[
            binary_mem_summary["loss_type"] == "BCEWithLogitsLoss"
        ]["memory_bytes"].values[0]
        binary_mem_summary["relative_memory"] = (
            binary_mem_summary["memory_bytes"] / binary_ref_mem
        )

        # Plot binary results
        binary_focal_mem = binary_mem_summary[
            binary_mem_summary["loss_type"] == "BinaryFocalLoss"
        ]
        ax6.plot(  # type: ignore
            binary_focal_mem["gamma"],
            binary_focal_mem["relative_memory"],
            "o-b",
            label="BinaryFocalLoss",
        )

        # For multiclass focal loss
        multiclass_mem_summary = (
            multiclass_mem.groupby(["loss_type", "gamma"])["memory_bytes"]
            .mean()
            .reset_index()
        )

        # Calculate relative memory usage
        multiclass_ref_mem = multiclass_mem_summary[
            multiclass_mem_summary["loss_type"] == "CrossEntropyLoss"
        ]["memory_bytes"].values[0]
        multiclass_mem_summary["relative_memory"] = (
            multiclass_mem_summary["memory_bytes"] / multiclass_ref_mem
        )

        # Plot multiclass results
        multiclass_focal_mem = multiclass_mem_summary[
            multiclass_mem_summary["loss_type"] == "MultiClassFocalLoss"
        ]
        ax6.plot(  # type: ignore
            multiclass_focal_mem["gamma"],
            multiclass_focal_mem["relative_memory"],
            "o-r",
            label="MultiClassFocalLoss",
        )

        ax6.set_title("Memory Usage vs γ Parameter")  # type: ignore
        ax6.set_xlabel("γ Value")  # type: ignore
        ax6.set_ylabel("Relative Memory Usage")  # type: ignore
        ax6.axhline(  # type: ignore
            y=1.0, color="k", linestyle="--", alpha=0.7
        )
        ax6.legend()  # type: ignore
        ax6.grid(True)  # type: ignore
    else:
        ax6.text(  # type: ignore
            0.5,
            0.5,
            "Memory usage data not available\n(CUDA not used)",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax6.transAxes,
        )
        ax6.axis("off")

    # Add a title for the entire figure
    plt.suptitle(  # type: ignore
        "Benchmark Performance: Focal Loss vs Standard Loss Functions",
        fontsize=16,
    )

    # Save plot if needed
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")  # type: ignore

    # return plot
    return fig


def run_advanced_benchmarks():
    """Run advanced benchmarks"""
    # start clock
    time0 = time()

    # Determine if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmarks on: {device}")

    # Binary classification parameters
    binary_batch_sizes = [32, 64, 128, 256, 512, 1024, 2048]
    binary_gammas = [0.0, 1.0, 2.0, 5.0]
    binary_feature_dims = [1, 3, 10, 20]  # 1 for binary, >1 for multi-label

    # Multi-class classification parameters
    multiclass_batch_sizes = [32, 64, 128, 256, 512, 1024, 2048]
    multiclass_gammas = [0.0, 1.0, 2.0, 5.0]
    multiclass_class_counts = [2, 5, 10, 20, 50, 100]

    # Run benchmarks
    print("\nRunning binary classification benchmarks...")
    binary_results = benchmark_binary_focal_loss(
        batch_sizes=binary_batch_sizes,
        gammas=binary_gammas,
        feature_dims=binary_feature_dims,
        device=device,
    )
    time1 = time()
    print(f"Benchmarks took {(time1 - time0):.2f} s.")

    print("\nRunning multi-class classification benchmarks...")
    multiclass_results = benchmark_multiclass_focal_loss(
        batch_sizes=multiclass_batch_sizes,
        gammas=multiclass_gammas,
        class_counts=multiclass_class_counts,
        device=device,
    )
    time2 = time()
    print(f"Benchmarks took {(time2 - time1):.2f} s.")

    # Save results to CSV
    binary_results.to_csv(
        "./benchmarking/results/binary_focal_loss_benchmark.csv", index=False
    )
    multiclass_results.to_csv(
        "./benchmarking/results/multiclass_focal_loss_benchmark.csv",
        index=False,
    )

    # Plot results
    try:
        fig = plot_benchmark_results(
            binary_results,
            multiclass_results,
            save_path=(
                "./benchmarking/results/focal_loss_benchmark_results.png"
            ),
        )
        print(
            "\nBenchmark plots saved to "
            "'./benchmarking/results/focal_loss_benchmark_results.png'"
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        fig = None
        print(f"\nFailed to generate plots: {e}")

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    # Binary classification summary
    print("\nBinary Focal Loss Performance:")
    binary_summary = (
        binary_results[binary_results["feature_dim"] == 1]  # type: ignore
        .groupby(["loss_type", "gamma"])["relative_time"]
        .mean()
        .reset_index()
    )
    for _, row in binary_summary.iterrows():  # type: ignore
        loss_type = row["loss_type"]  # type: ignore
        gamma = row["gamma"]  # type: ignore
        rel_time = row["relative_time"]  # type: ignore
        if loss_type == "BCEWithLogitsLoss":
            print(f"- {loss_type}: 1.00x (baseline)")
        else:
            print(
                f"- {loss_type} (gamma={gamma}): "
                f"{rel_time:.2f}x slower than BCE"
            )

    # Multi-label classification summary
    print("\nMulti-Label Classification Performance:")
    multilabel_summary = (
        binary_results[binary_results["feature_dim"] > 1]  # type: ignore
        .groupby(["loss_type", "gamma"])["relative_time"]
        .mean()
        .reset_index()
    )
    for _, row in multilabel_summary.iterrows():  # type: ignore
        loss_type = row["loss_type"]  # type: ignore
        gamma = row["gamma"]  # type: ignore
        rel_time = row["relative_time"]  # type: ignore
        if loss_type == "BCEWithLogitsLoss":
            print(f"- {loss_type}: 1.00x (baseline)")
        else:
            print(
                f"- {loss_type} (gamma={gamma}): "
                f"{rel_time:.2f}x slower than BCE"
            )

    # Multi-class classification summary
    print("\nMulti-Class Focal Loss Performance:")
    multiclass_summary = (
        multiclass_results.groupby(["loss_type", "gamma"])[  # type: ignore
            "relative_time"
        ]
        .mean()
        .reset_index()
    )
    for _, row in multiclass_summary.iterrows():  # type: ignore
        loss_type = row["loss_type"]  # type: ignore
        gamma = row["gamma"]  # type: ignore
        rel_time = row["relative_time"]  # type: ignore
        if loss_type == "CrossEntropyLoss":
            print(f"- {loss_type}: 1.00x (baseline)")
        else:
            print(
                f"- {loss_type} (gamma={gamma}): "
                f"{rel_time:.2f}x slower than CE"
            )
    time3 = time()
    print(
        "\nBenchmarking complete! "
        f"Took {(time3 - time0):.2f} s total."
        "\n\nResults saved to CSV files in './benchmarking/results/'."
        "\nFigure saved as "
        "'./benchmarking/results/focal_loss_benchmark_results.png'."
        "\n\nIf open, close figure to terminate script."
    )

    # show figure if available
    if fig is not None:
        plt.show(block=True)  # type: ignore


if __name__ == "__main__":
    run_advanced_benchmarks()
