"""
Training loop benchmark for torch_focalloss.

This benchmark assesses the impact of focal loss on actual training time
by comparing a simple CNN model trained with:
- BinaryFocalLoss vs BCEWithLogitsLoss for binary image classification
- MultiClassFocalLoss vs CrossEntropyLoss for multi-class image classification

The benchmark simulates training on a small subset of a dataset
to measure performance in a realistic training scenario.
"""

import time
from typing import Any

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from torch_focalloss import BinaryFocalLoss, MultiClassFocalLoss

# Set random seed for reproducibility
torch.manual_seed(42)  # type: ignore


class SimpleCNN(nn.Module):
    """Simple CNN model for image classification"""

    def __init__(self, num_classes: int = 1, input_channels: int = 3) -> None:
        super(SimpleCNN, self).__init__()  # type: ignore

        # Simple CNN architecture
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layer
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x (torch.Tensor): tensor to evaluate

        Returns:
            torch.Tensor: model prediction on input
        """
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_synthetic_dataset(
    num_samples: int = 1000,
    num_classes: int = 10,
    image_size: int = 32,
    binary: bool = False,
):
    """Create a synthetic dataset for testing"""

    # Create random images
    images = torch.randn(num_samples, 3, image_size, image_size)

    if binary:
        # Create binary labels
        labels = torch.randint(0, 2, (num_samples,), dtype=torch.float32)
    else:
        # Create multi-class labels
        labels = torch.randint(0, num_classes, (num_samples,))

    return TensorDataset(images, labels)


def train_with_loss(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: str | torch.device,
    num_epochs: int = 2,
) -> dict[str, Any]:
    """Train the model with the specified loss function and measure time"""

    # Reset model weights to ensure fair comparison
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    model.to(device)
    model.train()

    # Dictionary to store metrics
    metrics: dict[str, list[float] | float] = {
        "epoch_times": [],
        "batch_times": [],
        "loss_values": [],
    }

    # Training loop
    start_time = time.time()
    for _ in range(num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0

        for _, (inputs, targets) in enumerate(train_loader):
            batch_start = time.time()

            inputs = inputs.to(device)
            targets = targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update metrics
            batch_time = time.time() - batch_start
            metrics["batch_times"].append(batch_time)  # type: ignore
            epoch_loss += loss.item()
            metrics["loss_values"].append(loss.item())  # type: ignore

        # Calculate epoch metrics
        epoch_time = time.time() - epoch_start
        metrics["epoch_times"].append(epoch_time)  # type: ignore

    # Calculate total training time
    total_time = time.time() - start_time
    metrics["total_time"] = total_time

    return metrics


def benchmark_binary_classification() -> dict[str, Any]:
    """Benchmark BinaryFocalLoss vs BCEWithLogitsLoss in a training loop"""

    print("\n" + "=" * 80)
    print("BENCHMARKING BINARY CLASSIFICATION")
    print("=" * 80)

    # Parameters
    batch_size = 64
    num_samples = 1000
    learning_rate = 0.001
    num_epochs = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create synthetic dataset
    dataset = create_synthetic_dataset(
        num_samples=num_samples, image_size=32, binary=True
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    model = SimpleCNN(num_classes=1)

    # Define loss functions to test
    bce_loss = nn.BCEWithLogitsLoss()

    focal_losses = [
        ("BinaryFocalLoss (γ=0)", BinaryFocalLoss(gamma=0)),
        ("BinaryFocalLoss (γ=1)", BinaryFocalLoss(gamma=1)),
        ("BinaryFocalLoss (γ=2)", BinaryFocalLoss(gamma=2)),
    ]

    # Dictionary to store results
    results: dict[str, Any] = {}

    # Benchmark BCE Loss
    print("\nTraining with BCEWithLogitsLoss...")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    bce_metrics = train_with_loss(
        model, train_loader, bce_loss, optimizer, device, num_epochs
    )
    results["BCEWithLogitsLoss"] = bce_metrics

    # Benchmark Focal Loss variants
    for name, focal_loss in focal_losses:
        print(f"\nTraining with {name}...")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        focal_metrics = train_with_loss(
            model, train_loader, focal_loss, optimizer, device, num_epochs
        )
        results[name] = focal_metrics

    # Print summary
    print("\nBinary Classification Results:")
    print("-" * 40)
    print(
        f"{'Loss Function':<30} | {'Total Time':<10} | {'Relative Time':<15}"
    )
    print("-" * 40)

    # Calculate relative time compared to BCE
    bce_time = results["BCEWithLogitsLoss"]["total_time"]

    for name, metrics in results.items():
        total_time = metrics["total_time"]
        relative_time = total_time / bce_time
        print(f"{name:<30} | {total_time:.4f}s | {relative_time:.4f}x")

    return results


def benchmark_multiclass_classification() -> dict[str, Any]:
    """Benchmark MultiClassFocalLoss vs CrossEntropyLoss in a training loop"""

    print("\n" + "=" * 80)
    print("BENCHMARKING MULTI-CLASS CLASSIFICATION")
    print("=" * 80)

    # Parameters
    batch_size = 64
    num_samples = 1000
    num_classes = 10
    learning_rate = 0.001
    num_epochs = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create synthetic dataset
    dataset = create_synthetic_dataset(
        num_samples=num_samples, num_classes=num_classes, image_size=32
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    model = SimpleCNN(num_classes=num_classes)

    # Define loss functions to test
    ce_loss = nn.CrossEntropyLoss()

    focal_losses = [
        ("MultiClassFocalLoss (γ=0)", MultiClassFocalLoss(gamma=0)),
        ("MultiClassFocalLoss (γ=1)", MultiClassFocalLoss(gamma=1)),
        ("MultiClassFocalLoss (γ=2)", MultiClassFocalLoss(gamma=2)),
    ]

    # Dictionary to store results
    results: dict[str, Any] = {}

    # Benchmark CE Loss
    print("\nTraining with CrossEntropyLoss...")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    ce_metrics = train_with_loss(
        model, train_loader, ce_loss, optimizer, device, num_epochs
    )
    results["CrossEntropyLoss"] = ce_metrics

    # Benchmark Focal Loss variants
    for name, focal_loss in focal_losses:
        print(f"\nTraining with {name}...")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        focal_metrics = train_with_loss(
            model, train_loader, focal_loss, optimizer, device, num_epochs
        )
        results[name] = focal_metrics

    # Print summary
    print("\nMulti-Class Classification Results:")
    print("-" * 40)
    print(
        f"{'Loss Function':<30} | {'Total Time':<10} | {'Relative Time':<15}"
    )
    print("-" * 40)

    # Calculate relative time compared to CE
    ce_time = results["CrossEntropyLoss"]["total_time"]

    for name, metrics in results.items():
        total_time = metrics["total_time"]
        relative_time = total_time / ce_time
        print(f"{name:<30} | {total_time:.4f}s | {relative_time:.4f}x")

    return results


def batch_time_analysis(
    binary_results: dict[str, Any], multiclass_results: dict[str, Any]
):
    """Analyze batch processing times"""

    print("\n" + "=" * 80)
    print("BATCH TIME ANALYSIS")
    print("=" * 80)

    # Binary classification batch times
    print("\nBinary Classification - Average Batch Time:")
    for name, metrics in binary_results.items():
        avg_batch_time = sum(metrics["batch_times"]) / len(
            metrics["batch_times"]
        )
        print(f"{name:<30}: {avg_batch_time * 1000:.3f} ms")

    # Multi-class classification batch times
    print("\nMulti-Class Classification - Average Batch Time:")
    for name, metrics in multiclass_results.items():
        avg_batch_time = sum(metrics["batch_times"]) / len(
            metrics["batch_times"]
        )
        print(f"{name:<30}: {avg_batch_time * 1000:.3f} ms")


def run_training_benchmarks():
    """Run all training loop benchmarks"""

    print("\n" + "=" * 80)
    print("BENCHMARKING FOCAL LOSS IN TRAINING LOOPS")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning benchmarks on: {device}")

    # Run binary classification benchmark
    binary_results = benchmark_binary_classification()

    # Run multi-class classification benchmark
    multiclass_results = benchmark_multiclass_classification()

    # Analyze batch times
    batch_time_analysis(binary_results, multiclass_results)

    print("\nTraining loop benchmarks complete!")


if __name__ == "__main__":
    run_training_benchmarks()
