# PyTorch Focal Loss Benchmarks

This directory contains benchmarking scripts for the `pytorch-focalloss` package, comparing its performance to the standard PyTorch loss functions.

## Benchmarks Available

1. **Basic Benchmark** (`benchmark_focal_loss.py`): Measures the raw execution time of focal loss functions compared to standard losses across various batch sizes and parameter configurations.

2. **Advanced Benchmark** (`benchmark_advanced.py`): Provides comprehensive analysis with detailed visualizations, memory usage tracking, and performance metrics across different parameters.

3. **Training Loop Benchmark** (`benchmark_in_training_loop.py`): Assesses the impact of focal loss on actual training time by simulating a realistic training scenario with a simple CNN model.

## Requirements

- PyTorch >= 1.10.0
- pytorch-focalloss
- matplotlib (for visualization)
- pandas (for data processing)
- numpy

Install requirements with:
```bash
pip install pytorch-focalloss matplotlib pandas numpy
```

## Usage

Run any of the benchmark scripts:

```bash
# Basic benchmark
python benchmark_focal_loss.py

# Advanced benchmark with visualizations
python benchmark_advanced.py

# Training loop benchmark
python benchmark_in_training_loop.py
```

## What's Being Benchmarked

The benchmarks compare:

- `BinaryFocalLoss` vs. `BCEWithLogitsLoss` for binary classification
- `BinaryFocalLoss` vs. `BCEWithLogitsLoss` for multi-label classification
- `MultiClassFocalLoss` vs. `CrossEntropyLoss` for multi-class classification

Each comparison is run with different:
- Batch sizes (from 32 to 2048)
- Gamma values (0, 1, 2, 5)
- Number of classes / feature dimensions (for multi-class/multi-label)

## Output

The benchmarks provide:

- Execution time measurements
- Relative performance comparisons
- Memory usage analysis (CUDA only)
- Detailed visualizations (in `benchmark_advanced.py`)
- CSV exports of all benchmark data
- Training loop performance impact (in `benchmark_in_training_loop.py`)

## Interpreting Results

Typical results show:

- Focal loss is generally slower than standard cross-entropy loss
- Performance impact increases with γ (gamma) value
- Memory usage is typically higher for focal loss
- Batch size significantly affects relative performance
- Multi-class focal loss has higher overhead than binary focal loss
