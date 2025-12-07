# Ground Truth Generation Module

This module provides comprehensive ground truth extraction and preprocessing functionality for CoLog's log anomaly detection system. It handles log parsing, sequence construction, labeling, and dataset preparation across multiple log dataset types.

## Overview

The ground truth module processes raw log files and generates structured training/validation/test datasets for the CoLog neural network. It supports multiple dataset types with different parsing and labeling strategies, handles class imbalance, and provides embedding computation capabilities.

### Key Features

- **Multi-strategy extraction**: Supports 4 different dataset types with specialized labeling approaches
- **Integrated preprocessing**: Built-in log parsing using Drain and NER parsers
- **Embedding computation**: Automatic generation of message embeddings using SentenceTransformer models
- **Sequence building**: Context-aware sequence construction with configurable window sizes
- **Class imbalance handling**: Multiple resampling methods (SMOTE, TomekLinks, etc.)
- **Reproducibility**: Random seed control for deterministic splits and resampling
- **Dry-run mode**: Preview operations without modifying files

## Directory Structure

```
groundtruth/
├── main.py                          # Main entry point for ground truth generation
├── extraction/                      # Ground truth extraction logic
│   ├── __init__.py
│   ├── main.py                      # GroundTruth facade class
│   ├── groundtruth_base.py          # Base extractor with shared functionality
│   ├── groundtruth_type1.py         # Type1: WARN-based labels (hadoop, zookeeper)
│   ├── groundtruth_type2.py         # Type2: Wordlist-based labels (spark, windows)
│   ├── groundtruth_type3.py         # Type3: Label column (bgl)
│   ├── groundtruth_type4.py         # Type4: NER parser (casper-rw, dfrws*, honeynet*)
│   └── groundtruth_aggregator.py    # Multi-dataset aggregation
├── preprocessing/                   # Log preprocessing and parsing
│   ├── __init__.py
│   └── preprocessor.py              # Drain/NER parsing and embedding computation
├── sampling/                        # Class imbalance handling
│   ├── __init__.py
│   └── sampler.py                   # ClassImbalanceSolver for resampling
├── utils/                           # Utilities and configuration
│   ├── __init__.py
│   ├── cli.py                       # Command-line argument parsing
│   ├── constants.py                 # System-wide constants
│   ├── dataclasses.py               # Data structures
│   ├── exceptions.py                # Custom exceptions
│   ├── settings_provider.py         # Settings interface
│   ├── settings.py                  # Dataset configurations
│   ├── validators.py                # Input validation
│   └── logparsers/                  # Log parsing implementations
│       ├── Drain/                   # Drain parser
│       └── nerlogparser/            # NER-based parser
└── wordlists/                       # Anomaly detection wordlists
    ├── auth.txt
    ├── daemon.txt
    ├── spark.txt
    ├── windows.txt
    └── ...
```

## Dataset Types

The module supports four distinct extraction strategies based on dataset characteristics:

### Type 1: Level-based Labeling
- **Datasets**: `hadoop`, `zookeeper`
- **Strategy**: Uses the `Level` column from parsed CSV files
- **Labeling**: `WARN` → anomaly (0), others → normal (1)
- **Parser**: Drain

### Type 2: Wordlist-based Labeling
- **Datasets**: `spark`, `windows`
- **Strategy**: Searches for keywords from dataset-specific wordlists
- **Labeling**: Keyword match → anomaly (0), no match → normal (1)
- **Parser**: Drain

### Type 3: Label Column
- **Datasets**: `bgl`
- **Strategy**: Uses existing `Label` column in parsed data
- **Labeling**: `-` → normal (1), other values → anomaly (0)
- **Parser**: Drain

### Type 4: NER-based Parsing
- **Datasets**: `casper-rw`, `dfrws-2009-jhuisi`, `dfrws-2009-nssal`, `honeynet-challenge5`, `honeynet-challenge7`
- **Strategy**: Named Entity Recognition for structured parsing
- **Labeling**: Token-level parsing determines anomalies
- **Parser**: NER (nerlogparser)

## Usage

### Basic Usage

```bash
# Extract ground truth for Hadoop dataset with default settings
python main.py --dataset hadoop

# Extract with custom sequence type and window size
python main.py --dataset spark --sequence-type context --window-size 2

# Use specific embedding model and device
python main.py --dataset bgl --model all-MiniLM-L6-v2 --device cuda --batch-size 128

# Force regeneration even if files exist
python main.py --dataset zookeeper --force

# Dry run to preview operations
python main.py --dataset windows --dry-run
```

### Advanced Usage

```bash
# Enable class imbalance resampling with TomekLinks
python main.py --dataset hadoop --resample --resample-method TomekLinks

# Custom train/validation split ratios
python main.py --dataset spark --train-ratio 0.7 --valid-ratio 0.15

# Set random seed for reproducibility
python main.py --dataset bgl --random-seed 42

# Enable groundbreaking mode (multi-label/sequence-level labeling)
python main.py --dataset zookeeper --groundbreaking

# Verbose logging for debugging
python main.py --dataset hadoop --verbose
```

### Multiple Datasets

```bash
# Process datasets sequentially
python main.py --dataset hadoop
python main.py --dataset spark
python main.py --dataset zookeeper
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | `hadoop` | Dataset name to process |
| `--dataset-dir` | str | `datasets/` | Root directory for datasets |
| `--model` | str | `all-MiniLM-L6-v2` | SentenceTransformer model for embeddings |
| `--batch-size` | int | `64` | Batch size for embedding computation |
| `--device` | str | `auto` | Device for computation (`cpu`, `cuda`, `auto`) |
| `--sequence-type` | str | `context` | Sequence type (`background` or `context`) |
| `--window-size` | int | `1` | Number of messages on each side of current message |
| `--train-ratio` | float | `0.6` | Training set fraction |
| `--valid-ratio` | float | `0.2` | Validation set fraction |
| `--force` | flag | `False` | Force regeneration of existing files |
| `--verbose` | flag | `False` | Enable DEBUG logging |
| `--groundbreaking` | flag | `False` | Enable multi-label/sequence-level labeling |
| `--resample` | flag | `False` | Run resampling after extraction |
| `--resample-method` | str | `None` | Resampling method (e.g., `TomekLinks`, `SMOTE`) |
| `--random-seed` | int | `100` | Random seed for reproducibility |
| `--dry-run` | flag | `False` | Preview without executing |

## Sequence Types

### Background Sequence
- **Format**: `[previous_message, current_message]`
- **Use case**: Simple context using only historical messages
- **Example** (window_size=1): `[msg_i-1, msg_i]`

### Context Sequence
- **Format**: `[left_context, current_message, right_context]`
- **Use case**: Full bidirectional context
- **Example** (window_size=1): `[msg_i-1, msg_i, msg_i+1]`
- **Example** (window_size=2): `[msg_i-2, msg_i-1, msg_i, msg_i+1, msg_i+2]`

Padding is applied using the `UNK` token for messages at dataset boundaries.

## Output Files

The module generates the following files in each dataset's `groundtruth/` directory:

### Core Files
- **`messages.p`**: Dictionary mapping message ID → tokenized message (NumPy array)
- **`sequences.p`**: Dictionary mapping message ID → sequence of surrounding messages
- **`labels.p`**: Dictionary mapping message ID → label (0=anomaly, 1=normal)
- **`keys.p`**: Ordered list of all message IDs

### Dataset Splits
- **`train_set.p`**: Training dataset split
- **`valid_set.p`**: Validation dataset split
- **`test_set.p`**: Test dataset split (or `generalization_set.p` for some datasets)

### Resampled Data (if `--resample` used)
- **`resampled_groundtruth/<method>/`**: Directory containing resampled datasets
  - `train_set.p`, `valid_set.p`, `test_set.p`

### Embeddings
- **`log_embeddings/<dataset>_embeddings.p`**: Precomputed message embeddings

## Preprocessing Pipeline

1. **Log Collection**: Gather raw log files from `datasets/<dataset>/logs/`
2. **Parsing**: 
   - Drain parser: Extract structured fields using log format patterns
   - NER parser: Token-level Named Entity Recognition
3. **Embedding**: Compute semantic embeddings using SentenceTransformer
4. **Sequence Building**: Construct context windows around each message
5. **Labeling**: Assign anomaly labels based on dataset-specific strategy
6. **Splitting**: Divide into train/validation/test sets
7. **Resampling** (optional): Balance class distribution

## Class Imbalance Handling

The `ClassImbalanceSolver` supports multiple resampling strategies from `imbalanced-learn`:

### Available Methods
- **TomekLinks**: Remove majority class samples that are close to minority class
- **SMOTE**: Synthetic Minority Over-sampling Technique
- **ADASYN**: Adaptive Synthetic Sampling
- **BorderlineSMOTE**: SMOTE variant focusing on borderline cases
- **RandomOverSampler**: Random duplication of minority class
- **RandomUnderSampler**: Random removal from majority class
- **And more**: See `utils/constants.py` for full list

### Usage
```bash
python main.py --dataset hadoop --resample --resample-method TomekLinks
```

The solver attempts to reuse precomputed embeddings from `log_embeddings/` for efficiency.

## Configuration

Dataset-specific configurations are defined in `utils/settings.py`:

```python
'hadoop': {
    'in_dir': 'hadoop/logs/',
    'out_dir': 'hadoop/logs_structured/',
    'embs_dir': 'hadoop/log_embeddings/',
    'groundtruth_dir': 'hadoop/groundtruth/',
    'log_format': '<Date> <Time> <Level> \\[<Process>\\] <Component>: <Content>',
    'regex': [r'(\d+\.){3}\d+'],  # Mask IP addresses
    'st': 0.5,      # Similarity threshold
    'depth': 4      # Parse tree depth
}
```

### Adding New Datasets

1. Add configuration to `settings` dict in `utils/settings.py`
2. Add dataset name to appropriate type list in `utils/constants.py`
3. Create wordlist file in `wordlists/` if using Type 2 labeling
4. Place raw logs in `datasets/<new_dataset>/logs/`

## Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `tqdm`: Progress bars
- `sentence-transformers`: Embedding computation
- `torch`: PyTorch backend
- `imbalanced-learn`: Class imbalance handling
- `scikit-learn`: Machine learning utilities

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Logging

The module uses Python's `logging` framework:

- **Default**: INFO level (shows progress and major steps)
- **Verbose** (`--verbose`): DEBUG level (detailed operation logs)
- **Dry-run**: Shows what would be executed without making changes

## Error Handling

Custom exceptions in `utils/exceptions.py`:
- `ValidationError`: Invalid input parameters
- `FileNotFoundError`: Missing required files
- `DataFormatError`: Malformed data
- `ConfigurationError`: Invalid configuration

## Performance Considerations

- **Embedding computation**: Most time-consuming step; use GPU (`--device cuda`) for speedup
- **Batch size**: Larger batches faster but require more memory
- **Preprocessing cache**: Reuses precomputed embeddings when available
- **Memory management**: Explicit garbage collection after major operations

## Examples

### Complete Workflow for Hadoop
```bash
# Step 1: Extract ground truth with custom settings
python main.py --dataset hadoop \
    --sequence-type context \
    --window-size 2 \
    --model all-MiniLM-L6-v2 \
    --batch-size 128 \
    --device cuda \
    --train-ratio 0.7 \
    --valid-ratio 0.15 \
    --random-seed 42 \
    --verbose

# Step 2: Apply class imbalance resampling
python main.py --dataset hadoop \
    --resample \
    --resample-method TomekLinks \
    --random-seed 42
```

### Batch Processing Multiple Datasets
```bash
# Process all Type 1 datasets
for dataset in hadoop zookeeper; do
    python main.py --dataset $dataset --force --verbose
done

# Process with resampling
for dataset in hadoop spark bgl; do
    python main.py --dataset $dataset --resample --resample-method SMOTE
done
```

## Troubleshooting

### Issue: "Dataset not found"
- Ensure raw logs exist in `datasets/<dataset>/logs/`
- Check dataset name spelling matches configuration

### Issue: "Out of memory during embedding"
- Reduce `--batch-size`
- Use `--device cpu` instead of GPU
- Process smaller chunks of data

### Issue: "Parser fails on log format"
- Verify `log_format` regex in `utils/settings.py`
- Check raw log files match expected format
- Enable `--verbose` to see parsing errors

### Issue: "Resampling produces poor results"
- Try different resampling methods
- Adjust train/valid ratios
- Check class distribution in original data

## Related Modules

- **`train/`**: Training pipeline that consumes ground truth files
- **`neuralnetwork/`**: Model architectures used with ground truth data
- **`utils/`**: Shared utilities for metrics and predictions

## License

Part of the CoLog project. See main project README for license information.

## Authors

Developed for the CoLog log anomaly detection system as part of thesis research.

## Version

Current version: 1.0.0

Last updated: December 2025
