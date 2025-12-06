# Datasets Directory

This directory contains the log datasets used for training, validation, and testing the CoLog log anomaly detection system. The datasets represent diverse system logs from various software systems and computing environments, providing comprehensive coverage for anomaly detection research.

## Overview

The datasets folder stores raw log files organized by system type. Each dataset contains system logs that may include both normal operational logs and anomalous logs representing system failures, security incidents, or performance issues. These logs are processed by the CoLog pipeline to extract features, generate ground truth labels, and train anomaly detection models.

### Key Features

- **10 diverse datasets**: Covering distributed systems, security forensics, and enterprise applications
- **Multiple log formats**: Structured and semi-structured log formats
- **Real-world data**: Authentic system logs from production and research environments
- **Labeled anomalies**: Ground truth labels derived from system-specific indicators
- **Scalable storage**: Organized structure for easy dataset management
- **Processing pipeline ready**: Compatible with Drain and NER parsers

## Directory Structure

```
datasets/
├── hadoop/                          # Apache Hadoop MapReduce logs
│   └── logs/                        # Raw log files (container logs)
├── spark/                           # Apache Spark distributed processing logs
│   └── logs/                        # Raw log files (container logs)
├── zookeeper/                       # Apache ZooKeeper coordination service logs
│   └── logs/                        # Raw log files (Zookeeper.log)
├── bgl/                             # Blue Gene/L supercomputer logs
│   └── logs/                        # Raw log files with labels
├── windows/                         # Windows system logs (CBS)
│   └── logs/                        # Raw log files (windows.log)
├── casper-rw/                       # Casper-RW forensic challenge logs
│   └── logs/                        # Raw log files (auth, daemon, etc.)
├── dfrws-2009-jhuisi/              # DFRWS 2009 forensic challenge (JHU/ISI)
│   └── logs/                        # Raw log files (auth, daemon, debug)
├── dfrws-2009-nssal/               # DFRWS 2009 forensic challenge (NSSAL)
│   └── logs/                        # Raw log files (multiple syslog types)
├── honeynet-challenge5/            # Honeynet Project Challenge 5
│   └── logs/                        # Raw log files (security logs)
└── honeynet-challenge7/            # Honeynet Project Challenge 7
    └── logs/                        # Raw log files (security logs)
```

## Dataset Categories

### Type 1: Level-Based Labeling (WARN Detection)
Datasets where anomalies are identified by log level (e.g., WARN, ERROR).

#### Hadoop
- **Source**: Apache Hadoop MapReduce framework
- **Description**: Distributed data processing logs from YARN containers
- **Files**: Multiple container log files (`container_*.log`)
- **Log Format**: `<Date> <Time> <Level> [<Process>] <Component>: <Content>`
- **Anomaly Indicator**: `WARN` level messages indicate anomalies
- **Typical Entry**:
  ```
  2015-10-17 15:37:56,547 INFO [main] org.apache.hadoop.mapreduce.v2.app.MRAppMaster: Created MRAppMaster for application
  ```
- **Parser**: Drain
- **Use Case**: Training and validation

#### ZooKeeper
- **Source**: Apache ZooKeeper distributed coordination service
- **Description**: Consensus and coordination service logs
- **Files**: `Zookeeper.log`
- **Log Format**: `<Date> <Time> - <Level> [<Node>:<Component>@<Id>] - <Content>`
- **Anomaly Indicator**: `WARN` level messages indicate anomalies
- **Typical Entry**:
  ```
  2015-07-29 17:41:41,536 - INFO [main:QuorumPeerConfig@101] - Reading configuration from: /etc/zookeeper/conf/zoo.cfg
  ```
- **Parser**: Drain
- **Use Case**: Training and validation

### Type 2: Wordlist-Based Labeling
Datasets where anomalies are detected using keyword matching from curated wordlists.

#### Spark
- **Source**: Apache Spark distributed processing framework
- **Description**: Large-scale data processing and analytics logs
- **Files**: Multiple container log files (`container_*.log`)
- **Log Format**: `<Date> <Time> <Level> <Component>: <Content>`
- **Anomaly Indicator**: Keyword matching against `spark.txt` wordlist
- **Wordlist Examples**: error, exception, fail, timeout, lost
- **Parser**: Drain
- **Use Case**: Training, validation, and testing

#### Windows
- **Source**: Windows Component-Based Servicing (CBS)
- **Description**: Windows Update and servicing logs
- **Files**: `windows.log`
- **Log Format**: `<Date> <Time>, <Level> <Component> <Content>`
- **Anomaly Indicator**: Keyword matching against `windows.txt` wordlist
- **Typical Entry**:
  ```
  2016-09-28 04:30:30, Info CBS    Starting TrustedInstaller initialization.
  ```
- **Parser**: Drain
- **Use Case**: Generalization testing only (not used for training)

### Type 3: Label Column
Datasets with explicit label columns indicating normal/anomaly status.

#### BGL (Blue Gene/L)
- **Source**: Blue Gene/L supercomputer system
- **Description**: High-performance computing system logs
- **Files**: Large structured log files with embedded labels
- **Log Format**: `<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>`
- **Anomaly Indicator**: Label column where `-` indicates normal, other values indicate anomalies
- **Characteristics**: Pre-labeled dataset widely used in log analysis research
- **Parser**: Drain
- **Use Case**: Training, validation, and testing

### Type 4: NER-Based Parsing
Datasets processed using Named Entity Recognition parser for semi-structured logs.

#### Casper-RW
- **Source**: Casper-RW forensic challenge
- **Description**: Security forensics dataset with various system logs
- **Files**: Multiple log files (auth, daemon, syslog, etc.)
- **Log Types**: Authentication, daemon, system logs
- **Anomaly Indicator**: Token-level NER parsing identifies anomalous patterns
- **Parser**: NER (nerlogparser)
- **Use Case**: Training, validation, and testing

#### DFRWS 2009 JHU/ISI
- **Source**: Digital Forensic Research Workshop (DFRWS) 2009 Challenge
- **Description**: Forensic analysis dataset from Johns Hopkins/ISI
- **Files**: Multiple categorized log files (auth_*.log, daemon_*.log, debug_*.log)
- **Log Types**: auth, daemon, debug system logs
- **Anomaly Indicator**: Forensic events and intrusion patterns
- **Parser**: NER (nerlogparser)
- **Use Case**: Training, validation, and testing

#### DFRWS 2009 NSSAL
- **Source**: Digital Forensic Research Workshop (DFRWS) 2009 Challenge
- **Description**: Forensic analysis dataset from Naval Surface Warfare Center
- **Files**: Multiple categorized log files
- **Log Types**: Various syslog types
- **Anomaly Indicator**: Security incidents and anomalous activities
- **Parser**: NER (nerlogparser)
- **Use Case**: Training, validation, and testing

#### Honeynet Challenge 5
- **Source**: Honeynet Project Forensic Challenge 5
- **Description**: Security incident logs from honeypot systems
- **Files**: Security and system logs
- **Anomaly Indicator**: Intrusion attempts and malicious activities
- **Parser**: NER (nerlogparser)
- **Use Case**: Generalization testing only (not used for training)

#### Honeynet Challenge 7
- **Source**: Honeynet Project Forensic Challenge 7
- **Description**: Advanced persistent threat (APT) scenario logs
- **Files**: Security and system logs
- **Anomaly Indicator**: APT activities and security incidents
- **Parser**: NER (nerlogparser)
- **Use Case**: Training, validation, and testing

## Dataset Statistics

| Dataset | Type | Parser | Files | Approx. Size | Anomaly Rate | Split |
|---------|------|--------|-------|--------------|--------------|-------|
| Hadoop | Type 1 | Drain | Multiple containers | Large | Low-Medium | Train/Val/Test |
| Spark | Type 2 | Drain | Multiple containers | Large | Low-Medium | Train/Val/Test |
| ZooKeeper | Type 1 | Drain | Single file | Medium | Low | Train/Val/Test |
| BGL | Type 3 | Drain | Single file | Very Large | Low | Train/Val/Test |
| Windows | Type 2 | Drain | Single file | Large | Medium | Test only |
| Casper-RW | Type 4 | NER | Multiple files | Medium | Medium | Train/Val/Test |
| DFRWS JHU/ISI | Type 4 | NER | Multiple files | Medium | Medium-High | Train/Val/Test |
| DFRWS NSSAL | Type 4 | NER | Multiple files | Medium | Medium-High | Train/Val/Test |
| Honeynet Ch5 | Type 4 | NER | Multiple files | Small-Medium | High | Test only |
| Honeynet Ch7 | Type 4 | NER | Multiple files | Medium | Medium-High | Train/Val/Test |

## Log Format Specifications

### Hadoop/Spark Format
```
<Date> <Time> <Level> [<Process>] <Component>: <Content>
```
**Example**:
```
2015-10-17 15:37:56,547 INFO [main] org.apache.hadoop.mapreduce.v2.app.MRAppMaster: Created MRAppMaster
```

### ZooKeeper Format
```
<Date> <Time> - <Level> [<Node>:<Component>@<Id>] - <Content>
```
**Example**:
```
2015-07-29 17:41:41,536 - INFO [main:QuorumPeerConfig@101] - Reading configuration from: /etc/zookeeper/conf/zoo.cfg
```

### BGL Format
```
<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>
```
**Example**:
```
- 1131623764 2005.11.09 R02-M1-N0-C:J12-U11 2005-11-09-15.42.44.371441 R02-M1-N0-C:J12-U11 RAS KERNEL INFO generating core.17525
```

### Windows Format
```
<Date> <Time>, <Level> <Component> <Content>
```
**Example**:
```
2016-09-28 04:30:30, Info CBS    Starting TrustedInstaller initialization.
```

### NER-Based Logs
Semi-structured syslog formats processed by NER parser:
- Authentication logs (auth.log)
- Daemon logs (daemon.log)
- Debug logs (debug.log)
- System logs (syslog, messages, etc.)

## Usage

### Adding Raw Logs

Place raw log files in the appropriate dataset's `logs/` subdirectory:

```bash
# Example: Adding Hadoop logs
datasets/hadoop/logs/container_1445062781478_0011_01_000001.log
datasets/hadoop/logs/container_1445062781478_0011_01_000002.log
```

### Processing Datasets

Use the ground truth extraction pipeline to process raw logs:

```bash
# Process Hadoop dataset
python groundtruth/main.py --dataset hadoop --force

# Process Spark dataset
python groundtruth/main.py --dataset spark --force

# Process all datasets
for dataset in hadoop spark zookeeper bgl casper-rw; do
    python groundtruth/main.py --dataset $dataset
done
```

### Dataset Selection

Specify dataset in configuration or command-line arguments:

```python
from utils.process_arguments import Process_Arguments

# Parse arguments
args = Process_Arguments()
dataset_name = args.dataset  # e.g., 'hadoop', 'spark', etc.

# Load dataset
from train.utils.groundtruth_loader import GroundTruthLoader
train_loader = GroundTruthLoader(
    dataset=dataset_name,
    batch_size=32,
    split='train'
)
```

## Generated Outputs

After processing, each dataset will have additional subdirectories:

```
<dataset>/
├── logs/                           # Original raw logs
├── logs_structured/                # Parsed and structured logs (CSV/pickle)
├── log_embeddings/                 # Pre-computed message embeddings
└── groundtruth/                    # Ground truth labels and splits
    ├── messages.p                  # Tokenized messages
    ├── sequences.p                 # Context sequences
    ├── labels.p                    # Anomaly labels
    ├── keys.p                      # Message IDs
    ├── train_set.p                 # Training split
    ├── valid_set.p                 # Validation split
    ├── test_set.p                  # Test split
    └── resampled_groundtruth/      # Class-balanced datasets (optional)
```

## Data Processing Pipeline

1. **Raw Logs**: Original log files stored in `logs/`
2. **Parsing**: Drain or NER parser extracts structured fields
3. **Structured Storage**: Parsed logs saved to `logs_structured/`
4. **Embedding**: SentenceTransformer computes message embeddings
5. **Embedding Storage**: Embeddings saved to `log_embeddings/`
6. **Labeling**: Dataset-specific strategy assigns anomaly labels
7. **Sequence Building**: Context windows constructed around each message
8. **Ground Truth**: Labels and sequences saved to `groundtruth/`
9. **Splitting**: Data divided into train/validation/test sets
10. **Resampling** (optional): Class imbalance correction

## Dataset-Specific Configurations

Configurations are defined in `groundtruth/utils/settings.py`:

```python
# Hadoop configuration
'hadoop': {
    'in_dir': 'hadoop/logs/',
    'out_dir': 'hadoop/logs_structured/',
    'embs_dir': 'hadoop/log_embeddings/',
    'groundtruth_dir': 'hadoop/groundtruth/',
    'log_format': '<Date> <Time> <Level> \\[<Process>\\] <Component>: <Content>',
    'regex': [r'(\d+\.){3}\d+'],  # Mask IP addresses
    'st': 0.5,      # Similarity threshold for Drain
    'depth': 4      # Parse tree depth for Drain
}
```

### Configuration Parameters

- **in_dir**: Raw log files location
- **out_dir**: Structured output location
- **embs_dir**: Embeddings storage location
- **groundtruth_dir**: Ground truth output location
- **log_format**: Drain parser format string (Type 1-3 only)
- **regex**: Pattern list for masking dynamic content (Type 1-3 only)
- **st**: Similarity threshold for Drain clustering (Type 1-3 only)
- **depth**: Drain parse tree depth (Type 1-3 only)

## Adding New Datasets

To add a new dataset to CoLog:

### 1. Create Directory Structure

```bash
mkdir -p datasets/new_dataset/logs
```

### 2. Add Raw Logs

Place log files in `datasets/new_dataset/logs/`

### 3. Configure Settings

Add configuration to `groundtruth/utils/settings.py`:

```python
'new_dataset': {
    'in_dir': 'new_dataset/logs/',
    'out_dir': 'new_dataset/logs_structured/',
    'embs_dir': 'new_dataset/log_embeddings/',
    'groundtruth_dir': 'new_dataset/groundtruth/',
    'log_format': '<Date> <Time> <Level> <Content>',  # If using Drain
    'regex': [r'pattern1', r'pattern2'],  # If using Drain
    'st': 0.5,
    'depth': 4
}
```

### 4. Add to Dataset Lists

Update `groundtruth/utils/constants.py`:

```python
LOGS_LIST.append('new_dataset')

# Add to appropriate type list
LOGS_TYPE1.append('new_dataset')  # or TYPE2, TYPE3, TYPE4
LOGS_DRAIN.append('new_dataset')  # or LOGS_NER
```

### 5. Create Wordlist (Type 2 Only)

If using wordlist-based labeling, create `groundtruth/wordlists/new_dataset.txt`:

```
error
exception
fail
crash
timeout
```

### 6. Process Dataset

```bash
python groundtruth/main.py --dataset new_dataset --force
```

## Dataset Sources and References

### Open-Source Datasets

- **Hadoop/Spark/ZooKeeper**: Apache Software Foundation logs
- **BGL**: Lawrence Livermore National Laboratory
- **Windows**: Microsoft Windows system logs

### Forensic Challenge Datasets

- **DFRWS 2009**: Digital Forensic Research Workshop
  - Website: https://www.dfrws.org/
  - Challenges focused on digital forensics and incident response

- **Honeynet Project**: Security research organization
  - Website: https://www.honeynet.org/
  - Challenges featuring real-world attack scenarios

### Research Publications

These datasets have been used in numerous log analysis research papers:

1. **Drain Parser**
   - He et al. "Drain: An Online Log Parsing Approach with Fixed Depth Tree"
   - Used with: Hadoop, Spark, ZooKeeper, BGL, Windows

2. **Log Anomaly Detection**
   - Du et al. "DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning"
   - Used with: BGL, Hadoop

3. **Forensic Analysis**
   - Various DFRWS and Honeynet publications
   - Used with: DFRWS datasets, Honeynet challenges

## Best Practices

### Storage Management

1. **Keep raw logs**: Always preserve original log files
2. **Use version control**: Track changes to configurations
3. **Backup datasets**: Critical data should be backed up
4. **Document sources**: Maintain provenance information

### Processing Guidelines

1. **Verify log format**: Check format strings match actual logs
2. **Test on small samples**: Validate parsing before full processing
3. **Monitor disk space**: Embeddings and structured data require significant storage
4. **Use consistent seeds**: Ensure reproducible splits with `--random-seed`

### Quality Assurance

1. **Inspect parsed output**: Verify structured logs are correctly parsed
2. **Check label distribution**: Ensure balanced or expected class distribution
3. **Validate sequences**: Confirm context windows are properly constructed
4. **Review embeddings**: Spot-check embedding quality and dimensions

## Troubleshooting

### Issue: Parsing Fails

**Symptoms**: Empty or incomplete structured logs

**Solutions**:
- Verify log format string matches actual log structure
- Check regex patterns are valid
- Inspect raw logs for unexpected format variations
- Enable verbose logging: `--verbose`

### Issue: Out of Disk Space

**Symptoms**: Processing stops during embedding computation

**Solutions**:
- Clear temporary files and caches
- Use batch processing for large datasets
- Store embeddings on separate disk
- Process datasets incrementally

### Issue: Incorrect Labels

**Symptoms**: Unexpected anomaly/normal ratios

**Solutions**:
- Verify labeling strategy (WARN, wordlist, label column)
- Check wordlist file exists and is correct (Type 2)
- Inspect structured logs for label column (Type 3)
- Review dataset-specific extraction logic

### Issue: Memory Errors

**Symptoms**: Out of memory during processing

**Solutions**:
- Reduce batch size: `--batch-size 32`
- Process datasets sequentially instead of in parallel
- Use CPU instead of GPU if memory-constrained: `--device cpu`
- Split large log files into smaller chunks

## Privacy and Ethics

### Sensitive Information

Some datasets may contain:
- IP addresses
- Usernames
- File paths
- System configurations

**Recommendations**:
- Apply privacy masking using regex patterns
- Anonymize sensitive fields before sharing
- Follow institutional data protection policies
- Respect dataset licenses and usage terms

### Research Ethics

- **Cite original sources**: Give credit to dataset creators
- **Follow licenses**: Respect usage restrictions
- **Share responsibly**: Do not redistribute without permission
- **Document methodology**: Maintain reproducibility

## Performance Characteristics

### Processing Time Estimates

| Dataset | Parsing | Embedding | Total | Hardware |
|---------|---------|-----------|-------|----------|
| Hadoop | 5-10 min | 15-30 min | 20-40 min | GPU |
| Spark | 10-20 min | 30-60 min | 40-80 min | GPU |
| ZooKeeper | 2-5 min | 5-10 min | 7-15 min | GPU |
| BGL | 20-40 min | 60-120 min | 80-160 min | GPU |
| Windows | 10-15 min | 20-40 min | 30-55 min | GPU |
| Casper-RW | 5-10 min | 10-20 min | 15-30 min | GPU |

*Note: Times vary based on hardware, batch size, and model selection*

### Storage Requirements

| Component | Typical Size | Notes |
|-----------|-------------|-------|
| Raw logs | 100 MB - 10 GB | Original log files |
| Structured logs | 50 MB - 5 GB | Parsed CSV/pickle files |
| Embeddings | 200 MB - 20 GB | Pre-computed vectors |
| Ground truth | 50 MB - 2 GB | Labels and splits |
| **Total per dataset** | **400 MB - 37 GB** | Varies by dataset size |

## Related Modules

- **`groundtruth/`**: Processes datasets to generate training data
- **`train/`**: Uses datasets for model training and evaluation
- **`neuralnetwork/`**: Model architectures trained on datasets
- **`utils/`**: Shared utilities for dataset handling

## License

Dataset licenses vary by source. Consult original dataset documentation for licensing information. The CoLog processing code is part of the CoLog project.

## Authors

Dataset collection and integration by the CoLog research team. Original datasets created by their respective organizations and research communities.

## Acknowledgments

We thank the following organizations and communities for making these datasets publicly available:
- Apache Software Foundation
- Lawrence Livermore National Laboratory
- Digital Forensic Research Workshop (DFRWS)
- Honeynet Project
- Microsoft Corporation

## Version

Dataset collection version: 1.0.0

Last updated: December 2025
