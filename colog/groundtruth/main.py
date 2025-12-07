"""
main.py — prepare dataset ground truth for CoLog

This script extracts structured ground truth for different system log
collections used by CoLog. The script includes integrated preprocessing
functionality to parse logs and compute embeddings when needed. It expects
parsed logs as input (either CSV files produced by Drain or pickled 
structured files from the NER parser), constructs sequences for each log 
message using a configurable window size, assigns labels (normal/anomaly) 
based on a dataset-specific strategy, and writes per-dataset ground truth 
files for training/validation/testing.

Key concepts:
    - sequence_type: 'background' uses the current and previous messages as
        context; 'context' additionally includes following messages within the
        window (left + current + right context).
    - window_size: number of messages on each side (left/right) used to build
        sequences. For example, window_size=1 yields sequences like:
            background: [prev_msg, current_msg]
            context: [prev_msg, current_msg, next_msg]
    - Integrated preprocessing: The Preprocessor class handles log parsing 
        (Drain/NER) and embedding computation using SentenceTransformer.
    - Labeling: Some datasets use structured fields (e.g., Level/WARN); others
        use a wordlist heuristic; NER datasets rely on token-level parsing.
    - Class imbalance: ClassImbalanceSolver can resample training data using
        various methods (TomekLinks, SMOTE, etc.) to handle imbalanced classes.
    - Reproducibility: Use --random-seed to control dataset splitting and resampling.
    - Dry run mode: Use --dry-run to preview operations without modifying files.

Usage examples:
    python main.py --dataset hadoop --sequence-type context --window-size 1 \
        --model all-MiniLM-L6-v2 --batch-size 64 --device auto
    python main.py --dataset casper-rw --sequence-type background \
        --model all-MiniLM-L6-v2 --batch-size 32 --device cuda --force --verbose
    python main.py --dataset bgl --resample --resample-method TomekLinks \
        --random-seed 42
    python main.py --dataset spark --dataset-dir datasets/ --dry-run
    python main.py --dataset zookeeper --groundbreaking --train-ratio 0.7 \
        --valid-ratio 0.15

Files produced:
    - messages.p (id -> tokenized message as NumPy array)
    - sequences.p (id -> list of messages around the id; 'UNK' for padding)
    - labels.p (id -> label integer)
    - keys.p (ordered list of ids)
    - train_set/valid_set/test_set.p (split datasets for models) or generalization set
    - Resampled datasets in resampled_groundtruth/<method>/ subdirectories

Dependencies: pandas, numpy, tqdm, sentence-transformers, imblearn, torch
"""

from pathlib import Path
from datetime import datetime
import logging
import sys
from typing import List, Dict
import warnings

from utils import Settings
from utils import parse_arguments
from preprocessing import Preprocessor
from extraction import GroundTruth
from sampling import ClassImbalanceSolver
from utils import constants

# Ignore all warnings (comment out if you want to see warnings during
# debugging or development)
warnings.filterwarnings("ignore", category=UserWarning)

# module-level logger: matches preprocess.py behavior; we prefer structured logs
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Prevent 'No handler could be found' warnings if imported before logging configured
    logger.addHandler(logging.NullHandler())


if __name__ == '__main__':
    start_time = datetime.now()

    # Given args
    args = parse_arguments()

    # Dataset mapping used to validate and dispatch parsing pipelines
    logs_dict: Dict[str, List[str]] = {
        'logs_list': constants.LOGS_LIST,
        'logs_drain': constants.LOGS_DRAIN,
        'logs_ner': constants.LOGS_NER,
    }

    # Validate dataset name. While `argparse` choices will already prevent
    # invalid values, this explicit check provides a clearer error message
    # and is consistent with `preprocess.py`.
    if args.dataset not in logs_dict['logs_list']:
        print(f"Please select a valid dataset. Available choices: {', '.join(logs_dict['logs_list'])}")
        sys.exit(1)

    if args.sequence_type not in [constants.SEQ_TYPE_BACKGROUND, constants.SEQ_TYPE_CONTEXT]:
        print('Please select a valid sequence type. [background or context]')
        sys.exit(1)
    
    else:
        # Configure logging; make sure log dir exists
        log_path = Path(constants.DIR_LOGS) / constants.FILE_COLOG_LOG
        log_dir = log_path.parent
        if log_dir and not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
        # Configure logging
        log_level = logging.DEBUG if args.verbose else logging.INFO
        logging.basicConfig(
            filename=str(Path(constants.DIR_LOGS) / constants.FILE_COLOG_LOG),
            level=log_level,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
        logger.info('Groundtruth run start dataset=%s sequence_type=%s window=%d force=%s groundbreaking=%s',
                    args.dataset, args.sequence_type, args.window_size, args.force, args.groundbreaking)

        # Run preprocess.py automatically before extracting ground truth
        # Skip preprocessing for 'all' dataset as it combines already-processed datasets
        if args.dataset != 'all':
            preprocessor = Preprocessor(args.dataset, args.dataset_dir, args.model, args.batch_size, 
                                       args.device, args.force, args.verbose, args.dry_run)
            preprocess_success = preprocessor.run(args)
            if not preprocess_success:
                logger.error('[%s] Preprocessing failed, cannot continue with ground truth extraction', args.dataset)
                print(f"\n[X] Aborting ground truth extraction for '{args.dataset}' due to preprocessing failure.")
                sys.exit(1)
            if not args.dry_run:
                print(f"[OK] Preprocessing completed\n")
            else:
                print(f"[DRY RUN] Preprocessing preview completed\n")
        else:
            print(f"[all] Start aggregating existing datasets)\n")

        # Create GroundTruth helper with CLI options including configurable random seed
        if args.dataset != 'all':
            print(f"[STEP 2/2] Extracting ground truth...")
        else:
            print(f"[STEP 1/2] Aggregating ground truth from existing datasets...")
        gt = GroundTruth(dataset=args.dataset, sequence_type=args.sequence_type,
                         window_size=args.window_size, train_ratio=args.train_ratio,
                         valid_ratio=args.valid_ratio, sampling_method=args.resample_method, datasets_dir=args.dataset_dir,
                         force=args.force, groundbreaking=args.groundbreaking,
                         random_seed=args.random_seed, dry_run=args.dry_run)

        # Execute ground truth extraction using facade pattern
        # The GroundTruth facade dispatches to specialized extractors:
        # - Type1GroundTruthExtractor (hadoop, zookeeper): Level column-based labeling
        # - Type2GroundTruthExtractor (spark, windows): wordlist heuristic
        # - Type3GroundTruthExtractor (bgl): label '-' indicates normal
        # - Type4GroundTruthExtractor (casper-rw, dfrws*, honeynet*): NER parser stored pickle
        # - GroundTruthAggregator (all): aggregates pre-extracted groundtruths
        if gt.extract_groundtruth():
            print(f"\n{'='*70}")
            if args.dry_run:
                print(f"[OK] Ground truth extraction preview completed!")
                print(f"  (No files were modified - this was a dry run)")
            else:
                if args.dataset != 'all':
                    print(f"[OK] Ground truth extraction completed successfully!")
                else:
                    print(f"[OK] Ground truth aggregation completed successfully!")
            print(f"  Time taken: {datetime.now() - start_time}")
            print(f"{'='*70}\n")
        else:
            sys.exit(1)

        # Optionally run resampling after groundtruth extraction. The
        # resampling step uses the same `--device` and `--verbose` flags
        # supplied to the groundtruth script to avoid duplicate arguments.
        if (args.resample and args.dataset != 'all'):
            # Skip resampling for windows and honeynet-challenge5 datasets
            if args.dataset in constants.DATASETS_NO_RESAMPLING:
                logger.info('[%s] This dataset is not used for training CoLog. Resampling ignored.', args.dataset)
                print(f"[{args.dataset}] This dataset is not used for training CoLog. Resampling ignored.")
                sys.exit(1)
            elif not args.resample_method:
                print('Resample requested but --resample-method not provided; skipping resampling.')
                logger.warning('Resample requested without method; skipping')
            else:
                print(f"\n[OPTIONAL] Resampling with method '{args.resample_method}'...")
                try:
                    sampler = ClassImbalanceSolver(dataset=args.dataset,
                                                   method=args.resample_method,
                                                   groundtruth_dir=str(Path(args.dataset_dir) / Settings(args.dataset).get_settings()['groundtruth_dir'] / f"{args.sequence_type.lower()}_{args.window_size}"),
                                                   train_ratio=args.train_ratio,
                                                   device=args.device,
                                                   batch_size=args.batch_size,
                                                   verbose=args.verbose,
                                                   force=args.force,
                                                   random_seed=args.random_seed,
                                                   dry_run=args.dry_run)
                    sampler.sample_groundtruth()
                    if args.dry_run:
                        print(f"[DRY RUN] Resampling preview completed\n")
                    else:
                        print(f"[OK] Resampling completed\n")
                except Exception:
                    logger.exception('[%s] Resampling failed method=%s', args.dataset, args.resample_method)
                    print(f"[X] Resampling failed. See logs for details.\n")
                    sys.exit(1)

        logger.info('Groundtruth run finished dataset=%s duration=%s', args.dataset, datetime.now() - start_time)
        
        # Print final summary
        print(f"{'='*70}")
        if args.dry_run:
            print(f"[OK] All preview tasks completed!")
            print(f"  (No files were modified - this was a dry run)")
        else:
            print(f"[OK] All tasks completed successfully!")
        print(f"  Total time: {datetime.now() - start_time}")
        print(f"{'='*70}")
