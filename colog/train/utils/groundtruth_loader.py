"""
groundtruth_loader.py — PyTorch Dataset loader for CoLog ground truth data

This module provides the GroundTruthLoader class, a PyTorch Dataset implementation
for loading preprocessed ground truth data for log anomaly detection. It handles
textual log message embeddings, sequential features, and binary anomaly labels.

Key functionality:
    - Message preprocessing: Tokenization, stopword removal, and normalization
    - Token-to-index mapping: Creation and caching of vocabulary dictionaries
    - Word embeddings: spaCy-based vector extraction for semantic features
    - Sequence padding: Fixed-length sequence handling for batch processing
    - Multi-modal data loading: Combined text and sequence feature retrieval

The loader supports train/validation/test splits with efficient caching of
embeddings and token mappings to avoid redundant computation across epochs.

Usage
-----
    from train.utils.groundtruth_loader import GroundTruthLoader
    
    # Create training dataset
    train_loader = GroundTruthLoader(
        data_split='train_set',
        args=config,
        dataset_dir='datasets/',
        print_hints=True
    )
    
    # Use with PyTorch DataLoader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(train_loader, batch_size=32, shuffle=True)

Notes
-----
- Token embeddings are cached to disk after first extraction
- Preprocessed messages filter stopwords and non-alphabetic characters
- Sequence features are padded/truncated to fixed length for batching
- Labels are binary: 0 (anomaly) or 1 (normal)

See Also
--------
torch.utils.data.Dataset : Base class for custom datasets
"""

from __future__ import print_function
import os
import pickle
import re
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset
import en_core_web_lg
from nltk import corpus

from groundtruth.utils import DatasetAttributes
from train.utils import constants

# Create module alias for pickle compatibility
# Pickle files reference 'utils.dataclasses' but the actual module is 'groundtruth.utils.dataclasses'
import groundtruth.utils.dataclasses as utils_dataclasses
sys.modules['utils.dataclasses'] = utils_dataclasses

# Ignore all warnings (comment out during debugging if needed)
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# ============================================================================
# Stopwords Initialization
# ============================================================================

try:
    STOPWORDS = corpus.stopwords.words(constants.NLTK_LANGUAGE)
except LookupError:
    import nltk
    logger.warning("NLTK stopwords not found. Downloading...")
    nltk.download(constants.NLTK_STOPWORDS_PACKAGE, quiet=True)
    STOPWORDS = corpus.stopwords.words(constants.NLTK_LANGUAGE)
    logger.info("NLTK stopwords downloaded successfully")


# ============================================================================
# Text Preprocessing Functions
# ============================================================================

def preprocess_word(word: str) -> str:
    """Preprocess a single word by normalizing and filtering.
    
    This function performs the following transformations:
    1. Converts to lowercase
    2. Removes non-alphabetic characters
    3. Filters out single-character words
    4. Removes English stopwords
    
    Parameters
    ----------
    word : str
        The word to preprocess.
    
    Returns
    -------
    str
        Preprocessed word or empty string if filtered out.
    
    Examples
    --------
    >>> preprocess_word("Hello123")
    'hello'
    >>> preprocess_word("the")
    ''
    >>> preprocess_word("a")
    ''
    
    Notes
    -----
    Empty strings are returned for words that should be filtered from
    the vocabulary (stopwords, single characters, non-alphabetic tokens).
    """
    try:
        # Convert to lowercase and remove non-alphabetic characters
        word_lower = word.lower()
        alphabets_only = re.sub(r'[^a-zA-Z]', '', word_lower)
        
        # Filter single-character words and stopwords
        if len(alphabets_only) <= 1 or alphabets_only in STOPWORDS:
            return ''
        
        return alphabets_only
    except Exception as e:
        logger.warning(f"Error preprocessing word '{word}': {e}")
        return ''

def preprocess_messages(messages: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Preprocess all messages by applying word preprocessing.
    
    Applies `preprocess_word` to each word in each message, filtering out
    empty results (stopwords, single characters, non-alphabetic tokens).
    
    Parameters
    ----------
    messages : Dict[str, List[str]]
        Dictionary mapping message keys to lists of raw words.
    
    Returns
    -------
    Dict[str, List[str]]
        Dictionary with preprocessed messages containing only valid tokens.
    
    Examples
    --------
    >>> messages = {'msg1': ['Hello', 'the', 'world', '123']}
    >>> preprocess_messages(messages)
    {'msg1': ['hello', 'world']}
    
    Notes
    -----
    This function is applied to all messages in the dataset to create a
    clean vocabulary for embedding extraction.
    """
    try:
        preprocessed_messages = {}
        for key, word_list in messages.items():
            # Preprocess and filter words in a single pass
            preprocessed_words = [
                preprocessed_word 
                for word in word_list 
                if (preprocessed_word := preprocess_word(word))
            ]
            preprocessed_messages[key] = preprocessed_words
        return preprocessed_messages
    except Exception as e:
        logger.error(f"Error preprocessing messages: {e}")
        raise

# ============================================================================
# Embedding and Vocabulary Functions
# ============================================================================

def create_embeddings_dict(
    preprocessed_messages: Dict[str, List[str]], 
    data_path: str, 
    extract_embeddings: bool = True, 
    print_hints: bool = True
) -> Tuple[Dict[str, int], np.ndarray]:
    """Create or load token-to-index mapping and corresponding embeddings.
    
    This function builds a vocabulary from preprocessed messages and extracts
    word embeddings using spaCy's pre-trained language model. Results are
    cached to disk to avoid redundant computation.
    
    Parameters
    ----------
    preprocessed_messages : Dict[str, List[str]]
        Dictionary of preprocessed messages with filtered tokens.
    data_path : str
        Directory path to save/load tokens and embeddings.
    extract_embeddings : bool, optional
        Whether to extract embeddings using spaCy. Default is True.
    print_hints : bool, optional
        Whether to print progress messages. Default is True.
    
    Returns
    -------
    Tuple[Dict[str, int], np.ndarray]
        tokens_dict : Dictionary mapping words to integer indices
        embeddings_array : NumPy array of word vectors (vocab_size, embedding_dim)
    
    Notes
    -----
    - The 'UNK' token is always added at index 0 for unknown words
    - Embeddings are extracted using spaCy's en_core_web_lg model (300d)
    - Files are saved as 'tokens.pkl' and 'train_embs.npy'
    
    Examples
    --------
    >>> messages = {'msg1': ['hello', 'world']}
    >>> tokens, embs = create_embeddings_dict(messages, './data')
    >>> tokens
    {'UNK': 0, 'hello': 1, 'world': 2}
    >>> embs.shape
    (3, 300)
    """
    # Define cache file paths
    data_path_obj = Path(data_path)
    data_path_obj.mkdir(parents=True, exist_ok=True)
    tokens_file = data_path_obj / constants.FILE_TOKENS
    embeddings_file = data_path_obj / constants.FILE_TRAIN_EMBEDDINGS
    
    # Load cached embeddings if available
    if tokens_file.exists() and embeddings_file.exists():
        if print_hints:
            print("Loading cached embeddings dictionary\n")
            logger.info(f"Loading cached embeddings from {data_path}")
        try:
            with open(tokens_file, "rb") as f:
                tokens_dict = pickle.load(f)
            embeddings = np.load(embeddings_file)
            logger.info(f"Loaded {len(tokens_dict)} tokens with embeddings of shape {embeddings.shape}")
            return tokens_dict, embeddings
        except Exception as e:
            logger.warning(f"Failed to load cached embeddings: {e}. Recreating...")
    
    # Create new embeddings dictionary
    if print_hints:
        print("Creating embeddings dictionary from scratch\n")
        logger.info("Building vocabulary and extracting embeddings")
    
    # Initialize with UNK token
    tokens_dict = {constants.UNKNOWN_TOKEN: 0}
    embeddings = []
    spacy_tool = None
    
    # Load spaCy model if embeddings extraction is enabled
    if extract_embeddings:
        try:
            spacy_tool = en_core_web_lg.load()
            embeddings.append(spacy_tool(constants.UNKNOWN_TOKEN).vector)
            logger.info("spaCy model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            raise RuntimeError(f"Could not load spaCy model 'en_core_web_lg': {e}")
    
    # Build vocabulary and extract embeddings
    try:
        for message_key, word_list in preprocessed_messages.items():
            for word in word_list:
                if word not in tokens_dict:
                    tokens_dict[word] = len(tokens_dict)
                    if extract_embeddings:
                        embeddings.append(spacy_tool(word).vector)
        
        # Convert to numpy array
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Save to disk
        np.save(embeddings_file, embeddings)
        with open(tokens_file, "wb") as f:
            pickle.dump(tokens_dict, f)
        
        if print_hints:
            print(f"Created vocabulary with {len(tokens_dict)} tokens\n")
            logger.info(f"Saved {len(tokens_dict)} tokens and embeddings of shape {embeddings.shape}")
        
        return tokens_dict, embeddings
        
    except Exception as e:
        logger.error(f"Error creating embeddings dictionary: {e}")
        raise

def create_message_indices(
    preprocessed_message: List[str], 
    tokens_dict: Dict[str, int], 
    max_tokens: int = 80
) -> np.ndarray:
    """Convert preprocessed message words to token indices.
    
    Maps each word in the message to its corresponding integer index using
    the provided vocabulary dictionary. Unknown words are mapped to the
    'UNK' token index. Messages are padded or truncated to a fixed length.
    
    Parameters
    ----------
    preprocessed_message : List[str]
        List of preprocessed words from a log message.
    tokens_dict : Dict[str, int]
        Dictionary mapping words to token indices.
    max_tokens : int, optional
        Maximum number of tokens to include. Default is 80.
    
    Returns
    -------
    np.ndarray
        Integer array of shape (max_tokens,) containing token indices.
    
    Examples
    --------
    >>> tokens = {'UNK': 0, 'hello': 1, 'world': 2}
    >>> message = ['hello', 'world', 'unknown']
    >>> create_message_indices(message, tokens, max_tokens=5)
    array([1, 2, 0, 0, 0])
    
    Notes
    -----
    - Messages longer than max_tokens are truncated
    - Messages shorter than max_tokens are zero-padded
    - Unknown words are mapped to tokens_dict['UNK']
    """
    try:
        message_indices = np.zeros(max_tokens, dtype=np.int64)
        unk_index = tokens_dict.get(constants.UNKNOWN_TOKEN, 0)
        
        for index, word in enumerate(preprocessed_message):
            if index >= max_tokens:
                break
            message_indices[index] = tokens_dict.get(word, unk_index)
        
        return message_indices
        
    except Exception as e:
        logger.error(f"Error creating message indices: {e}")
        raise

# ============================================================================
# Sequence Processing Functions
# ============================================================================

def pad_sequence(sequence: np.ndarray, max_length: int) -> np.ndarray:
    """Pad or truncate a sequence to a fixed length.
    
    This function ensures sequences have uniform length for batch processing.
    Sequences longer than max_length are truncated from the end, while
    shorter sequences are zero-padded.
    
    Parameters
    ----------
    sequence : np.ndarray
        The sequence array to pad or truncate with shape (seq_len, feature_dim).
    max_length : int
        The target sequence length.
    
    Returns
    -------
    np.ndarray
        Padded or truncated sequence with shape (max_length, feature_dim).
    
    Examples
    --------
    >>> seq = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> pad_sequence(seq, max_length=3)
    array([[1, 2],
           [3, 4],
           [5, 6]])
    
    >>> pad_sequence(seq, max_length=6)
    array([[1, 2],
           [3, 4],
           [5, 6],
           [7, 8],
           [0, 0],
           [0, 0]])
    
    Notes
    -----
    - Padding is applied at the end of the sequence
    - Zero-padding is used for consistency with PyTorch conventions
    - Sequences must be 2D arrays (seq_len, feature_dim)
    """
    try:
        if sequence.ndim != 2:
            raise ValueError(f"Expected 2D sequence array, got {sequence.ndim}D")
        
        # Truncate if longer than max_length
        if sequence.shape[0] > max_length:
            sequence = sequence[:max_length]
        
        # Pad if shorter than max_length
        if sequence.shape[0] < max_length:
            padding = ((0, max_length - sequence.shape[0]), (0, 0))
            sequence = np.pad(
                sequence,
                padding,
                mode='constant',
                constant_values=0
            )
        
        return sequence
        
    except Exception as e:
        logger.error(f"Error padding sequence: {e}")
        raise

# ============================================================================
# Label Processing Functions
# ============================================================================

def get_label(label_value: int, max_classes: int = 2) -> int:
    """Convert label value to classification label.
    
    This function validates and normalizes label values for anomaly detection.
    Supports both binary (2 classes) and multi-class (4 classes for groundbreaking mode).
    
    Parameters
    ----------
    label_value : int
        The raw label value from ground truth data.
    max_classes : int, optional
        Maximum number of classes. Default is 2 for binary classification.
        Set to 4 for multi-class groundbreaking mode.
    
    Returns
    -------
    int
        The validated label value.
    
    Raises
    ------
    ValueError
        If label_value is not in valid range [0, max_classes).
    
    Examples
    --------
    >>> get_label(0)
    0
    >>> get_label(1)
    1
    >>> get_label(2, max_classes=4)
    2
    
    Notes
    -----
    - For binary classification (max_classes=2):
        - Label 0 represents anomalous log sequences
        - Label 1 represents normal log sequences
    - For multi-class classification (max_classes=4, groundbreaking mode):
        - Label 0: message=anomaly, sequence=anomaly
        - Label 1: message=anomaly, sequence=normal
        - Label 2: message=normal, sequence=anomaly
        - Label 3: message=normal, sequence=normal
    """
    if label_value not in range(max_classes):
        logger.warning(f"Invalid label value: {label_value}. Expected 0 to {max_classes-1}.")
        raise ValueError(f"Label must be 0 to {max_classes-1}, got {label_value}")
    return label_value

# ============================================================================
# PyTorch Dataset Class
# ============================================================================

class GroundTruthLoader(Dataset):
    """PyTorch Dataset for loading ground truth data for log anomaly detection.
    
    This class loads preprocessed ground truth data including textual log messages,
    sequential embeddings, and binary anomaly labels for training, validation, or
    testing. It handles vocabulary creation, word embedding extraction, and
    efficient data loading with caching.
    
    The dataset supports multiple log types and provides both message-level
    and sequence-level features for multi-modal anomaly detection models.
    
    Parameters
    ----------
    data_split : str
        The data split to load. Must be one of:
        - 'train_set': Training data
        - 'valid_set': Validation data  
        - 'test_set': Test data
    args : object
        Arguments object containing dataset configuration. Expected attributes:
        - dataset: Tuple[str, str] - (dataset_name, configuration)
        - train_ratio: float - Training set ratio
        - method: str - Sampling method name
        - len_messages: int - Maximum message token length
        - len_sequences: int - Maximum sequence length
    tokens_dict : Dict[str, int], optional
        Pre-existing token-to-index mapping. If None, will be created from
        training data. Default is None.
    dataset_dir : str, optional
        Root directory for datasets. Default is 'datasets/'.
    print_hints : bool, optional
        Whether to print progress messages during initialization. Default is True.
    
    Attributes
    ----------
    data_split : str
        The data split being used (train/valid/test).
    keys : List[str]
        List of sample keys for this split.
    messages : Dict[str, List[str]]
        Raw tokenized messages indexed by key.
    sequences : Dict[str, np.ndarray]
        Pre-computed sequence embeddings indexed by key.
    labels : Dict[str, int]
        Labels indexed by key. For binary: 0=anomaly, 1=normal.
        For multi-class groundbreaking: 0-3 representing combinations of message/sequence labels.
    num_classes : int
        Number of classification classes (2 for binary, 4 for multi-class groundbreaking).
    preprocessed_messages : Dict[str, List[str]]
        Cleaned and filtered messages.
    tokens_dict : Dict[str, int]
        Vocabulary mapping words to integer indices.
    vocab_size : int
        Total vocabulary size.
    max_len_messages : int
        Maximum number of tokens per message.
    max_len_sequences : int
        Maximum sequence length for padding.
    
    Examples
    --------
    >>> # Create training dataset
    >>> train_dataset = GroundTruthLoader(
    ...     data_split='train_set',
    ...     args=config,
    ...     dataset_dir='datasets/'
    ... )
    >>> print(f\"Dataset size: {len(train_dataset)}\")
    >>> print(f\"Vocabulary size: {train_dataset.vocab_size}\")
    >>> 
    >>> # Get a single sample
    >>> key, message, sequence, label = train_dataset[0]
    >>> print(f\"Message shape: {message.shape}\")  # (max_len_messages,)
    >>> print(f\"Sequence shape: {sequence.shape}\") # (max_len_sequences, embedding_dim)
    >>> print(f\"Label: {label.item()}\")
    
    Notes
    -----
    - For training splits, embeddings and tokens are created and cached
    - For validation/test splits, tokens_dict should be passed from training
    - Sequence embeddings are pre-computed and loaded from pickle files
    - Zero-padding is applied to ensure uniform batch sizes
    - The dataset automatically filters out missing keys across modalities
    
    See Also
    --------
    torch.utils.data.Dataset : Base class for PyTorch datasets
    torch.utils.data.DataLoader : Efficient batching and loading
    """
    
    def __init__(
        self, 
        data_split: str, 
        args, 
        tokens_dict: Optional[Dict[str, int]] = None, 
        dataset_dir: str = constants.DEFAULT_DATASET_DIR, 
        print_hints: bool = True
    ) -> None:
        """Initialize GroundTruthLoader with validation and data loading."""
        super(GroundTruthLoader, self).__init__()
        
        # Validate data_split
        valid_splits = constants.VALID_SPLITS
        if data_split not in valid_splits:
            raise ValueError(f"data_split must be one of {valid_splits}, got '{data_split}'\")")
        
        # Store configuration
        self.data_split = data_split
        self.args = args
        self.dataset_dir = Path(dataset_dir)
        self.print_hints = print_hints
        
        # Determine data paths based on dataset type
        # Handle both tuple format (test mode) and separate args format (train mode)
        if isinstance(self.args.dataset, (list, tuple)) and len(self.args.dataset) >= 2:
            # Test mode: dataset is ['name', 'context_1']
            dataset_name = self.args.dataset[0]
            dataset_config = self.args.dataset[1]
        else:
            # Train mode: dataset is a string, sequence_type and window_size are separate
            dataset_name = self.args.dataset
            dataset_config = f"{self.args.sequence_type}_{self.args.window_size}"
        
        if dataset_name not in constants.UNSEEN_LOGS:
            # Datasets with train/valid/test splits
            self.data_path = self.dataset_dir / dataset_name / constants.DIR_GROUNDTRUTH / dataset_config / f'{constants.DIR_TRAIN_VALID_TEST_PREFIX}{self.args.train_ratio}'
        else:
            # Datasets with only train/test splits
            self.data_path = self.dataset_dir / dataset_name / constants.DIR_GROUNDTRUTH / dataset_config
        
        logger.info(f"Initializing GroundTruthLoader for {dataset_name} - {data_split}")

        
        # Load ground truth data based on dataset type and split
        try:
            if dataset_name not in constants.UNSEEN_LOGS:
                # Datasets with separate train/valid/test files
                # Check if resample_method is specified and not None
                if hasattr(self.args, 'resample_method') and self.args.resample_method:
                    # Use resampled data
                    resampled_path = self.data_path / constants.DIR_RESAMPLED / self.args.resample_method.lower()
                    
                    if data_split == constants.SPLIT_TRAIN:
                        groundtruth_file = resampled_path / f'{dataset_name}{constants.SUFFIX_TRAIN_SET}'
                    elif data_split == constants.SPLIT_VALID:
                        groundtruth_file = resampled_path / f'{dataset_name}{constants.SUFFIX_VALID_SET}'
                    else:  # test_set
                        groundtruth_file = self.data_path / f'{dataset_name}{constants.SUFFIX_TEST_SET}'
                else:
                    # Use original (non-resampled) data
                    if data_split == constants.SPLIT_TRAIN:
                        groundtruth_file = self.data_path / f'{dataset_name}{constants.SUFFIX_TRAIN_SET}'
                    elif data_split == constants.SPLIT_VALID:
                        groundtruth_file = self.data_path / f'{dataset_name}{constants.SUFFIX_VALID_SET}'
                    else:  # test_set
                        groundtruth_file = self.data_path / f'{dataset_name}{constants.SUFFIX_TEST_SET}'
            else:
                # Datasets with combined train+test file
                groundtruth_file = self.data_path / f'{dataset_name}{constants.EXT_PICKLE}'
            
            # Validate file exists
            if not groundtruth_file.exists():
                error_msg = f"Ground truth file not found: {groundtruth_file}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            # Load ground truth data
            if self.print_hints:
                print(f"Loading ground truth from {groundtruth_file.name}")
            
            with open(groundtruth_file, 'rb') as handle:
                groundtruth_data = pickle.load(handle)
            
            logger.info(f"Loaded ground truth data from {groundtruth_file}")
            
        except FileNotFoundError as e:
            logger.error(f"Ground truth file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading ground truth data: {e}")
            raise RuntimeError(f"Failed to load ground truth: {e}")

        
        # Load pre-computed log message embeddings
        try:
            embeddings_path = self.dataset_dir / dataset_name / constants.DIR_LOG_EMBEDDINGS / f'{dataset_name}{constants.SUFFIX_EMBEDDINGS}'
            
            if not embeddings_path.exists():
                error_msg = f"Embeddings file not found: {embeddings_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            with open(embeddings_path, 'rb') as handle:
                embeddings_dict = pickle.load(handle)
            
            logger.info(f"Loaded {len(embeddings_dict)} log message embeddings")
            
        except FileNotFoundError as e:
            logger.error(f"Embeddings file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise RuntimeError(f"Failed to load embeddings: {e}")

        
        # Extract data components from loaded ground truth
        try:
            # Get keys for this split
            self.keys = groundtruth_data.keys if hasattr(groundtruth_data, constants.ATTR_KEYS) else []
            if not self.keys:
                logger.warning(f"No keys found in {data_split}")
            
            # Extract messages (tokenized log content)
            self.messages = groundtruth_data.messages if hasattr(groundtruth_data, constants.ATTR_MESSAGES) else {}
            
            # Extract and process sequences
            sequences = groundtruth_data.sequences if hasattr(groundtruth_data, constants.ATTR_SEQUENCES) else {}
            processed_sequences = {}
            
            for key, sequence_list in sequences.items():
                sentence_embeddings = []
                for sentence in sequence_list:
                    # Map sentence IDs to their embeddings
                    sentence_str = str(sentence) if sentence != '' else constants.NAN_TOKEN
                    embedding = embeddings_dict.get(sentence_str)
                    
                    if embedding is not None:
                        sentence_embeddings.append(embedding)
                    else:
                        # Use NaN embedding as fallback
                        nan_embedding = embeddings_dict.get(constants.NAN_TOKEN)
                        if nan_embedding is not None:
                            sentence_embeddings.append(nan_embedding)
                        else:
                            logger.warning(f"Missing embedding for sentence '{sentence_str}' in key '{key}'")
                
                # Convert to numpy array
                if sentence_embeddings:
                    processed_sequences[key] = np.array(sentence_embeddings, dtype=np.float32)
                else:
                    logger.warning(f"No embeddings found for sequence key '{key}'")
            
            self.sequences = processed_sequences
            
            # Extract labels
            self.labels = groundtruth_data.labels if hasattr(groundtruth_data, constants.ATTR_LABELS) else {}
            
            # Detect number of classes from labels
            if self.labels:
                unique_labels = set(self.labels.values())
                self.num_classes = max(unique_labels) + 1
                logger.info(f"Detected {self.num_classes} classes from labels: {sorted(unique_labels)}")
            else:
                self.num_classes = 2  # Default to binary classification
                logger.warning("No labels found, defaulting to binary classification")
            
            logger.info(f"Extracted {len(self.keys)} samples with {len(self.messages)} messages, "
                       f"{len(self.sequences)} sequences, and {len(self.labels)} labels")
            
        except Exception as e:
            logger.error(f"Error processing ground truth data: {e}")
            raise RuntimeError(f"Failed to process ground truth: {e}")

        
        # Validate and filter keys to ensure consistency across modalities
        invalid_keys = []
        for key in self.keys[:]:
            if not (key in self.messages and key in self.sequences and key in self.labels):
                invalid_keys.append(key)
                self.keys.remove(key)
        
        if invalid_keys:
            logger.warning(f"Removed {len(invalid_keys)} invalid keys missing from one or more modalities")
            if self.print_hints:
                print(f"Warning: Removed {len(invalid_keys)} samples with missing data")
        
        if not self.keys:
            error_msg = "No valid samples found after filtering"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Validated {len(self.keys)} consistent samples across all modalities")

        
        # Create or load vocabulary and word embeddings
        try:
            if self.print_hints:
                print("Preprocessing messages for vocabulary extraction...")
            
            self.preprocessed_messages = preprocess_messages(self.messages)
            
            if tokens_dict is not None:
                # Use provided vocabulary (for validation/test sets)
                self.tokens_dict = tokens_dict
                self.embeddings = None  # Not needed when tokens are provided
                logger.info(f"Using provided vocabulary with {len(tokens_dict)} tokens")
            else:
                # Create new vocabulary and extract embeddings (for training set)
                self.tokens_dict, self.embeddings = create_embeddings_dict(
                    self.preprocessed_messages, 
                    str(self.data_path),
                    print_hints=self.print_hints
                )
                logger.info(f"Created vocabulary with {len(self.tokens_dict)} tokens")
            
            self.vocab_size = len(self.tokens_dict)
            
        except Exception as e:
            logger.error(f"Error creating vocabulary: {e}")
            raise RuntimeError(f"Failed to create vocabulary: {e}")

        
        # Store sequence length configuration
        self.max_len_messages = args.len_messages
        self.max_len_sequences = args.len_sequences
        
        if self.print_hints:
            print(f"\nDataset initialized successfully:")
            print(f"  - Split: {data_split}")
            print(f"  - Samples: {len(self.keys)}")
            print(f"  - Vocabulary size: {self.vocab_size}")
            print(f"  - Max message length: {self.max_len_messages}")
            print(f"  - Max sequence length: {self.max_len_sequences}\n")
        
        logger.info(f"GroundTruthLoader initialized: {len(self.keys)} samples, "
                   f"vocab_size={self.vocab_size}, max_msg_len={self.max_len_messages}, "
                   f"max_seq_len={self.max_len_sequences}")

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single data sample by index.
        
        This method retrieves and processes a single sample from the dataset,
        converting raw data into tensors suitable for model training.
        
        Parameters
        ----------
        index : int
            Index of the sample to retrieve (0 <= index < len(dataset)).
        
        Returns
        -------
        Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]
            key : str
                Unique identifier for this sample
            message : torch.Tensor
                Token indices for the log message, shape (max_len_messages,)
            sequence : torch.Tensor
                Sequential embeddings, shape (max_len_sequences, embedding_dim)
            label : torch.Tensor
                Binary label tensor, shape (1,) with value 0 (anomaly) or 1 (normal)
        
        Raises
        ------
        IndexError
            If index is out of range.
        
        Examples
        --------
        >>> dataset = GroundTruthLoader('train_set', args)
        >>> key, message, sequence, label = dataset[0]
        >>> print(f"Message shape: {message.shape}")
        Message shape: torch.Size([80])
        >>> print(f"Sequence shape: {sequence.shape}")
        Sequence shape: torch.Size([50, 300])
        >>> print(f"Label: {label.item()}")
        Label: 1
        
        Notes
        -----
        - Messages are converted to token indices and padded/truncated
        - Sequences are padded/truncated to fixed length
        - Labels are converted to numpy arrays before PyTorch tensors
        - All tensors are on CPU by default (move to GPU in training loop)
        """
        try:
            # Get sample key
            if index < 0 or index >= len(self.keys):
                raise IndexError(f"Index {index} out of range for dataset of size {len(self.keys)}")
            
            key = self.keys[index]
            
            # Convert message to token indices
            message = create_message_indices(
                self.preprocessed_messages[key], 
                self.tokens_dict, 
                max_tokens=self.max_len_messages
            )
            
            # Pad or truncate sequence
            sequence = pad_sequence(
                self.sequences[key], 
                self.max_len_sequences
            )
            
            # Get and validate label
            label_value = self.labels[key]
            label = get_label(label_value, max_classes=self.num_classes)
            label_array = np.array(label, dtype=np.int64)
            
            # Convert to PyTorch tensors
            return (
                key,
                torch.from_numpy(message),
                torch.from_numpy(sequence),
                torch.from_numpy(label_array)
            )
            
        except KeyError as e:
            logger.error(f"Key error accessing sample {index}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error retrieving sample {index}: {e}")
            raise
    
    def __len__(self) -> int:
        """Return the total number of samples in the dataset.
        
        This method is required by PyTorch's Dataset interface and is used
        by DataLoader to determine iteration length and random sampling bounds.
        
        Returns
        -------
        int
            Total number of valid samples in this dataset split.
        
        Examples
        --------
        >>> dataset = GroundTruthLoader('train_set', args)
        >>> print(f"Dataset contains {len(dataset)} samples")
        Dataset contains 5000 samples
        
        Notes
        -----
        The length represents only valid samples that passed filtering
        during initialization (samples present in all modalities).
        """
        return len(self.keys)