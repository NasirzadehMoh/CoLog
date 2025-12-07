"""
groundtruth/settings.py — dataset configuration helper for preprocessing

This file provides a compact Settings class used by the preprocessing
scripts (e.g., `preprocess.py`) and other utilities in `groundtruth/` to
discover dataset-specific directories, parsing formats and other
parameters specific to each log collection.

Key configuration fields explained:
  - in_dir: relative path under `datasets/` where the raw, unstructured logs
            for the dataset are stored (e.g., `hadoop/logs/`)
  - out_dir: relative path where structured parsed logs will be written
  - embs_dir: relative path where message embeddings should be stored
  - groundtruth_dir: relative path for groundtruth labels or additional
                     dataset metadata
  - log_format: parser log format string used by Drain or other rule-based
                parsers. It tells the parser how to split a log line into
                fields like Date, Time, Level, and Content.
  - regex: list of regexes that the parser uses to tokenize or mask
           dynamic parts in messages (IP addresses, memory amounts, file
           paths, UUIDs, etc.) — these are removed/abstracted during parsing
  - st: similarity threshold parameter used by some parsers
  - depth: parse tree depth parameter used by some parsers

How to add another dataset:
  1) Add a new entry to the `settings` dict with the new dataset's name
     and paths relative to `datasets/`.
  2) Add parsing parameters (log_format / regex / st / depth) if the
     dataset needs to be parsed by Drain. If the dataset uses an NER parser,
     `log_format/st/depth` can be omitted.
    3) Add the dataset name to the appropriate lists used by preprocessing
         scripts (Drain vs NER) — these lists live in `preprocess.py`.

Note: `preprocess.py` is a CLI tool with additional options such as
`--dataset-dir`, `--model`, `--batch-size`, `--device` and `--verbose`.
When running the script, ensure those flags are passed where applicable.

Usage:
    settings = Settings('hadoop')
    hadoop_settings = settings.get_settings()

Note: The Settings object stores dataset names; it does not validate the
existence of the directories on disk. The caller should verify folder
existence if necessary.
"""

class Settings(object):
    """Simple container that returns dataset-specific config dictionaries.

    Parameters
    ----------
    log_type : str
        Name of the dataset used as the key in the internal `settings` dict.
    """

    def __init__(self, log_type):
        self.log_type = log_type

    def get_settings(self):
        """
        Return a configuration dict for the `log_type` stored on this
        instance. The returned dictionary contains the relevant paths and
        parsing parameters for each supported dataset.

        Returns
        -------
        dict
            The dataset configuration matching the provided `log_type`.
        """

        # The `settings` map uses dataset name keys. Each dataset dict contains
        # the commonly needed fields for parsing and embedding.
        settings = {
            'hadoop': {
                'in_dir': 'hadoop/logs/',
                'out_dir': 'hadoop/logs_structured/',
                'embs_dir': 'hadoop/log_embeddings/',
                'groundtruth_dir': 'hadoop/groundtruth/',
                'log_format': '<Date> <Time> <Level> \\[<Process>\\] <Component>: <Content>', 
                'regex': [r'(\d+\.){3}\d+'],
                'st': 0.5,
                'depth': 4
                },

            'spark': {
                'in_dir': 'spark/logs/',
                'out_dir': 'spark/logs_structured/',
                'embs_dir': 'spark/log_embeddings/',
                'groundtruth_dir': 'spark/groundtruth/',
                'log_format': '<Date> <Time> <Level> <Component>: <Content>', 
                'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
                'st': 0.5,
                'depth': 4
                },

            'zookeeper': {
                'in_dir': 'zookeeper/logs/',
                'out_dir': 'zookeeper/logs_structured/',
                'embs_dir': 'zookeeper/log_embeddings/',
                'groundtruth_dir': 'zookeeper/groundtruth/',
                'log_format': '<Date> <Time> - <Level>  \\[<Node>:<Component>@<Id>\\] - <Content>',
                'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?'],
                'st': 0.5,
                'depth': 4
                },

            'bgl': {
                'in_dir': 'bgl/logs/',
                'out_dir': 'bgl/logs_structured/',
                'embs_dir': 'bgl/log_embeddings/',
                'groundtruth_dir': 'bgl/groundtruth/',
                'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
                'regex': [r'core\.\d+'],
                'st': 0.5,
                'depth': 4
                },

            'windows': {
                'in_dir': 'windows/logs/',
                'out_dir': 'windows/logs_structured/',
                'embs_dir': 'windows/log_embeddings/',
                'groundtruth_dir': 'windows/groundtruth/',
                'log_format': '<Date> <Time>, <Level> <Component> <Content>',
                'regex': [r'0x.*?\s'],
                'st': 0.7,
                'depth': 5
                },

            'casper-rw': {
                'in_dir': 'casper-rw/logs/',
                'out_dir': 'casper-rw/logs_structured/',
                'embs_dir': 'casper-rw/log_embeddings/',
                'groundtruth_dir': 'casper-rw/groundtruth/',
                },

            'dfrws-2009-jhuisi': {
                'in_dir': 'dfrws-2009-jhuisi/logs/',
                'out_dir': 'dfrws-2009-jhuisi/logs_structured/',
                'embs_dir': 'dfrws-2009-jhuisi/log_embeddings/',
                'groundtruth_dir': 'dfrws-2009-jhuisi/groundtruth/',
                },

            'dfrws-2009-nssal': {
                'in_dir': 'dfrws-2009-nssal/logs/',
                'out_dir': 'dfrws-2009-nssal/logs_structured/',
                'embs_dir': 'dfrws-2009-nssal/log_embeddings/',
                'groundtruth_dir': 'dfrws-2009-nssal/groundtruth/',
                },

            'honeynet-challenge7': {
                'in_dir': 'honeynet-challenge7/logs/',
                'out_dir': 'honeynet-challenge7/logs_structured/',
                'embs_dir': 'honeynet-challenge7/log_embeddings/',
                'groundtruth_dir': 'honeynet-challenge7/groundtruth/',
                },

            'honeynet-challenge5': {
                'in_dir': 'honeynet-challenge5/logs/',
                'out_dir': 'honeynet-challenge5/logs_structured/',
                'embs_dir': 'honeynet-challenge5/log_embeddings/',
                'groundtruth_dir': 'honeynet-challenge5/groundtruth/',
                }
        }

        # For each supported dataset we return the corresponding dict.
        # If none match below, the method will return None — callers should
        # validate or handle that case. We intentionally avoid raising
        # exceptions here so the module remains a simple lookup helper.
        if self.log_type == 'hadoop':
            return settings['hadoop']

        if self.log_type == 'spark':
            return settings['spark']

        if self.log_type == 'zookeeper':
            return settings['zookeeper']

        if self.log_type == 'bgl':
            return settings['bgl']

        if self.log_type == 'windows':
            return settings['windows']

        if self.log_type == 'casper-rw':
            return settings['casper-rw']

        if self.log_type == 'dfrws-2009-jhuisi':
            return settings['dfrws-2009-jhuisi']

        if self.log_type == 'dfrws-2009-nssal':
            return settings['dfrws-2009-nssal']

        if self.log_type == 'honeynet-challenge7':
            return settings['honeynet-challenge7']

        if self.log_type == 'honeynet-challenge5':
            return settings['honeynet-challenge5']

        # If nothing matched, return None. Callers should either provide a
        # valid dataset key or handle the None return to give a useful message
        # to the user. Example usage in preprocess.py checks dataset_name
        # before calling this method.
