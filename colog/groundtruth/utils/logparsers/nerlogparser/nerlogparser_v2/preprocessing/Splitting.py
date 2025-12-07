import os
import errno
import shutil
import pickle
from configparser import ConfigParser
from math import floor
from nerlogparser.dataformat.toconll import ToConll

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

class Splitting(object):
    # split each file to three parts: train, dev, and test
    # compositition: train: 60, dev: 20, test: 20
    def __init__(self, data):
        self.dataset = data
        self.dataset_conf = {}
        self.files = {}
        self.punctuation_path = ''
        self.conll_path = ''
        self.train_file = ''
        self.dev_file = ''
        self.test_file = ''
        self.train_file_conll = ''
        self.dev_file_conll = ''
        self.test_file_conll = ''
        self.conll_stanford_path = ''
        self.train_file_conll_stanford = ''
        self.dev_file_conll_stanford = ''
        self.test_file_conll_stanford = ''
        self.conll_pos_path = ''
        self.train_file_conll_pos = ''
        self.dev_file_conll_pos = ''
        self.test_file_conll_pos = ''
        self.csv_path = ''
        self.file_csv = ''
        self.nltk_tree_path = ''
        self.train_file_nltk_tree = ''
        self.test_file_nltk_tree = ''

    @staticmethod
    def __check_path(path):
        # check a path is exist or not. if not exist, then create it
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    def __set_datapath(self):
        # set data path for output files
        self.__check_path(self.punctuation_path)
        self.train_file = os.path.join(self.punctuation_path, 'ep.train.txt')
        self.dev_file = os.path.join(self.punctuation_path, 'ep.dev.txt')
        self.test_file = os.path.join(self.punctuation_path, 'ep.test.txt')

    def __set_datapath_conll(self):
        # set data path for output files
        self.__check_path(self.conll_path)
        self.train_file_conll = os.path.join(self.conll_path, 'conll.train.txt')
        self.dev_file_conll = os.path.join(self.conll_path, 'conll.dev.txt')
        self.test_file_conll = os.path.join(self.conll_path, 'conll.test.txt')

    def __set_datapath_conll_stanford(self):
        # set data path for output files
        self.__check_path(self.conll_stanford_path)
        self.train_file_conll_stanford = os.path.join(self.conll_stanford_path, 'conll.stanford.train.txt')
        self.dev_file_conll_stanford = os.path.join(self.conll_stanford_path, 'conll.stanford.dev.txt')
        self.test_file_conll_stanford = os.path.join(self.conll_stanford_path, 'conll.stanford.test.txt')

    def __set_datapath_conll_pos(self):
        # set data path for output files
        self.__check_path(self.conll_pos_path)
        self.train_file_conll_pos = os.path.join(self.conll_pos_path, 'conll.pos.train.txt')
        self.dev_file_conll_pos = os.path.join(self.conll_pos_path, 'conll.pos.dev.txt')
        self.test_file_conll_pos = os.path.join(self.conll_pos_path, 'conll.pos.test.txt')

    def __set_datapath_nltk_tree(self):
        # set data path for output files
        self.__check_path(self.nltk_tree_path)
        self.train_file_nltk_tree = os.path.join(self.nltk_tree_path, 'nltk.tree.train.txt')
        self.test_file_nltk_tree = os.path.join(self.nltk_tree_path, 'nltk.tree.test.txt')

    def __set_datapath_csv(self):
        self.__check_path(self.csv_path)
        self.file_csv = os.path.join(self.csv_path, 'csv.all.txt')

    def __get_dataset(self):
        # get dataset configuration
        current_path = os.path.dirname(os.path.realpath(__file__))
        dataset_config_path = os.path.join(current_path, 'config', 'datasets.conf')

        # read dataset path from .conf file
        parser = ConfigParser()
        parser.read(dataset_config_path)
        for section_name in parser.sections():
            options = {}
            for name, value in parser.items(section_name):
                options[name] = value
            self.dataset_conf[section_name] = options

        # set output path
        self.punctuation_path = self.dataset_conf['main']['punctuation_path']
        self.conll_path = self.dataset_conf['main']['conll_path']
        self.conll_stanford_path = self.dataset_conf['main']['conll_stanford_path']
        self.conll_pos_path = self.dataset_conf['main']['conll_pos_path']
        self.csv_path = self.dataset_conf['main']['csv_path']
        self.nltk_tree_path = self.dataset_conf['main']['nltk_tree_path']

        # get dataset and groundtruth path
        dataset_path = os.path.join(self.dataset_conf['main']['dataset_path'], self.dataset)
        groundtruth_path = os.path.join(self.dataset_conf['main']['groundtruth_path'], self.dataset)
        groundtruth_pickle = os.path.join(self.dataset_conf['main']['groundtruth_pickle'], self.dataset)
        filenames = os.listdir(dataset_path)

        # get full path of each filename
        for filename in filenames:
            self.files[filename] = {
                'log_path': os.path.join(dataset_path, filename),
                'groundtruth_path': os.path.join(groundtruth_path, filename),
                'groundtruth_pickle_path': os.path.join(groundtruth_pickle, filename)
            }

    def split(self):
        # get dataset and output file path
        self.__get_dataset()
        self.__set_datapath()

        # open files for train, dev, and test
        f_train = open(self.train_file, 'a')
        f_dev = open(self.dev_file, 'a')
        f_test = open(self.test_file, 'a')

        for filename, properties in self.files.items():
            print('punctuation file:', self.dataset, properties['log_path'])

            with open(properties['groundtruth_path'], 'r') as f:
                # read lines and get various length
                lines = f.readlines()

            # note: test_length = dev_length
            lines_length = len(lines)
            train_length = floor(0.6 * lines_length)
            dev_length = floor(0.2 * lines_length)
            dev_end_index = train_length + dev_length

            # get training, dev, and test data
            for line in lines[:train_length]:
                f_train.write(line)

            for line in lines[train_length:dev_end_index]:
                f_dev.write(line)

            for line in lines[dev_end_index:]:
                f_test.write(line)

        # close files
        f_train.close()
        f_dev.close()
        f_test.close()

    def split_conll(self):
        # set conll output files and create conll-format instance
        self.__set_datapath_conll()
        conll = ToConll()

        # open files for train, dev, and test
        f_train = open(self.train_file_conll, 'a')
        f_dev = open(self.dev_file_conll, 'a')
        f_test = open(self.test_file_conll, 'a')

        for filename, properties in self.files.items():
            print('pickle file     :', self.dataset, properties['log_path'])

            # parsed_list is list of dictionaries containing parsed log entries
            with open(properties['groundtruth_pickle_path'], 'rb') as f_pickle:
                parsed_list = pickle.load(f_pickle)

            # note: test_length = dev_length
            lines_length = len(parsed_list)
            train_length = floor(0.6 * lines_length)
            dev_length = floor(0.2 * lines_length)
            dev_end_index = train_length + dev_length

            # get training, dev, and test data
            f_train.write('-DOCSTART- -X- O O\n\n')
            for line in parsed_list[:train_length]:
                line = conll.convert(line)
                f_train.write(line)

            f_dev.write('-DOCSTART- -X- O O\n\n')
            for line in parsed_list[train_length:dev_end_index]:
                line = conll.convert(line)
                f_dev.write(line)

            f_test.write('-DOCSTART- -X- O O\n\n')
            for line in parsed_list[dev_end_index:]:
                line = conll.convert(line)
                f_test.write(line)

        # close files
        f_train.close()
        f_dev.close()
        f_test.close()

    def split_conll_stanford(self):
        # set conll output files and create conll-format instance
        self.__set_datapath_conll_stanford()
        conll = ToConll()

        # open files for train, dev, and test
        f_train = open(self.train_file_conll_stanford, 'a')
        f_dev = open(self.dev_file_conll_stanford, 'a')
        f_test = open(self.test_file_conll_stanford, 'a')

        for filename, properties in self.files.items():
            print('pickle file     :', self.dataset, properties['log_path'])

            # parsed_list is list of dictionaries containing parsed log entries
            with open(properties['groundtruth_pickle_path'], 'rb') as f_pickle:
                parsed_list = pickle.load(f_pickle)

            # note: test_length = dev_length
            lines_length = len(parsed_list)
            train_length = floor(0.6 * lines_length)
            dev_length = floor(0.2 * lines_length)
            dev_end_index = train_length + dev_length

            # get training, dev, and test data
            for line in parsed_list[:train_length]:
                line = conll.convert(line, stanford=True)
                f_train.write(line)

            for line in parsed_list[train_length:dev_end_index]:
                line = conll.convert(line, stanford=True)
                f_dev.write(line)

            for line in parsed_list[dev_end_index:]:
                line = conll.convert(line, stanford=True)
                f_test.write(line)

        # close files
        f_train.close()
        f_dev.close()
        f_test.close()

    def split_conll_pos(self):
        # set conll output files and create conll-format instance
        self.__set_datapath_conll_pos()
        conll = ToConll()

        # open files for train, dev, and test
        f_train = open(self.train_file_conll_pos, 'a')
        f_dev = open(self.dev_file_conll_pos, 'a')
        f_test = open(self.test_file_conll_pos, 'a')

        for filename, properties in self.files.items():
            print('pickle file     :', self.dataset, properties['log_path'])

            # parsed_list is list of dictionaries containing parsed log entries
            with open(properties['groundtruth_pickle_path'], 'rb') as f_pickle:
                parsed_list = pickle.load(f_pickle)

            # note: test_length = dev_length
            lines_length = len(parsed_list)
            train_length = floor(0.6 * lines_length)
            dev_length = floor(0.2 * lines_length)
            dev_end_index = train_length + dev_length

            # get training, dev, and test data
            for line in parsed_list[:train_length]:
                line = conll.convert(line, ispos=True)
                f_train.write(line)

            for line in parsed_list[train_length:dev_end_index]:
                line = conll.convert(line, ispos=True)
                f_dev.write(line)

            for line in parsed_list[dev_end_index:]:
                line = conll.convert(line, ispos=True)
                f_test.write(line)

        # close files
        f_train.close()
        f_dev.close()
        f_test.close()

    def split_csv(self, f):
        # create conll-format instance
        conll = ToConll()

        for filename, properties in self.files.items():
            print('pickle file     :', self.dataset, properties['log_path'])

            # parsed_list is list of dictionaries containing parsed log entries
            with open(properties['groundtruth_pickle_path'], 'rb') as f_pickle:
                parsed_list = pickle.load(f_pickle)

            # get training, dev, and test data
            for line_id, line in enumerate(parsed_list):
                line_id += 1
                line = conll.convert(line, csv=True, csv_line_id=line_id)
                f.write(line)

    def split_nltk_tree(self):
        # set conll output files and create conll-format instance
        self.__set_datapath_nltk_tree()
        conll = ToConll()

        # open files for train and test
        f_train = open(self.train_file_nltk_tree, 'ab')
        f_test = open(self.test_file_nltk_tree, 'ab')

        for filename, properties in self.files.items():
            print('pickle file     :', self.dataset, properties['log_path'])

            # parsed_list is list of dictionaries containing parsed log entries
            with open(properties['groundtruth_pickle_path'], 'rb') as f_pickle:
                parsed_list = pickle.load(f_pickle)

            lines_length = len(parsed_list)
            train_length = floor(0.8 * lines_length)

            # get training, and test data
            for line in parsed_list[:train_length]:
                line = conll.convert(line, iobtree=True)
                pickle.dump(line, f_train)

            for line in parsed_list[train_length:]:
                line = conll.convert(line, iobtree=True)
                pickle.dump(line, f_test)

        # close files
        f_train.close()
        f_test.close()


def remove_directories():
    # remove output directories
    punctuation_path = '/home/hudan/Git/prlogparser/data/punctuation/'
    if os.path.isdir(punctuation_path):
        shutil.rmtree(punctuation_path)

    conll_path = '/home/hudan/Git/prlogparser/data/conll/'
    if os.path.isdir(conll_path):
        shutil.rmtree(conll_path)

    conll_path_stanford = '/home/hudan/Git/prlogparser/data/conll-stanford/'
    if os.path.isdir(conll_path_stanford):
        shutil.rmtree(conll_path_stanford)

    conll_path_pos = '/home/hudan/Git/prlogparser/data/conll-pos/'
    if os.path.isdir(conll_path_pos):
        shutil.rmtree(conll_path_pos)

    csv_path = '/home/hudan/Git/prlogparser/data/csv/'
    if os.path.isdir(csv_path):
        shutil.rmtree(csv_path)

    nltk_tree_path = '/home/hudan/Git/prlogparser/data/nltk-tree/'
    if os.path.isdir(nltk_tree_path):
        shutil.rmtree(nltk_tree_path)


def check_path(path):
    # check a path is exist or not. if not exist, then create it
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def open_csv_file():
    # set csv output files
    csv_path = '/home/hudan/Git/prlogparser/data/csv/'
    check_path(csv_path)
    csv_file = os.path.join(csv_path, 'csv.all.txt')
    f_csv = open(csv_file, 'a')
    f_csv.write('Sentence #\tWord\tPOS\tTag\n')

    return f_csv


if __name__ == '__main__':
    datasets = [
        'casper-rw',
        'dfrws-2009-jhuisi',
        'dfrws-2009-nssal',
        'dfrws-2016',
        'honeynet-challenge5',
        'honeynet-challenge7',
        'bgl2',
        'kippo',
        'proxifier',
        'secrepo-accesslog',
        'zookeeper'
    ]

    # remove to clean output directories because we use append mode for output files
    # and then run splitting for all datasets
    remove_directories()
    file_csv = open_csv_file()
    for dataset in datasets:
        s = Splitting(dataset)
        s.split()
        s.split_conll()
        s.split_conll_pos()
        s.split_csv(file_csv)
        # s.split_conll_stanford()
        # s.split_nltk_tree()

    file_csv.close()
