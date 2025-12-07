import os
import errno
import pickle
from configparser import ConfigParser
from nerlogparser.grammar.authlog import AuthLog
from nerlogparser.grammar.daemonlog import DaemonLog
from nerlogparser.grammar.debuglog import DebugLog
from nerlogparser.grammar.dmesglog import DmesgLog
from nerlogparser.grammar.kernellog import KernelLog
from nerlogparser.grammar.messageslog import MessagesLog
from nerlogparser.grammar.csvlog import CSVLog
from nerlogparser.grammar.bluegenelog import BlueGeneLog
from nerlogparser.grammar.kippolog import KippoLog
from nerlogparser.grammar.proxifierlog import ProxifierLog
from nerlogparser.grammar.weblog import WebLog
from nerlogparser.grammar.zookeeperlog import ZookeeperLog

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

class Preprocessing(object):
    """This class do four main tasks:
    1. Parse log entries based on developer-defined grammar
    2. Put punctuations between parsed entities
    3. Save punctuations result to files
    4. Save parsed log entries to pickle files to be used in Splitting.py.

    """
    def __init__(self, data):
        self.dataset = data
        self.dataset_conf = {}
        self.files = {}

    @staticmethod
    def __check_path(path):
        """Check a path is exist or not. If not exist, then create it.

        Parameters
        ----------
        path    : str
            Path of a directory to be checked.
        """
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

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

        # get dataset and groundtruth path
        dataset_path = os.path.join(self.dataset_conf['main']['dataset_path'], self.dataset)
        groundtruth_path = os.path.join(self.dataset_conf['main']['groundtruth_path'], self.dataset)
        groundtruth_pickle_path = os.path.join(self.dataset_conf['main']['groundtruth_pickle'], self.dataset)
        self.__check_path(groundtruth_path)
        self.__check_path(groundtruth_pickle_path)
        filenames = os.listdir(dataset_path)

        # get full path of each filename
        for filename in filenames:
            self.files[filename] = {
                'log_path': os.path.join(dataset_path, filename),
                'groundtruth_path': os.path.join(groundtruth_path, filename),
                'groundtruth_pickle': os.path.join(groundtruth_pickle_path, filename),
                'type': filename.split('.')[0]
            }

    def __get_grammar(self, file_type):
        if file_type == 'auth':
            authlog_grammar = AuthLog(self.dataset)
            return authlog_grammar

        elif file_type == 'daemon':
            daemonlog_grammar = DaemonLog(self.dataset)
            return daemonlog_grammar

        elif file_type == 'debug':
            debuglog_grammar = DebugLog(self.dataset)
            return debuglog_grammar.get_grammar()

        elif file_type == 'dmesg':
            dmesglog_grammar = DmesgLog(self.dataset)
            return dmesglog_grammar.get_grammar()

        elif file_type == 'kern':
            kernellog_grammar = KernelLog(self.dataset)
            return kernellog_grammar.get_grammar()

        elif file_type == 'messages' or file_type == 'syslog':
            messageslog_grammar = MessagesLog(self.dataset)
            return messageslog_grammar.get_grammar()

        elif file_type == 'csv':
            messageslog_grammar = CSVLog(self.dataset)
            return messageslog_grammar

        elif file_type == 'bgl2':
            bluegenelog_grammar = BlueGeneLog(self.dataset)
            return bluegenelog_grammar

        elif file_type == 'kippo':
            kippolog_grammar = KippoLog(self.dataset)
            return kippolog_grammar

        elif file_type == 'proxifier':
            proxifierlog_grammar = ProxifierLog(self.dataset)
            return proxifierlog_grammar

        elif file_type == 'access':
            weblog_grammar = WebLog(self.dataset)
            return weblog_grammar

        elif file_type == 'zookeeper':
            zookeeperlog_grammar = ZookeeperLog(self.dataset)
            return zookeeperlog_grammar

    @staticmethod
    def __set_punctuation(parsed_line):
        # set punctuation from parsed line
        punctuated = ''
        for field_name, field_value in parsed_line.items():
            if field_value != '' and field_value != ' ':
                if field_name == 'message' or field_name == 'client_agent':
                    # if there is no period, then add one
                    field_value = field_value.rstrip()
                    if field_value[-1] != '.':
                        punctuated += field_value + ' .PERIOD\n'
                    else:
                        punctuated += field_value + ' .PERIOD\n'
                else:
                    punctuated += field_value + ' ,COMMA '
            else:
                if field_name == 'message':
                    punctuated += '\n'
                elif field_name == 'client_agent':
                    punctuated += ' .PERIOD\n'

        return punctuated

    def punctuate(self):
        # get dataset
        self.__get_dataset()

        # punctuate log entries
        for filename, properties in self.files.items():
            print(self.dataset, filename)

            # get grammar based on file type
            file_type = properties['type']
            grammar = self.__get_grammar(file_type)

            # prepare output files
            f_groundtruth = open(properties['groundtruth_path'], 'w')

            # parse log entries
            parsed_list = []
            if file_type != 'csv':
                with open(properties['log_path'], 'r') as f:
                    for line in f:
                        parsed = grammar.parse_log(line)
                        parsed_list.append(parsed)

                        # set punctuation
                        punctuated_line = self.__set_punctuation(parsed)
                        f_groundtruth.write(punctuated_line)

            else:
                parsed_lines = grammar.parse_log()
                for parsed_line in parsed_lines:
                    parsed_list.append(parsed_line)
                    punctuated_line = self.__set_punctuation(parsed_line)
                    f_groundtruth.write(punctuated_line)

            f_groundtruth.close()

            # save parsed list to a pickle file
            with open(properties['groundtruth_pickle'], 'wb') as f_pickle:
                pickle.dump(parsed_list, f_pickle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # put punctuation (comma and period) to all datasets
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

    for dataset in datasets:
        preprocess = Preprocessing(dataset)
        preprocess.punctuate()
