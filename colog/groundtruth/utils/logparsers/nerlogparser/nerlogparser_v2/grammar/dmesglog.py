import os
import csv
from pyparsing import Word, alphas, nums, Optional, Regex
from collections import OrderedDict
from nerlogparser.grammar.grammar_utility import GrammarUtility


class DmesgLog(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.groups = {
            'group1': ['casper-rw', 'dfrws-2009-jhuisi', 'honeynet-challenge5', 'honeynet-challenge7'],
            'group2': ['dfrws-2009-nssal']
        }

    def get_grammar(self):
        dl = None
        if self.dataset in self.groups['group1']:
            dl = DmesgLog1(self.dataset)

        elif self.dataset in self.groups['group2']:
            dl = DmesgLog2(self.dataset)

        return dl


class DmesgLog1(object):
    def __init__(self, dataset):
        """Constructor for class DmesgLog.

        Parameters
        ----------
        dataset : str
            Dataset name.
        """
        self.dataset = dataset
        self.dmesglog_grammar = self.__get_dmesglog_grammar()

    @staticmethod
    def __get_dmesglog_grammar():
        """The definition of dmesg log grammar. Supported dataset:
        casper-rw
        dfrws-2009-jhuisi
        honeynet-challenge5
        honeynet-challenge7

        Returns
        -------
        dmesglog_grammar : pyparsing.And
            Grammar for dmesg log
        """
        # unix time
        unix_time = '[' + Word(nums + '.' + ']')
        subservice = Optional(Word(alphas + nums + '_' + '-' + ':' + '[' + ']' + '.' + '='))
        subservice_two_words = Optional(Word(alphas + nums + '_' + '-' + ':' + '[' + ']' + ',' + '.' + '='))
        message = Regex('.*')

        # dmesg log grammar
        dmesglog_grammar = unix_time + subservice + subservice_two_words + message
        return dmesglog_grammar

    def parse_log(self, log_line):
        """Parse dmesg log based on defined grammar.

        Parameters
        ----------
        log_line    : str
            A log line to be parsed.

        Returns
        -------
        parsed  : dict[str, str]
            A parsed dmesg log containing these elements: unix_time, subservice and message.
        """
        parsed_dmesglog = self.dmesglog_grammar.parseString(log_line)

        # get parsed dmesg log
        parsed = OrderedDict()
        parsed['unix_time'] = GrammarUtility.get_unix_timestamp(parsed_dmesglog[0], parsed_dmesglog[1])

        if len(parsed_dmesglog) == 3:
            parsed['subservice'] = ''
            parsed['message'] = parsed_dmesglog[2]

        else:
            # subservice one word
            if parsed_dmesglog[2].endswith(':') and not parsed_dmesglog[3].endswith(':'):
                parsed['subservice'] = parsed_dmesglog[2]
                parsed['message'] = ' '.join(parsed_dmesglog[3:])

            # subservice two words
            elif not parsed_dmesglog[2].endswith(':') and parsed_dmesglog[3].endswith(':'):
                parsed['subservice'] = parsed_dmesglog[2] + ' ' + parsed_dmesglog[3]
                parsed['message'] = ' '.join(parsed_dmesglog[4:])

            # subservice two words
            elif parsed_dmesglog[2].endswith(':') and parsed_dmesglog[3].endswith(':'):
                parsed['subservice'] = parsed_dmesglog[2] + ' ' + parsed_dmesglog[3]
                parsed['message'] = ' '.join(parsed_dmesglog[4:])

            else:
                parsed['subservice'] = ''
                parsed['message'] = ' '.join(parsed_dmesglog[2:])

            if not parsed['subservice'].endswith(':') and parsed['subservice']:
                parsed['message'] = parsed['subservice'] + ' ' + parsed['message']
                parsed['subservice'] = ''

        return parsed


class DmesgLog2(object):
    # this class is written for dfrws-2009-jhuisi/nssal/dmesg*
    def __init__(self, dataset):
        self.dataset = dataset
        self.dmesglog_grammar = self.__get_dmesglog_grammar()

    @staticmethod
    def __get_dmesglog_grammar():
        subservice = Optional(Word(alphas + nums + '_' + '-' + ':' + '[' + ']' + '.' + '='))
        subservice_two_words = Optional(Word(alphas + nums + '_' + '-' + ':' + '[' + ']' + ',' + '.' + '='))
        message = Regex('.*')

        # dmesg log grammar
        dmesglog_grammar = subservice + subservice_two_words + message
        return dmesglog_grammar

    def parse_log(self, log_line):
        # get grammar
        parsed_dmesglog = self.dmesglog_grammar.parseString(log_line)

        # get parsed dmesg log
        parsed = OrderedDict()
        if len(parsed_dmesglog) == 1:
            if parsed_dmesglog[0].endswith(':'):
                parsed['subservice'] = parsed_dmesglog[0]
                parsed['message'] = ''
            else:
                parsed['subservice'] = ''
                parsed['message'] = parsed_dmesglog[0]

        elif len(parsed_dmesglog) == 2:
            if parsed_dmesglog[0].endswith(':'):
                parsed['subservice'] = parsed_dmesglog[0]
                parsed['message'] = parsed_dmesglog[1]
            else:
                parsed['subservice'] = ''
                parsed['message'] = ' '.join(parsed_dmesglog[0:])

        else:
            # subservice one word
            if parsed_dmesglog[0].endswith(':') and not parsed_dmesglog[1].endswith(':'):
                parsed['subservice'] = parsed_dmesglog[0]
                parsed['message'] = ' '.join(parsed_dmesglog[1:])

            # subservice two words
            elif not parsed_dmesglog[0].endswith(':') and parsed_dmesglog[1].endswith(':'):
                parsed['subservice'] = parsed_dmesglog[0] + ' ' + parsed_dmesglog[1]
                parsed['message'] = ' '.join(parsed_dmesglog[2:])

            # subservice two words
            elif parsed_dmesglog[0].endswith(':') and parsed_dmesglog[1].endswith(':'):
                parsed['subservice'] = parsed_dmesglog[0] + ' ' + parsed_dmesglog[1]
                parsed['message'] = ' '.join(parsed_dmesglog[2:])

            else:
                parsed['subservice'] = ''
                parsed['message'] = ' '.join(parsed_dmesglog[:])

            if not parsed['subservice'].endswith(':') and parsed['subservice']:
                parsed['message'] = parsed['subservice'] + ' ' + parsed['message']
                parsed['subservice'] = ''

        return parsed


class Main(object):
    def __init__(self, datasets):
        self.datasets = datasets
        self.dataset_path = '/home/hudan/Git/prlogparser/datasets/'
        self.groups = {
            'group1': ['casper-rw', 'dfrws-2009-jhuisi', 'honeynet-challenge5', 'honeynet-challenge7'],
            'group2': ['dfrws-2009-nssal']
        }

    def run(self):
        # parse dmesg log
        for group_name, group in self.groups.items():
            # setup test csv file to save results
            base_name = '/home/hudan/Git/prlogparser/groundtruth/dmesg-'
            test_file = base_name + group_name + '.csv'
            f = open(test_file, 'w', newline='')
            writer = csv.writer(f)

            for dataset in group:
                # get grammar
                dl = None
                if group_name == 'group1':
                    dl = DmesgLog1(dataset)

                elif group_name == 'group2':
                    dl = DmesgLog2(dataset)

                # start parsing
                for filename in self.datasets[dataset]:
                    filename = os.path.join(self.dataset_path, filename)
                    with open(filename, 'r') as f:
                        for line in f:
                            # get parsed line and print
                            parsed_line = dl.parse_log(line)
                            print(parsed_line)

                            # write to csv
                            row = list(parsed_line.values())
                            writer.writerow(row)

            f.close()


if __name__ == '__main__':
    # get dmesg log datasets
    datasets_files = {
        'casper-rw': [
            'casper-rw/dmesg',
            'casper-rw/dmesg.0',
            'casper-rw/dmesg.1',
            'casper-rw/dmesg.2',
            'casper-rw/dmesg.3'
        ],
        'dfrws-2009-jhuisi': [
            'dfrws-2009-jhuisi/dmesg',
            'dfrws-2009-jhuisi/dmesg.0',
            'dfrws-2009-jhuisi/dmesg.1',
            'dfrws-2009-jhuisi/dmesg.2',
            'dfrws-2009-jhuisi/dmesg.3',
            'dfrws-2009-jhuisi/dmesg.4'
        ],
        'dfrws-2009-nssal': [
            'dfrws-2009-nssal/dmesg',
            'dfrws-2009-nssal/dmesg.0',
            'dfrws-2009-nssal/dmesg.1',
            'dfrws-2009-nssal/dmesg.2',
            'dfrws-2009-nssal/dmesg.3',
            'dfrws-2009-nssal/dmesg.4'
        ],
        'honeynet-challenge5': [
            'honeynet-challenge5/dmesg',
            'honeynet-challenge5/dmesg.0'
        ],
        'honeynet-challenge7': [
            'honeynet-challenge7/dmesg',
            'honeynet-challenge7/dmesg.0',
            'honeynet-challenge7/dmesg.1',
            'honeynet-challenge7/dmesg.2',
            'honeynet-challenge7/dmesg.3',
            'honeynet-challenge7/dmesg.4'
        ]
    }

    main = Main(datasets_files)
    main.run()
