import os
import csv
from pyparsing import Word, alphas, Combine, nums, string, Optional, Regex
from collections import OrderedDict
from nerlogparser.grammar.grammar_utility import GrammarUtility


class KernelLog(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.groups = {
            'group1': ['casper-rw', 'dfrws-2009-jhuisi', 'dfrws-2009-nssal', 'honeynet-challenge7'],
            'group2': ['honeynet-challenge5']
        }

    def get_grammar(self):
        dl = None
        if self.dataset in self.groups['group1']:
            dl = KernelLog1(self.dataset)

        elif self.dataset in self.groups['group2']:
            dl = KernelLog2(self.dataset)

        return dl


class KernelLog1(object):
    def __init__(self, dataset):
        """Constructor for class KernelLog.

        Parameters
        ----------
        dataset : str
            Dataset name.
        """
        self.dataset = dataset

    @staticmethod
    def __get_kernellog_grammar():
        """The definition of kernel log grammar. Supported dataset:
        casper-rw
        dfrws-2009
        honeynet-challenge7

        Returns
        -------
        kernellog_grammar :
            Grammar for kernel log
        """
        ints = Word(nums)

        # timestamp
        month = Word(string.ascii_uppercase, string.ascii_lowercase, exact=3)
        day = ints
        hour = Combine(ints + ':' + ints + ':' + ints)
        timestamp = month + day + hour

        # hostname, service name, message
        hostname = Word(alphas + nums + '_' + '-' + '.')
        service = Word(alphas + nums + '/' + '-' + '_' + '.' + '[' + ']' + ':')

        # unix time
        unix_time = Optional('[' + Word(nums + '.' + ']'))
        subservice = Optional(Word(alphas + nums + '_' + '-' + ':' + '[' + ']' + '.' + '=' + '(' + ')' + '*' +
                                   '<' + '>'))
        subservice_two_words = Optional(Word(alphas + nums + '_' + '-' + ':' + '[' + ']' + ',' + '.' + '=' + '/' +
                                             '(' + ')' + '*'))
        message = Optional(Regex('.*'))

        # kernel log grammar
        debuglog_grammar = timestamp + hostname + service + unix_time + subservice + subservice_two_words + message
        return debuglog_grammar

    def parse_log(self, log_line):
        """Parse kernel log based on defined grammar.

        Parameters
        ----------
        log_line    : str
            A log line to be parsed.

        Returns
        -------
        parsed  : dict[str, str]
            A parsed kernel log containing these elements: timestamp, hostname, service, unix_time,
            subservice and message.
        """
        kernellog_grammar = self.__get_kernellog_grammar()
        parsed_kernellog = kernellog_grammar.parseString(log_line)

        # get parsed kernel log
        parsed = OrderedDict()
        parsed['timestamp'] = parsed_kernellog[0] + ' ' + parsed_kernellog[1] + ' ' + parsed_kernellog[2]
        parsed['hostname'] = parsed_kernellog[3]
        parsed['service'] = parsed_kernellog[4]

        if len(parsed_kernellog) == 6:
            parsed['unix_time'] = ''
            parsed['subservice'] = ''
            parsed['message'] = parsed_kernellog[5]

        elif len(parsed_kernellog) == 7:
            parsed['unix_time'] = ''
            parsed['subservice'] = ''
            if not parsed_kernellog[5].endswith(':'):
                parsed['message'] = parsed_kernellog[5] + ' ' + parsed_kernellog[6]

        elif len(parsed_kernellog) == 8:
            parsed['unix_time'] = ''
            parsed['subservice'] = ''

            # no message
            if parsed_kernellog[7] == '' and parsed_kernellog[5].startswith('[') and parsed_kernellog[6].endswith(']'):
                parsed['unix_time'] = GrammarUtility.get_unix_timestamp(parsed_kernellog[5], parsed_kernellog[6])
                parsed['message'] = ''

            # message exists
            elif parsed_kernellog[7] != '' and parsed_kernellog[5].startswith('[') and \
                    parsed_kernellog[6].endswith(']'):
                parsed['unix_time'] = GrammarUtility.get_unix_timestamp(parsed_kernellog[5], parsed_kernellog[6])
                parsed['message'] = parsed_kernellog[7]

            # subservice one word
            elif parsed_kernellog[5].endswith(':') and not parsed_kernellog[6].endswith(':'):
                parsed['subservice'] = parsed_kernellog[5]
                parsed['message'] = ' '.join(parsed_kernellog[6:])

            # subservice two words
            elif not parsed_kernellog[5].endswith(':') and parsed_kernellog[6].endswith(':'):
                parsed['subservice'] = parsed_kernellog[5] + ' ' + parsed_kernellog[6]
                parsed['message'] = parsed_kernellog[7]

            # subservice two words
            elif parsed_kernellog[5].endswith(':') and parsed_kernellog[6].endswith(':'):
                parsed['subservice'] = parsed_kernellog[5] + ' ' + parsed_kernellog[6]
                parsed['message'] = parsed_kernellog[7]

            # no timestamp, no subservice, just message
            else:
                parsed['message'] = ' '.join(parsed_kernellog[5:])

        else:
            # if timestamp exists
            parsed['unix_time'] = GrammarUtility.get_unix_timestamp(parsed_kernellog[5], parsed_kernellog[6])

            # subservice one word
            if parsed_kernellog[7].endswith(':') and not parsed_kernellog[8].endswith(':'):
                parsed['subservice'] = parsed_kernellog[7]
                parsed['message'] = ' '.join(parsed_kernellog[8:])

            # subservice two words
            elif not parsed_kernellog[7].endswith(':') and parsed_kernellog[8].endswith(':'):
                parsed['subservice'] = parsed_kernellog[7] + ' ' + parsed_kernellog[8]
                parsed['message'] = ' '.join(parsed_kernellog[9:])

            # subservice two words
            elif parsed_kernellog[7].endswith(':') and parsed_kernellog[8].endswith(':'):
                parsed['subservice'] = parsed_kernellog[7] + ' ' + parsed_kernellog[8]
                parsed['message'] = ' '.join(parsed_kernellog[9:])

            else:
                parsed['subservice'] = ''
                parsed['message'] = ' '.join(parsed_kernellog[7:])

            if not parsed['subservice'].endswith(':'):
                parsed['message'] = parsed['subservice'] + ' ' + parsed['message']
                parsed['subservice'] = ''

        return parsed


class KernelLog2(object):
    def __init__(self, dataset):
        """Constructor for class KernelLog.

        Parameters
        ----------
        dataset : str
            Dataset name.
        """
        self.dataset = dataset

    @staticmethod
    def __get_kernellog_grammar():
        """The definition of kernel log grammar. Supported dataset:
        honeynet-challenge5

        Returns
        -------
        kernellog_grammar :
            Grammar for kernel log
        """
        ints = Word(nums)

        # timestamp
        month = Word(string.ascii_uppercase, string.ascii_lowercase, exact=3)
        day = ints
        hour = Combine(ints + ':' + ints + ':' + ints)
        timestamp = month + day + hour

        # hostname, service name, message
        # there is Optional(':')
        hostname = Word(alphas + nums + '_' + '-' + '.')
        service = Word(alphas + nums + '/' + '-' + '_' + '.' + '[' + ']' + ':') + Optional(':')

        # unix time
        unix_time = Optional('[' + Word(nums + '.' + ']'))
        subservice = Optional(Word(alphas + nums + '_' + '-' + ':' + '[' + ']' + '.' + '=' + '(' + ')' + '*' +
                                   '<' + '>'))
        subservice_two_words = Optional(Word(alphas + nums + '_' + '-' + ':' + '[' + ']' + ',' + '.' + '=' + '/' +
                                             '(' + ')' + '*' + "'"))
        message = Optional(Regex('.*'))

        # kernel log grammar
        debuglog_grammar = timestamp + hostname + service + unix_time + subservice + subservice_two_words + message
        return debuglog_grammar

    def parse_log(self, log_line):
        """Parse kernel log based on defined grammar.

        Parameters
        ----------
        log_line    : str
            A log line to be parsed.

        Returns
        -------
        parsed  : dict[str, str]
            A parsed kernel log containing these elements: timestamp, hostname, service, unix_time,
            subservice and message.
        """
        kernellog_grammar = self.__get_kernellog_grammar()
        parsed_kernellog = kernellog_grammar.parseString(log_line)

        # get parsed kernel log
        parsed = OrderedDict()
        parsed['timestamp'] = parsed_kernellog[0] + ' ' + parsed_kernellog[1] + ' ' + parsed_kernellog[2]
        parsed['hostname'] = parsed_kernellog[3]
        parsed['service'] = parsed_kernellog[4]

        if len(parsed_kernellog) == 6:
            parsed['unix_time'] = ''
            parsed['subservice'] = ''
            parsed['message'] = parsed_kernellog[5]

        elif len(parsed_kernellog) == 7:
            parsed['unix_time'] = ''
            parsed['subservice'] = ''
            if not parsed_kernellog[5].endswith(':'):
                parsed['message'] = parsed_kernellog[5] + ' ' + parsed_kernellog[6]

        elif len(parsed_kernellog) == 8:
            parsed['unix_time'] = ''
            parsed['subservice'] = ''

            # timestamp exists
            if parsed_kernellog[5].startswith('[') and parsed_kernellog[5].endswith(']'):
                parsed['unix_time'] = GrammarUtility.get_unix_timestamp(parsed_kernellog[5], parsed_kernellog[6])
                parsed['message'] = parsed_kernellog[7]

            # no timestamp, no subservice, just message
            else:
                parsed['message'] = ' '.join(parsed_kernellog[5:])

        else:
            if parsed_kernellog[5] == ':':
                parsed['service'] = parsed['service'] + ' ' + parsed_kernellog[5]
                parsed['unix_time'] = GrammarUtility.get_unix_timestamp(parsed_kernellog[6], parsed_kernellog[7])

                # subservice one word
                if parsed_kernellog[8].endswith(':') and not parsed_kernellog[9].endswith(':'):
                    parsed['subservice'] = parsed_kernellog[8]
                    parsed['message'] = ' '.join(parsed_kernellog[9:])

                # subservice two words
                elif not parsed_kernellog[8].endswith(':') and parsed_kernellog[9].endswith(':'):
                    parsed['subservice'] = parsed_kernellog[8] + ' ' + parsed_kernellog[9]
                    parsed['message'] = ' '.join(parsed_kernellog[10:])

                # subservice two words
                elif parsed_kernellog[8].endswith(':') and parsed_kernellog[9].endswith(':'):
                    parsed['subservice'] = parsed_kernellog[8] + ' ' + parsed_kernellog[9]
                    parsed['message'] = ' '.join(parsed_kernellog[10:])

                # no subservice, only message
                else:
                    parsed['subservice'] = ''
                    parsed['message'] = ' '.join(parsed_kernellog[8:])

            else:
                parsed['unix_time'] = GrammarUtility.get_unix_timestamp(parsed_kernellog[5], parsed_kernellog[6])

                # subservice one word
                if parsed_kernellog[7].endswith(':') and not parsed_kernellog[8].endswith(':'):
                    parsed['subservice'] = parsed_kernellog[7]
                    parsed['message'] = ' '.join(parsed_kernellog[8:])

                # subservice two words
                elif not parsed_kernellog[7].endswith(':') and parsed_kernellog[8].endswith(':'):
                    parsed['subservice'] = parsed_kernellog[7] + ' ' + parsed_kernellog[8]
                    parsed['message'] = ' '.join(parsed_kernellog[9:])

                # subservice two words
                elif parsed_kernellog[7].endswith(':') and parsed_kernellog[8].endswith(':'):
                    parsed['subservice'] = parsed_kernellog[7] + ' ' + parsed_kernellog[8]
                    parsed['message'] = ' '.join(parsed_kernellog[9:])

                # no subservice, only message
                else:
                    parsed['subservice'] = ''
                    parsed['message'] = ' '.join(parsed_kernellog[7:])

            if not parsed['subservice'].endswith(':'):
                parsed['message'] = parsed['subservice'] + ' ' + parsed['message']
                parsed['subservice'] = ''

        return parsed


class Main(object):
    def __init__(self, datasets):
        self.datasets = datasets
        self.dataset_path = '/home/hudan/Git/prlogparser/datasets/'
        self.groups = {
            'group1': ['casper-rw', 'dfrws-2009-jhuisi', 'dfrws-2009-nssal', 'honeynet-challenge7'],
            'group2': ['honeynet-challenge5']
        }

    def run(self):
        # parse kernel.log
        for group_name, group in self.groups.items():
            # setup test csv file to save results
            base_name = '/home/hudan/Git/prlogparser/groundtruth/kernel-'
            test_file = base_name + group_name + '.csv'
            f = open(test_file, 'w', newline='')
            writer = csv.writer(f)

            for dataset in group:
                # get grammar
                dl = None
                if group_name == 'group1':
                    dl = KernelLog1(dataset)

                elif group_name == 'group2':
                    dl = KernelLog2(dataset)

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
    datasets_files = {
        'casper-rw': ['casper-rw/kern.log'],
        'dfrws-2009-jhuisi': [
            'dfrws-2009-jhuisi/kern.log',
            'dfrws-2009-jhuisi/kern.log.0',
            'dfrws-2009-jhuisi/kern.log.1',
        ],
        'dfrws-2009-nssal': [
            'dfrws-2009-nssal/kern.log',
            'dfrws-2009-nssal/kern.log.0',
            'dfrws-2009-nssal/kern.log.1',
            'dfrws-2009-nssal/kern.log.2',
            'dfrws-2009-nssal/kern.log.3'
        ],
        'honeynet-challenge5': ['honeynet-challenge5/kern.log'],
        'honeynet-challenge7': ['honeynet-challenge7/kern.log']
    }

    main = Main(datasets_files)
    main.run()
