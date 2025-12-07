import os
import csv
from pyparsing import Word, alphas, Combine, nums, string, Optional, Regex
from collections import OrderedDict
from nerlogparser.grammar.grammar_utility import GrammarUtility


class MessagesLog(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.groups = {
            'group1': ['casper-rw', 'dfrws-2009-jhuisi', 'dfrws-2009-nssal', 'honeynet-challenge7'],
            'group2': ['honeynet-challenge5']
        }

    def get_grammar(self):
        dl = None
        if self.dataset in self.groups['group1']:
            dl = MessagesLog1(self.dataset)

        elif self.dataset in self.groups['group2']:
            dl = MessagesLog2(self.dataset)

        return dl


class MessagesLog1(object):
    def __init__(self, dataset):
        """Constructor for class MessagesLog. This parser also supports syslog.

        Parameters
        ----------
        dataset : str
            Dataset name.
        """
        self.dataset = dataset
        self.messageslog_grammar = self.__get_messageslog_grammar()

    @staticmethod
    def __get_messageslog_grammar():
        """The definition of messages log grammar. Supported dataset:
        casper-rw
        dfrws-2009
        honeynet-challenge7

        Returns
        -------
        messageslog_grammar :
            Grammar for messages log
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
                                   '<' + '>' + ','))
        subservice_two_words = Optional(Word(alphas + nums + '_' + '-' + ':' + '[' + ']' + ',' + '.' + '=' + '/' +
                                             '(' + ')' + '*'))
        message = Optional(Regex('.*'))

        # messages log grammar
        messageslog_grammar = timestamp + hostname + service + unix_time + subservice + subservice_two_words + message
        return messageslog_grammar

    def parse_log(self, log_line):
        """Parse messages log based on defined grammar.

        Parameters
        ----------
        log_line    : str
            A log line to be parsed.

        Returns
        -------
        parsed  : dict[str, str]
            A parsed messages log containing these elements: timestamp, hostname, service, unix_time,
            subservice and message.
        """
        parsed_messageslog = self.messageslog_grammar.parseString(log_line)

        # get parsed kernel log
        parsed = OrderedDict()
        parsed['timestamp'] = parsed_messageslog[0] + ' ' + parsed_messageslog[1] + ' ' + parsed_messageslog[2]
        parsed['hostname'] = parsed_messageslog[3]
        parsed['service'] = parsed_messageslog[4]

        if len(parsed_messageslog) == 6:
            parsed['unix_time'] = ''
            parsed['subservice'] = ''
            if not parsed['service'].endswith(':'):
                parsed['message'] = parsed['service'] + ' ' + parsed_messageslog[5]
                parsed['service'] = ''
            else:
                parsed['message'] = ''

        elif len(parsed_messageslog) == 7:
            parsed['unix_time'] = ''
            parsed['subservice'] = ''
            if not parsed_messageslog[5].endswith(':'):
                parsed['message'] = parsed_messageslog[5] + ' ' + parsed_messageslog[6]

            if not parsed['service'].endswith(':'):
                parsed['message'] = parsed['service'] + ' ' + parsed_messageslog[5] + ' ' + parsed_messageslog[6]
                parsed['service'] = ''

        elif len(parsed_messageslog) == 8:
            parsed['unix_time'] = ''
            parsed['subservice'] = ''

            # no message
            if parsed_messageslog[7] == '' and parsed_messageslog[5].startswith('[') and \
                    parsed_messageslog[6].endswith(']'):
                parsed['unix_time'] = GrammarUtility.get_unix_timestamp(parsed_messageslog[5], parsed_messageslog[6])
                parsed['message'] = ''

            # message exists
            elif parsed_messageslog[7] != '' and parsed_messageslog[5].startswith('[') and \
                    parsed_messageslog[6].endswith(']'):
                parsed['unix_time'] = GrammarUtility.get_unix_timestamp(parsed_messageslog[5], parsed_messageslog[6])
                parsed['message'] = parsed_messageslog[7]

            # subservice one word
            elif parsed_messageslog[5].endswith(':') and not parsed_messageslog[6].endswith(':'):
                parsed['subservice'] = parsed_messageslog[5]
                parsed['message'] = ' '.join(parsed_messageslog[6:])

            # subservice two words
            elif not parsed_messageslog[5].endswith(':') and parsed_messageslog[6].endswith(':'):
                parsed['subservice'] = parsed_messageslog[5] + ' ' + parsed_messageslog[6]
                parsed['message'] = parsed_messageslog[7]

            # subservice two words
            elif parsed_messageslog[5].endswith(':') and parsed_messageslog[6].endswith(':'):
                parsed['subservice'] = parsed_messageslog[5] + ' ' + parsed_messageslog[6]
                parsed['message'] = parsed_messageslog[7]

            # no timestamp, no subservice, just message
            else:
                parsed['message'] = ' '.join(parsed_messageslog[5:])

            if not parsed['subservice'].endswith(':'):
                parsed['message'] = parsed['subservice'] + ' ' + parsed['message']
                parsed['subservice'] = ''

            if not parsed['service'].endswith(':'):
                parsed['message'] = parsed['service'] + ' ' + parsed['message']
                parsed['service'] = ''

        else:
            parsed['unix_time'] = GrammarUtility.get_unix_timestamp(parsed_messageslog[5], parsed_messageslog[6])

            # subservice one word
            if parsed_messageslog[7].endswith(':') and not parsed_messageslog[8].endswith(':'):
                parsed['subservice'] = parsed_messageslog[7]
                parsed['message'] = ' '.join(parsed_messageslog[8:])

            # subservice two words
            elif not parsed_messageslog[7].endswith(':') and parsed_messageslog[8].endswith(':'):
                parsed['subservice'] = parsed_messageslog[7] + ' ' + parsed_messageslog[8]
                parsed['message'] = ' '.join(parsed_messageslog[9:])

            # subservice two words
            elif parsed_messageslog[7].endswith(':') and parsed_messageslog[8].endswith(':'):
                parsed['subservice'] = parsed_messageslog[7] + ' ' + parsed_messageslog[8]
                parsed['message'] = ' '.join(parsed_messageslog[9:])

            else:
                parsed['subservice'] = ''
                parsed['message'] = ' '.join(parsed_messageslog[7:])

            if not parsed['subservice'].endswith(':'):
                parsed['message'] = parsed['subservice'] + ' ' + parsed['message']
                parsed['subservice'] = ''

        return parsed


class MessagesLog2(object):
    def __init__(self, dataset):
        """Constructor for class MessagesLog. This parser also supports syslog.

        Parameters
        ----------
        dataset : str
            Dataset name.
        """
        self.dataset = dataset
        self.messageslog_grammar = self.__get_messageslog_grammar()

    @staticmethod
    def __get_messageslog_grammar():
        """The definition of messages log grammar. Supported dataset:
        honeynet-challenge5

        Returns
        -------
        messageslog_grammar :
            Grammar for messages log
        """
        ints = Word(nums)

        # timestamp
        month = Word(string.ascii_uppercase, string.ascii_lowercase, exact=3)
        day = ints
        hour = Combine(ints + ':' + ints + ':' + ints)
        timestamp = month + day + hour

        # hostname, service name, message
        hostname = Word(alphas + nums + '_' + '-' + '.')
        service = Word(alphas + nums + '/' + '-' + '_' + '.' + '[' + ']' + ':') + Optional(':')

        # unix time
        unix_time = Optional('[' + Word(nums + '.' + ']'))
        subservice = Optional(Word(alphas + nums + '_' + '-' + ':'))
        subservice_two_words = Optional(Word(alphas + nums + '_' + '-' + ':' + '[' + ']' + ',' + '.' + '=' + '/' +
                                             '(' + ')' + '*'))
        message = Regex('.*')

        # messages log grammar
        messageslog_grammar = timestamp + hostname + service + unix_time + subservice + subservice_two_words + message
        return messageslog_grammar

    def parse_log(self, log_line):
        """Parse messages log based on defined grammar.

        Parameters
        ----------
        log_line    : str
            A log line to be parsed.

        Returns
        -------
        parsed  : dict[str, str]
            A parsed messages log containing these elements: timestamp, hostname, service, unix_time,
            subservice and message.
        """
        parsed_messageslog = self.messageslog_grammar.parseString(log_line)

        # get parsed kernel log
        parsed = OrderedDict()
        parsed['timestamp'] = parsed_messageslog[0] + ' ' + parsed_messageslog[1] + ' ' + parsed_messageslog[2]
        parsed['hostname'] = parsed_messageslog[3]
        parsed['service'] = parsed_messageslog[4]

        if len(parsed_messageslog) == 6:
            parsed['unix_time'] = ''
            parsed['subservice'] = ''
            if not parsed['service'].endswith(':'):
                parsed['message'] = parsed['service'] + ' ' + parsed_messageslog[5]
                parsed['service'] = ''
            else:
                parsed['message'] = ''

        elif len(parsed_messageslog) == 7:
            parsed['unix_time'] = ''
            parsed['subservice'] = ''
            if not parsed_messageslog[5].endswith(':'):
                parsed['message'] = parsed_messageslog[5] + ' ' + parsed_messageslog[6]

            if not parsed['service'].endswith(':'):
                parsed['message'] = parsed['service'] + ' ' + parsed_messageslog[5] + ' ' + parsed_messageslog[6]
                parsed['service'] = ''

        elif len(parsed_messageslog) == 8:
            parsed['unix_time'] = ''
            parsed['subservice'] = ''

            # timestamp exists
            if parsed_messageslog[5].startswith('[') and parsed_messageslog[5].endswith(']'):
                parsed['unix_time'] = GrammarUtility.get_unix_timestamp(parsed_messageslog[5], parsed_messageslog[6])
                parsed['message'] = parsed_messageslog[7]

            # no timestamp, no subservice, just message
            else:
                parsed['message'] = ' '.join(parsed_messageslog[5:])

        else:
            if parsed_messageslog[5] == ':':
                parsed['service'] = parsed['service'] + ' ' + parsed_messageslog[5]
                parsed['unix_time'] = GrammarUtility.get_unix_timestamp(parsed_messageslog[6], parsed_messageslog[7])

                # subservice one word
                if parsed_messageslog[8].endswith(':') and not parsed_messageslog[9].endswith(':'):
                    parsed['subservice'] = parsed_messageslog[8]
                    parsed['message'] = ' '.join(parsed_messageslog[9:])

                # subservice two words
                elif not parsed_messageslog[8].endswith(':') and parsed_messageslog[9].endswith(':'):
                    parsed['subservice'] = parsed_messageslog[8] + ' ' + parsed_messageslog[9]
                    parsed['message'] = ' '.join(parsed_messageslog[10:])

                # subservice two words
                elif parsed_messageslog[8].endswith(':') and parsed_messageslog[9].endswith(':'):
                    parsed['subservice'] = parsed_messageslog[8] + ' ' + parsed_messageslog[9]
                    parsed['message'] = ' '.join(parsed_messageslog[10:])

                # no subservice, only message
                else:
                    parsed['subservice'] = ''
                    parsed['message'] = ' '.join(parsed_messageslog[8:])

            else:
                parsed['unix_time'] = GrammarUtility.get_unix_timestamp(parsed_messageslog[5], parsed_messageslog[6])

                # subservice one word
                if parsed_messageslog[7].endswith(':') and not parsed_messageslog[8].endswith(':'):
                    parsed['subservice'] = parsed_messageslog[7]
                    parsed['message'] = ' '.join(parsed_messageslog[8:])

                # subservice two words
                elif not parsed_messageslog[7].endswith(':') and parsed_messageslog[8].endswith(':'):
                    parsed['subservice'] = parsed_messageslog[7] + ' ' + parsed_messageslog[8]
                    parsed['message'] = ' '.join(parsed_messageslog[9:])

                # subservice two words
                elif parsed_messageslog[7].endswith(':') and parsed_messageslog[8].endswith(':'):
                    parsed['subservice'] = parsed_messageslog[7] + ' ' + parsed_messageslog[8]
                    parsed['message'] = ' '.join(parsed_messageslog[9:])

                # no subservice, only message
                else:
                    parsed['subservice'] = ''
                    parsed['message'] = ' '.join(parsed_messageslog[7:])

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
        # parse messages.log
        for group_name, group in self.groups.items():
            # setup test csv file to save results
            base_name = '/home/hudan/Git/prlogparser/groundtruth/messages-'
            test_file = base_name + group_name + '.csv'
            f = open(test_file, 'w', newline='')
            writer = csv.writer(f)

            for dataset in group:
                # get grammar
                dl = None
                if group_name == 'group1':
                    dl = MessagesLog1(dataset)

                elif group_name == 'group2':
                    dl = MessagesLog2(dataset)

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
        'casper-rw': [
            'casper-rw/messages',
            'casper-rw/syslog',
            'casper-rw/syslog.0',
            'casper-rw/syslog.1',
            'casper-rw/syslog.2',
            'casper-rw/syslog.3'
        ],
        'dfrws-2009-jhuisi': [
            'dfrws-2009-jhuisi/messages',
            'dfrws-2009-jhuisi/messages.0',
            'dfrws-2009-jhuisi/messages.1',
            'dfrws-2009-jhuisi/syslog',
            'dfrws-2009-jhuisi/syslog.0',
            'dfrws-2009-jhuisi/syslog.1',
            'dfrws-2009-jhuisi/syslog.2'
        ],
        'dfrws-2009-nssal': [
            'dfrws-2009-nssal/messages',
            'dfrws-2009-nssal/messages.0',
            'dfrws-2009-nssal/messages.1',
            'dfrws-2009-nssal/messages.2',
            'dfrws-2009-nssal/messages.3',
            'dfrws-2009-nssal/syslog',
            'dfrws-2009-nssal/syslog.0',
            'dfrws-2009-nssal/syslog.1',
            'dfrws-2009-nssal/syslog.2',
            'dfrws-2009-nssal/syslog.3',
            'dfrws-2009-nssal/syslog.4',
            'dfrws-2009-nssal/syslog.5',
            'dfrws-2009-nssal/syslog.6'
        ],
        'honeynet-challenge5': ['honeynet-challenge5/messages'],
        'honeynet-challenge7': [
            'honeynet-challenge7/messages',
            'honeynet-challenge7/syslog'
        ]
    }

    main = Main(datasets_files)
    main.run()
