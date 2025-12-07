import os
import csv
from pyparsing import Word, alphas, Combine, nums, string, Optional, Regex
from collections import OrderedDict


class DaemonLog(object):
    def __init__(self, dataset):
        """Constructor for class DaemonLog.

        Parameters
        ----------
        dataset : str
            Dataset name.
        """
        self.dataset = dataset
        self.daemonlog_grammar = self.__get_daemonlog_grammar()

    @staticmethod
    def __get_daemonlog_grammar():
        """The definition of daemon.log grammar. Supported dataset:
        casper-rw
        dfrws-2009
        honeynet-challenge5
        honeynet-challenge7

        Returns
        -------
        daemonlog_grammar :
            Grammar for daemon.log
        """
        ints = Word(nums)

        # timestamp
        month = Word(string.ascii_uppercase, string.ascii_lowercase, exact=3)
        day = ints
        hour = Combine(ints + ':' + ints + ':' + ints)
        timestamp = month + day + hour

        # hostname, service name, message
        hostname = Word(alphas + nums + '_' + '-' + '.')
        service = Word(alphas + nums + '/' + '-' + '_' + '.' + '[' + ']' + '(' + ')' + ':')
        subservice = Optional(Word(alphas + nums + ':' + '-' + '_' + '<' + '>' + '.' + "'" + ','))
        subservice_two_words = Optional(Word(alphas + nums + ':' + '-' + '_' + '(' + ')' + '.' + ','))
        message = Regex('.*')

        # daemon log grammar
        daemon_grammar = timestamp + hostname + service + subservice + subservice_two_words + message
        return daemon_grammar

    def parse_log(self, log_line):
        """Parse auth.log based on defined grammar.

        Parameters)
        ----------
        log_line    : str
            A log line to be parsed.

        Returns
        -------
        parsed  : dict[str, str]
            A parsed auth.log containing these elements: timestamp, hostname, service, subservice and message.
        """
        parsed_daemonlog = self.daemonlog_grammar.parseString(log_line)

        # remove empty string
        parsed_daemonlog = list(filter(None, parsed_daemonlog))

        # get parsed auth.log
        parsed = OrderedDict()
        parsed['timestamp'] = parsed_daemonlog[0] + ' ' + parsed_daemonlog[1] + ' ' + parsed_daemonlog[2]
        parsed['hostname'] = parsed_daemonlog[3]
        parsed['service'] = parsed_daemonlog[4]

        if len(parsed_daemonlog) == 5:
            parsed['subservice'] = ''
            parsed['message'] = ''

        elif len(parsed_daemonlog) == 6:
            parsed['subservice'] = ''
            parsed['message'] = parsed_daemonlog[5]

        else:
            # subservice one word
            if parsed_daemonlog[5].endswith(':'):
                parsed['subservice'] = parsed_daemonlog[5]
                parsed['message'] = ' '.join(parsed_daemonlog[6:])

            # subservice two words
            elif not parsed_daemonlog[5].endswith(':') and parsed_daemonlog[6].endswith(':'):
                parsed['subservice'] = parsed_daemonlog[5] + ' ' + parsed_daemonlog[6]
                parsed['message'] = ' '.join(parsed_daemonlog[7:])

            # subservice two words
            elif parsed_daemonlog[5].endswith('>') and parsed_daemonlog[6].endswith(':'):
                parsed['subservice'] = parsed_daemonlog[5] + ' ' + parsed_daemonlog[6]
                parsed['message'] = ' '.join(parsed_daemonlog[7:])

            # subservice one word
            elif parsed_daemonlog[5].endswith('>') and not parsed_daemonlog[6].endswith(':'):
                parsed['subservice'] = parsed_daemonlog[5]
                parsed['message'] = ' '.join(parsed_daemonlog[6:])

            else:
                parsed['subservice'] = ''
                parsed['message'] = parsed_daemonlog[5] + ' ' + ' '.join(parsed_daemonlog[6:])

            if not parsed['service'].endswith(':'):
                parsed['message'] = parsed['service'] + parsed['message']
                parsed['service'] = ''

        return parsed


if __name__ == '__main__':
    # get daemon.log datasets
    dataset_path = '/home/hudan/Git/prlogparser/datasets/'
    filenames = [
        'casper-rw/daemon.log',
        'dfrws-2009-jhuisi/daemon.log',
        'dfrws-2009-jhuisi/daemon.log.0',
        'dfrws-2009-jhuisi/daemon.log.1',
        'dfrws-2009-nssal/daemon.log',
        'dfrws-2009-nssal/daemon.log.0',
        'dfrws-2009-nssal/daemon.log.1',
        'dfrws-2009-nssal/daemon.log.2',
        'dfrws-2009-nssal/daemon.log.3',
        'honeynet-challenge5/daemon.log',
        'honeynet-challenge7/daemon.log'
    ]

    # setup test csv file to save results
    test_file = '/home/hudan/Git/prlogparser/groundtruth/daemon-test.csv'
    f = open(test_file, 'w', newline='')
    writer = csv.writer(f)

    # parse daemon.log
    dl = DaemonLog('')
    for filename in filenames:
        filename = os.path.join(dataset_path, filename)
        with open(filename, 'r') as f:
            for line in f:
                # get parsed line and print
                parsed_line = dl.parse_log(line)
                print(parsed_line)

                # write to csv
                row = list(parsed_line.values())
                writer.writerow(row)

    f.close()
