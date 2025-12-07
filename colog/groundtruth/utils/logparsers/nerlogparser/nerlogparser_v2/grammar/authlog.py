import os
import csv
from pyparsing import Word, alphas, Combine, nums, string, Optional, Regex
from collections import OrderedDict


class AuthLog(object):
    def __init__(self, dataset):
        """Constructor for class AuthLog.

        Parameters
        ----------
        dataset : str
            Dataset name.
        """
        self.dataset = dataset
        self.authlog_grammar = self.__get_authlog_grammar()

    @staticmethod
    def __get_authlog_grammar():
        """The definition of auth.log grammar. Supported dataset:
        casper-rw
        dfrws-2009
        honeynet-challenge5
        honeynet-challenge7

        Returns
        -------
        authlog_grammar :
            Grammar for auth.log
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
        subservice = Optional(Word(alphas + ':' + '-' + '_' + '(' + ')'))
        subservice_two_words = Optional(Word(alphas + ':' + '-' + '_' + '(' + ')')) + \
            Optional(Word(alphas + ':' + '-' + '_' + '(' + ')'))
        message = Regex('.*')

        # auth log grammar
        authlog_grammar = timestamp + hostname + service + subservice + subservice_two_words + message
        return authlog_grammar

    def parse_log(self, log_line):
        """Parse auth.log based on defined grammar.

        Parameters
        ----------
        log_line    : str
            A log line to be parsed.

        Returns
        -------
        parsed  : dict[str, str]
            A parsed auth.log containing these elements: timestamp, hostname, service, pid, subservice and message.
        """
        parsed_authlog = self.authlog_grammar.parseString(log_line)

        # get parsed auth.log
        parsed = OrderedDict()
        parsed['timestamp'] = parsed_authlog[0] + ' ' + parsed_authlog[1] + ' ' + parsed_authlog[2]
        parsed['hostname'] = parsed_authlog[3]
        parsed['service'] = parsed_authlog[4]

        if len(parsed_authlog) == 6:
            parsed['subservice'] = ''
            parsed['message'] = parsed_authlog[5]

        elif len(parsed_authlog) == 8:
            if not parsed_authlog[6].endswith(':'):
                parsed['subservice'] = ''
                parsed['message'] = ' '.join(parsed_authlog[5:])

            if parsed_authlog[5].endswith(':'):
                parsed['subservice'] = parsed_authlog[5]
                parsed['message'] = ' '.join(parsed_authlog[6:])

        else:
            if parsed_authlog[6].endswith(':'):
                parsed['subservice'] = parsed_authlog[5] + ' ' + parsed_authlog[6]
                parsed['message'] = ' '.join(parsed_authlog[7:])

            else:
                parsed['subservice'] = parsed_authlog[5]
                parsed['message'] = ' '.join(parsed_authlog[6:])

            if not parsed['subservice'].endswith(':'):
                parsed['message'] = parsed['subservice'] + ' ' + parsed['message']
                parsed['subservice'] = ''

        return parsed


if __name__ == '__main__':
    # get auth.log datasets
    dataset_path = '/home/hudan/Git/prlogparser/datasets/'
    filenames = [
        'casper-rw/auth.log',
        'dfrws-2009-jhuisi/auth.log',
        'dfrws-2009-jhuisi/auth.log.0',
        'dfrws-2009-jhuisi/auth.log.1',
        'dfrws-2009-nssal/auth.log',
        'dfrws-2009-nssal/auth.log.0',
        'dfrws-2009-nssal/auth.log.1',
        'dfrws-2009-nssal/auth.log.2',
        'dfrws-2009-nssal/auth.log.3',
        'dfrws-2009-nssal/auth.log.4',
        'honeynet-challenge5/auth.log',
        'honeynet-challenge7/auth.log'
    ]

    # setup test csv file to save results
    test_file = '/home/hudan/Git/prlogparser/groundtruth/auth-test.csv'
    f = open(test_file, 'w', newline='')
    writer = csv.writer(f)

    # parse auth.log
    dl = AuthLog('')
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
