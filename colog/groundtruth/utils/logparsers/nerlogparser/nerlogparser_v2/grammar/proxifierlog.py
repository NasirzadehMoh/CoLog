import os
import csv
from pyparsing import Word, Combine, nums, alphas, Optional, Regex
from collections import OrderedDict


class ProxifierLog(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.proxifierlog_grammar = self.__get_proxifierlog_grammar()

    @staticmethod
    def __get_proxifierlog_grammar():
        # get proxifier grammar
        ints = Word(nums)

        date = Combine('[' + ints + '.' + ints)
        time = Combine(ints + ':' + ints + ':' + ints + ']')
        timestamp = date + time

        service = Word(alphas + nums + '.' + '-' + '_')
        arch = Optional(Word('*' + nums))
        domain_or_ip = Optional(Word('-')) + Word(alphas + nums + '.' + ':' + '-')
        status = Optional(Word(alphas + ',')) + Optional(':')
        message = Regex('.*')

        proxifierlog_grammar = timestamp + service + arch + domain_or_ip + status + message
        return proxifierlog_grammar

    def parse_log(self, log_line):
        # parse proxifier log entries
        parsed_proxifierlog = self.proxifierlog_grammar.parseString(log_line)

        parsed = OrderedDict()
        parsed['timestamp'] = parsed_proxifierlog[0] + ' ' + parsed_proxifierlog[1]
        parsed['service'] = parsed_proxifierlog[2]

        if len(parsed_proxifierlog) == 6:
            parsed['service'] = parsed_proxifierlog[2] + ' ' + parsed_proxifierlog[3]
            parsed['arch'] = ''
            parsed['domain_or_ip'] = ''
            parsed['status'] = ''
            parsed['message'] = ' '.join(parsed_proxifierlog[4:])

        elif len(parsed_proxifierlog) == 7:
            parsed['arch'] = ''
            parsed['domain_or_ip'] = parsed_proxifierlog[3] + ' ' + parsed_proxifierlog[4]

            if parsed_proxifierlog[5].endswith(','):
                parsed['status'] = parsed_proxifierlog[5]
                parsed['message'] = parsed_proxifierlog[6]
            else:
                parsed['status'] = ''
                parsed['message'] = ' '.join(parsed_proxifierlog[5:])

        elif len(parsed_proxifierlog) == 8:
            if parsed_proxifierlog[3].startswith('*'):
                parsed['arch'] = parsed_proxifierlog[3]
                parsed['domain_or_ip'] = parsed_proxifierlog[4] + ' ' + parsed_proxifierlog[5]

                if parsed_proxifierlog[6].endswith(','):
                    parsed['status'] = parsed_proxifierlog[6]
                    parsed['message'] = parsed_proxifierlog[7]
                else:
                    parsed['status'] = ''
                    parsed['message'] = ' '.join(parsed_proxifierlog[6:])

            else:
                parsed['arch'] = ''
                parsed['domain_or_ip'] = parsed_proxifierlog[3] + ' ' + parsed_proxifierlog[4]

                if parsed_proxifierlog[6] == ':':
                    parsed['status'] = parsed_proxifierlog[5] + ' ' + parsed_proxifierlog[6]
                    parsed['message'] = parsed_proxifierlog[7]
                else:
                    parsed['status'] = ''
                    parsed['message'] = ' '.join(parsed_proxifierlog[5:])

        elif len(parsed_proxifierlog) == 9:
            parsed['arch'] = parsed_proxifierlog[3]
            parsed['domain_or_ip'] = parsed_proxifierlog[4] + ' ' + parsed_proxifierlog[5]
            parsed['status'] = parsed_proxifierlog[6] + ' ' + parsed_proxifierlog[7]
            parsed['message'] = parsed_proxifierlog[8]

        return parsed


if __name__ == '__main__':
    dataset_path = '/home/hudan/Git/prlogparser/datasets/proxifier/'
    filenames = ['proxifier.log']

    test_file = '/home/hudan/Git/prlogparser/groundtruth/test-results/proxifier-test.csv'
    f = open(test_file, 'w', newline='')
    writer = csv.writer(f)

    pl = ProxifierLog('')
    for filename in filenames:
        filename = os.path.join(dataset_path, filename)
        with open(filename, 'r') as f:
            for line in f:
                parsed_line = pl.parse_log(line)
                print(parsed_line)

                row = list(parsed_line.values())
                writer.writerow(row)

    f.close()
