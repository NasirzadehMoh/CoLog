import os
import csv
from pyparsing import Word, Combine, nums, Regex, nestedExpr
from collections import OrderedDict


class KippoLog(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.kippolog_grammar = self.__get_kippolog_grammar()

    @staticmethod
    def __get_kippolog_grammar():
        """The definition of Kippo honeypot log grammar.

        Returns
        -------
        kippolog_grammar    :
            Grammar for Kippo log.
        """
        ints = Word(nums)

        date = Combine(ints + '-' + ints + '-' + ints)
        time = Combine(ints + ':' + ints + ':' + ints + '+0000')
        timestamp = date + time
        service = nestedExpr(opener='[', closer=']')
        message = Regex(".*")

        # kippo honeypot log grammar
        kippolog_grammar = timestamp('timestamp') + service('service') + message('message')
        return kippolog_grammar

    def parse_log(self, log_line):
        parsed_kippolog = self.kippolog_grammar.parseString(log_line)

        parsed = OrderedDict()
        parsed['timestamp'] = ' '.join(parsed_kippolog.timestamp)
        if len(parsed_kippolog.service[0]) > 1:
            parsed['service'] = '[' + ' '.join(parsed_kippolog.service[0]) + ']'
        else:
            parsed['service'] = '[' + parsed_kippolog.service[0][0] + ']'
        parsed['message'] = parsed_kippolog.message

        return parsed


if __name__ == '__main__':
    dataset_path = '/home/hudan/Git/prlogparser/datasets/kippo/'
    filenames = [
        'kippo.2017-02-14.log',
        'kippo.2017-02-15.log',
        'kippo.2017-02-16.log',
        'kippo.2017-02-17.log',
        'kippo.2017-02-18.log',
        'kippo.2017-02-19.log',
        'kippo.2017-02-20.log'
    ]

    test_file = '/home/hudan/Git/prlogparser/groundtruth/test-results/kippo-test.csv'
    f = open(test_file, 'w', newline='')
    writer = csv.writer(f)

    kl = KippoLog('')
    for filename in filenames:
        filename = os.path.join(dataset_path, filename)
        with open(filename, 'r') as f:
            for line in f:
                parsed_line = kl.parse_log(line)
                print(parsed_line)

                row = list(parsed_line.values())
                writer.writerow(row)

    f.close()
