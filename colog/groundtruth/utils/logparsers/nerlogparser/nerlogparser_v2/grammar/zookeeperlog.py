import os
import csv
from pyparsing import Word, Combine, nums, alphas, Regex, Optional
from collections import OrderedDict


class ZookeeperLog(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.zookeeperlog_grammar = self.__get_zookeeperlog_grammar()

    @staticmethod
    def __get_zookeeperlog_grammar():
        ints = Word(nums)

        date = Combine(ints + '-' + ints + '-' + ints)
        time = Combine(ints + ':' + ints + ':' + ints + ',' + ints)
        timestamp = date + time

        dash = Word('-')
        status = Word(alphas)
        job = Word(alphas + nums + '[]:@=/.$()-') + Optional(Word(alphas + nums + '[]:@=/.$()-')) + Optional(Word('-'))
        message = Regex('.*')

        zookeperlog_grammar = timestamp('timestamp') + dash('dash') + status('status') + job('job') + message('message')
        return zookeperlog_grammar

    def parse_log(self, log_line):
        parsed_zookeeperlog = self.zookeeperlog_grammar.parseString(log_line)

        parsed = OrderedDict()
        parsed['timestamp'] = ' '.join(parsed_zookeeperlog.timestamp)
        parsed['dash'] = parsed_zookeeperlog.dash
        parsed['status'] = parsed_zookeeperlog.status
        parsed['job'] = ' '.join(parsed_zookeeperlog.job)
        parsed['message'] = parsed_zookeeperlog.message

        return parsed


if __name__ == '__main__':
    dataset_path = '/home/hudan/Git/prlogparser/datasets/zookeeper/'
    filenames = ['zookeeper.log']

    test_file = '/home/hudan/Git/prlogparser/groundtruth/test-results/zookeeper-test.csv'
    f = open(test_file, 'w', newline='')
    writer = csv.writer(f)

    zl = ZookeeperLog('')
    for filename in filenames:
        filename = os.path.join(dataset_path, filename)
        with open(filename, 'r') as f:
            for line in f:
                parsed_line = zl.parse_log(line)
                print(parsed_line)

                row = list(parsed_line.values())
                writer.writerow(row)

    f.close()
