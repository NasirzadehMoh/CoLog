# httpServerLogParser.py
#
# Copyright (c) 2016, Paul McGuire
#

import os
import csv
import string
from pyparsing import alphas, nums, dblQuotedString, Combine, Word, Group, delimitedList
from collections import OrderedDict


class WebLog(object):
    """This class is based on httpServerLogParser.py by Paul McGuire.
    http://pyparsing.wikispaces.com/file/detail/httpServerLogParser.py

    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.weblog_grammar = self.__get_weblog_grammar()

    @staticmethod
    def __get_weblog_grammar():
        integer = Word(nums)
        ip_address = delimitedList(integer, ".", combine=True)
        time_zone_offset = Word("+-", nums)
        month = Word(string.ascii_uppercase, string.ascii_lowercase, exact=3)
        server_date_time = Group(Combine("[" + integer + "/" + month + "/" + integer +
                                         ":" + integer + ":" + integer + ":" + integer) +
                                 Combine(time_zone_offset + "]"))

        weblog_grammar = (ip_address.setResultsName("ip_address") +
                          Word("-").setResultsName("dash") +
                          ("-" | Word(alphas + nums + "@._")).setResultsName("auth") +
                          server_date_time.setResultsName("timestamp") +
                          dblQuotedString.setResultsName("command") +
                          (integer | "-").setResultsName("status_code") +
                          (integer | "-").setResultsName("num_bytes") +
                          dblQuotedString.setResultsName("referrer") +
                          dblQuotedString.setResultsName("client_agent"))

        return weblog_grammar

    def parse_log(self, log_line):
        parsed_weblog = self.weblog_grammar.parseString(log_line)

        parsed = OrderedDict()
        parsed['ip_address'] = parsed_weblog.ip_address
        parsed['dash'] = parsed_weblog.dash
        parsed['auth'] = parsed_weblog.auth
        parsed['timestamp'] = ' '.join(parsed_weblog.timestamp[0:2])
        parsed['command'] = parsed_weblog.command
        parsed['status_code'] = parsed_weblog.status_code
        parsed['num_bytes'] = parsed_weblog.num_bytes
        parsed['referrer'] = parsed_weblog.referrer
        parsed['client_agent'] = parsed_weblog.client_agent

        return parsed

    @staticmethod
    def __get_filename(base_filename, month, day):
        # example:
        # access.log.2018-01-02
        # access.log.2018-01-01

        # check day format
        if day < 10:
            day = '0' + str(day)
        else:
            day = str(day)

        fn = base_filename + '0' + str(month) + '-' + day
        return fn

    def get_all_filenames(self):
        # setup variables
        base_filename = 'access.log.2018-'
        months = range(3, 6)
        day_odd = range(1, 32)
        day_even = range(1, 31)

        # get all filenames
        filenames = []
        for month in months:
            if month % 2 == 0:
                for day in day_even:
                    fn = self.__get_filename(base_filename, month, day)
                    filenames.append(fn)
            else:
                for day in day_odd:
                    fn = self.__get_filename(base_filename, month, day)
                    filenames.append(fn)

        return filenames


if __name__ == '__main__':
    wl = WebLog('')
    dataset_path = '/home/hudan/Git/prlogparser/datasets/secrepo-accesslog/'
    filenames_list = wl.get_all_filenames()

    test_file = '/home/hudan/Git/prlogparser/groundtruth/test-results/weblog-test.csv'
    f = open(test_file, 'w', newline='')
    writer = csv.writer(f)

    for filename in filenames_list:
        filename = os.path.join(dataset_path, filename)
        with open(filename, 'r') as f:
            for line in f:
                parsed_line = wl.parse_log(line)
                print(parsed_line)

                row = list(parsed_line.values())
                writer.writerow(row)

    f.close()
