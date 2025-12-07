from collections import OrderedDict
import csv


class CSVLog(object):
    def __init__(self, dataset):
        self.dataset = dataset
        if self.dataset == 'dfrws-2016':
            self.log_file = '/home/hudan/Git/prlogparser/datasets/dfrws-2016/csv.csv'

    def parse_log(self):
        parsed_logs = []
        with open(self.log_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                parsed = OrderedDict()
                parsed['timestamp'] = row[0]
                parsed['sequence_number'] = row[1]
                parsed['service'] = row[2]
                parsed['level'] = row[3]
                parsed['message'] = row[4]
                parsed_logs.append(parsed)

        return parsed_logs


if __name__ == '__main__':
    csvlog = CSVLog('dfrws-2016')
    results = csvlog.parse_log()
    for result in results:
        print(result)
