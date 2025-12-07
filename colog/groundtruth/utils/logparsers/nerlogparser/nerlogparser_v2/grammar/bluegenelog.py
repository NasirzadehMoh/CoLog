import os
import csv
from pyparsing import Word, alphas, Combine, nums, Regex, ParseException
from collections import OrderedDict


class BlueGeneLog(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.bluegenelog_grammar = self.__get_bluegenelog_grammar()

    @staticmethod
    def __get_bluegenelog_grammar():
        """The definition of BlueGene/L grammar.

        The BlueGene/L logs can be downloaded from [Usenix2006a]_ and
        this data was used in [Stearley2008]_.

        Returns
        -------
        bluegene_grammar    :
            Grammar for BlueGene/L supercomputer logs.

        References
        ----------
        .. [Usenix2006a]  The HPC4 data. URL: https://www.usenix.org/cfdr-data#hpc4
        .. [Stearley2008] Stearley, J., & Oliner, A. J. Bad words: Finding faults in Spirit's syslogs.
                          In 8th IEEE International Symposium on Cluster Computing and the Grid, pp. 765-770.
        """
        ints = Word(nums)

        sock = Word(alphas + '-' + '_')
        number = ints
        date = Combine(ints + '.' + ints + '.' + ints)
        core1 = Word(alphas + nums + '-' + ':' + '_')
        datetime = Combine(ints + '-' + ints + '-' + ints + '-' + ints + '.' + ints + '.' + ints + '.' + ints)
        core2 = Word(alphas + nums + '-' + ':' + '_')
        source = Word(alphas)
        service = Word(alphas)
        info_type = Word(alphas)
        message = Regex('.*')

        # blue gene log grammar
        bluegene_grammar = sock + number + date + core1 + datetime + core2 + source + service + info_type + message
        return bluegene_grammar

    def parse_log(self, log_line):
        """Parse the BlueGene/L logs based on defined grammar.

        Parameters
        ----------
        log_line    : str
            A log line to be parsed

        Returns
        -------
        parsed      : dict[str, str]
            A parsed BlueGene/L log.
        """
        parsed = OrderedDict()
        try:
            parsed_bluegenelog = self.bluegenelog_grammar.parseString(log_line)
            parsed['sock'] = parsed_bluegenelog[0]
            parsed['number'] = parsed_bluegenelog[1]
            parsed['timestamp'] = parsed_bluegenelog[2]
            parsed['core1'] = parsed_bluegenelog[3]
            parsed['timestamp_bgl'] = parsed_bluegenelog[4]
            parsed['core2'] = parsed_bluegenelog[5]
            parsed['source'] = parsed_bluegenelog[6]
            parsed['service'] = parsed_bluegenelog[7]
            parsed['level'] = parsed_bluegenelog[8]
            parsed['message'] = parsed_bluegenelog[9]

        except ParseException:
            print(log_line)

        return parsed


if __name__ == '__main__':
    dataset_path = '/home/hudan/Git/prlogparser/datasets/'
    filenames = ['bgl2/bgl2']

    test_file = '/home/hudan/Git/prlogparser/groundtruth/test-results/bgl-test.csv'
    f = open(test_file, 'w', newline='')
    writer = csv.writer(f)

    bl = BlueGeneLog('')
    for filename in filenames:
        filename = os.path.join(dataset_path, filename)
        with open(filename, 'r') as f:
            for line in f:
                parsed_line = bl.parse_log(line)
                print(parsed_line['timestamp'])

                row = list(parsed_line.values())
                writer.writerow(row)
    f.close()
