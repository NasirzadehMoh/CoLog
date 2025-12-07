import sys
  
# Insert the path of modules folder 
sys.path.insert(0, 'groundtruth/utils/logparsers/nerlogparser/nerlogparser_v2')

import pprint
from optparse import OptionParser
from collections import OrderedDict
from model.ner_model import NERModel
from model.config import Config
from output.to_json import ToJson

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

class Nerlogparser(object):
    def __init__(self):
        self.model = None
        self.config = None
        self.master_label = {}

        self.__load_pretrained_model()
        self.__load_label()

    def __load_pretrained_model(self):
        # create instance of config
        self.config = Config()

        # load pretrained model
        self.model = NERModel(self.config)
        self.model.build()
        self.model.restore_session(self.config.dir_model)

    def __load_label(self):
        # load NER label and its corresponding human-readable field label
        with open(self.config.label_file, 'r') as f:
            label = f.readlines()

        labels = {}
        for line in label:
            line_split = line.split(' ')
            ner_label, final_label = line_split[0], line_split[1]
            labels[ner_label] = final_label.rstrip()

        self.master_label = labels

    def __get_per_entity(self, words_raw, ner_label):
        # one entity can contain one or more words
        entity = OrderedDict()
        for index, label in enumerate(ner_label):
            if '-' in label:
                main_label = label.split('-')[1]
            else:
                main_label = label

            if main_label not in entity.keys():
                entity[main_label] = []

            entity[main_label].append(words_raw[index])

        # one entity is now one sentence
        final_entity = OrderedDict()
        for main_label, words in entity.items():
            final_label = self.master_label[main_label]
            final_entity[final_label] = ' '.join(words)

        if 'message' not in final_entity.keys():
            final_entity['message'] = ''

        return final_entity

    def parse_logs(self, log_file):
        # parse log files using pretrained model
        raw_logs = {}
        parsed_logs = OrderedDict()
        parsed_log_index = 0
        with open(log_file) as f:
            for line_index, line in enumerate(f):
                if line not in ['\n', '\r\n']:
                    raw_logs[parsed_log_index] = line
                    words_raw = line.strip().split()

                    ner_label = self.model.predict(words_raw)
                    parsed = self.__get_per_entity(words_raw, ner_label)
                    parsed_logs[parsed_log_index] = parsed
                    parsed_log_index += 1

        return parsed_logs


def main():
    parser = OptionParser(usage='usage: nerlogparser [options]')
    parser.add_option('-i', '--input',
                      action='store',
                      dest='input_file',
                      help='Input log file.')
    parser.add_option('-o', '--output',
                      action='store',
                      dest='output_file',
                      help='Parsed log file.')

    # get options
    (options, args) = parser.parse_args()
    input_file = options.input_file
    output_file = options.output_file

    if options.input_file:
        # parse log file
        nerlogparser = Nerlogparser()
        parsed_results = nerlogparser.parse_logs(input_file)

        if options.output_file:
            print('Write results to', output_file)
            ToJson.write_to_json(parsed_results, output_file)

        else:
            print('No output file. Print parsing results on terminal.')
            for line_id, parsed in parsed_results.items():
                print('Line:', line_id)
                pprint.pprint(parsed)
                print()

    else:
        print('Please see help: nerlogparser -h')


if __name__ == "__main__":
    main()
