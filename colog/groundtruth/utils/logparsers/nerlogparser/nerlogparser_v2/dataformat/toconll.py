from nltk import pos_tag, conlltags2tree


class ToConll(object):
    def __init__(self):
        self.entity_names = {
            'timestamp': 'I-TIM',
            'timestamp2': 'B-TIM',
            'sequence_number': 'I-SEQ',
            'level': 'I-LEV',
            'hostname': 'I-HOS',
            'service2': 'B-SER',
            'service': 'I-SER',
            'subservice2': 'B-SUB',
            'subservice': 'I-SUB',
            'unix_time2': 'B-UTIM',
            'unix_time': 'I-UTIM',
            'message': 'O',
            'sock': 'I-SOC',
            'number': 'I-NUM',
            'core1': 'I-COR',
            'core2': 'I-COR',
            'timestamp_bgl': 'I-TIM',
            'source': 'I-SOU',
            'arch': 'I-ARC',
            'domain_or_ip': 'I-DOM',
            'domain_or_ip2': 'B-DOM',
            'status': 'I-STA',
            'status2': 'B-STA',
            'ip_address': 'I-IPA',
            'dash': 'I-DAS',
            'auth': 'I-AUT',
            'command': 'I-COM',
            'command2': 'B-COM',
            'status_code': 'I-STC',
            'num_bytes': 'I-BYT',
            'referrer': 'I-REF',
            'referrer2': 'B-REF',
            'client_agent': 'I-CLI',
            'client_agent2': 'B-CLI',
            'job': 'I-JOB',
            'job2': 'B-JOB'
        }
        self.classes = ['I-TIM', 'B-TIM', 'I-SEQ', 'B-SEQ' 'I-LEV', 'B-LEV' 'I-HOS', 'B-HOS', 'I-SER', 'B-SER',
                        'B-SUB', 'I-SUB', 'B-UTIM', 'I-UTIM', 'O', 'B-SOC', 'I-SOC', 'B-NUM', 'I-NUM', 'I-COR', 'B-COR',
                        'B-SOU', 'I-SOU', 'B-ARC', 'I-ARC', 'I-DOM', 'B-DOM', 'I-STA', 'B-STA', 'B-IPA', 'I-IPA',
                        'I-DAS', 'B-DAS', 'B-AUT', 'I-AUT', 'B-COM', 'I-COM', 'B-STC', 'I-STC', 'B-BYT', 'I-BYT',
                        'I-REF', 'B-REF', 'I-CLI', 'B-CLI', 'I-JOB', 'B-JOB']

    def __get_conll_format(self, value_split, value_split_len, entity, stanford=False):
        if stanford is True:
            underscore = ' '
        else:
            underscore = ' _ _ '

        conll_format = ''
        if entity != 'message':
            if value_split_len == 1:
                conll_format += value_split[0] + underscore + self.entity_names[entity] + '\n'
            else:
                for index, value_name in enumerate(value_split):
                    if index == 0:
                        conll_format += value_name + underscore + self.entity_names[entity + '2'] + '\n'
                    else:
                        conll_format += value_name + underscore + self.entity_names[entity] + '\n'
        else:
            for value_name in value_split:
                conll_format += value_name + underscore + self.entity_names[entity] + '\n'

        return conll_format

    def __get_conll_pos(self, value_pos, value_split_len, entity):
        conll_format = ''
        if entity != 'message':
            if value_split_len == 1:
                conll_format += value_pos[0][0] + ' ' + value_pos[0][1] + ' ' + self.entity_names[entity] + '\n'
            else:
                for index, value_name in enumerate(value_pos):
                    if index == 0:
                        conll_format += value_name[0] + ' ' + value_name[1] + ' ' + \
                                        self.entity_names[entity + '2'] + '\n'
                    else:
                        conll_format += value_name[0] + ' ' + value_name[1] + ' ' + self.entity_names[entity] + '\n'
        else:
            for value_name in value_pos:
                conll_format += value_name[0] + ' ' + value_name[1] + ' ' + self.entity_names[entity] + '\n'

        return conll_format

    def __get_csv(self, value_pos, value_split_len, entity):
        csv_string = ''
        if entity != 'message':
            if value_split_len == 1:
                csv_string += '\t' + value_pos[0][0] + '\t' + value_pos[0][1] + '\t' + self.entity_names[entity] + '\n'
            else:
                for index, value_name in enumerate(value_pos):
                    if index == 0:
                        csv_string += '\t' + value_name[0] + '\t' + value_name[1] + '\t' + \
                                      self.entity_names[entity + '2'] + '\n'
                    else:
                        csv_string += '\t' + value_name[0] + '\t' + value_name[1] + '\t' + \
                                      self.entity_names[entity] + '\n'
        else:
            for value_name in value_pos:
                csv_string += '\t' + value_name[0] + '\t' + value_name[1] + '\t' + self.entity_names[entity] + '\n'

        return csv_string

    def __get_nltk_tree(self, value_pos, value_split_len, entity):
        iob_list = []
        if entity != 'message':
            if value_split_len == 1:
                iob_tuple = (value_pos[0][0], value_pos[0][1], self.entity_names[entity])
                iob_list.append(iob_tuple)
            else:
                for index, value_name in enumerate(value_pos):
                    if index == 0:
                        iob_tuple = (value_name[0], value_name[1], self.entity_names[entity + '2'])
                        iob_list.append(iob_tuple)
                    else:
                        iob_tuple = (value_name[0], value_name[1], self.entity_names[entity])
                        iob_list.append(iob_tuple)
        else:
            for value_name in value_pos:
                iob_tuple = (value_name[0], value_name[1], self.entity_names[entity])
                iob_list.append(iob_tuple)

        return iob_list

    def convert(self, parsed, stanford=False, ispos=False, csv=False, csv_line_id=0, iobtree=False):
        if csv:
            conll_format = 'Sentence: ' + str(csv_line_id)
        elif iobtree:
            conll_format = None
        else:
            conll_format = ''

        conll = []
        for entity, value in parsed.items():
            value_split = value.split()
            value_split_len = len(value_split)

            if value != '':
                if ispos:
                    value_pos = pos_tag(value_split)  # pos = part of speech
                    conll_format += self.__get_conll_pos(value_pos, value_split_len, entity)
                elif csv:
                    value_pos = pos_tag(value_split)
                    conll_format += self.__get_csv(value_pos, value_split_len, entity)
                elif iobtree:
                    value_pos = pos_tag(value_split)
                    iob = self.__get_nltk_tree(value_pos, value_split_len, entity)
                    conll = conll + iob
                else:
                    conll_format += self.__get_conll_format(value_split, value_split_len, entity, stanford)

        if csv is False and iobtree is False:
            conll_format += '\n'

        if iobtree:
            conll_format = conlltags2tree(conll)

        return conll_format
