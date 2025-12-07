

class GrammarUtility(object):
    @staticmethod
    def get_unix_timestamp(square_bracket, timestamp):
        main_digit = timestamp.split('.')[0]
        space = ''

        if len(main_digit) == 1:
            space = '    '
        elif len(main_digit) == 2:
            space = '   '
        elif len(main_digit) == 3:
            space = '  '
        elif len(main_digit) == 4:
            space = ' '
        elif len(main_digit) == 5:
            space = ''

        new_timestamp = square_bracket + space + timestamp
        return new_timestamp
