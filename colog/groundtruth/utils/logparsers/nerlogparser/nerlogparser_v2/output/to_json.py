import json


class ToJson(object):

    @staticmethod
    def write_to_json(parsed_logs, output_file):
        # write a dictionary to json file
        with open(output_file, 'w') as f:
            json.dump(parsed_logs, f)
