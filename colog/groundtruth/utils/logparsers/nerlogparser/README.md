# nerlogparser: An automatic log parser

This is source code implementation of our paper entitled ["Automatic log parser to support forensic analysis"](http://researchrepository.murdoch.edu.au/id/eprint/42841/) published in the 16th Australian Digital Forensics Conference, pp. 1-10, 2018. We name the tool as `nerlogparser` because it uses named entity recognition (NER) technique to parse log files. This repository is a fork from [sequence_tagging](https://github.com/guillaumegenthial/sequence_tagging) by Guillaume Genthial.

## Requirements
1. Python 3.5
2. TensorFlow 1.4.1
3. nltk 3.4

## How to install
1. Create a new directory for `nerlogparser` in your home directory

   `mkdir $HOME/nerlogparser`

2. Create virtual environment in newly created directory with specific Python version (3.5)

   `virtualenv $HOME/nerlogparser -p /usr/bin/python3.5`

3. Activate the virtual environment

   `source $HOME/nerlogparser/bin/activate`

4. Install `nerlogparser`

   `pip install nerlogparser`

## How to run
1. Make sure your are still in the virtual environment mode
2. For example, run `nerlogparser` to parse authentication log file from `/var/log/auth.log` and print output to the screen

   `nerlogparser -i /var/log/auth.log`

3. We can save parsing results in an output file such as `parsed-auth.json`. At the moment, the only supported file output format is JSON.

   `nerlogparser -i /var/log/auth.log -o parsed-auth.json`

4. Run `nerlogpaser` help

   `nerlogparser -h`

## Import from your Python script

```python
import pprint
from nerlogparser.nerlogparser import Nerlogparser  

parser = Nerlogparser()
parsed_logs = parser.parse_logs('/var/log/auth.log')  

for line_id, parsed in parsed_logs.items():
    print('Line:', line_id)
    pprint.pprint(parsed)
    print()
```    

## License
Apache License 2.0. Please check [LICENSE](https://github.com/studiawan/nerlogparser/blob/master/LICENSE.txt).
