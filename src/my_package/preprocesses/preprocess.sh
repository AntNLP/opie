#!/bin/bash
echo "preprocess begin"
echo "generates all sentence"
python3 process_raw_data.py -d $1

echo "split all sentence into 8 parts"
python3 split_data.py -d $1

python3 parse_sentence.py -d $1 -p 1

python3 parse_sentence.py -d $1 -p 2
python3 parse_sentence.py -d $1 -p 3
python3 parse_sentence.py -d $1 -p 4
python3 parse_sentence.py -d $1 -p 5
python3 parse_sentence.py -d $1 -p 6
python3 parse_sentence.py -d $1 -p 7
python3 parse_sentence.py -d $1 -p 8

echo "add parse info"
python3 add_parse_info.py -d $1 -p 1
python3 add_parse_info.py -d $1 -p 2
python3 add_parse_info.py -d $1 -p 3
python3 add_parse_info.py -d $1 -p 4
python3 add_parse_info.py -d $1 -p 5
python3 add_parse_info.py -d $1 -p 6
python3 add_parse_info.py -d $1 -p 7
python3 add_parse_info.py -d $1 -p 8
echo "preprocess end"
