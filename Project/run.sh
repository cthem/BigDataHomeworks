#! /usr/bin/env bash
[ -z $(which phantomjs) ] && echo "Need phantomjs to create output report files!" && exit 
python main.py "$(pwd)/input_folder" "$(pwd)/output_folder"
