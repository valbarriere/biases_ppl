#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <list_gender> <country1> [<country2> ...]"
    exit 1
fi

list_gender=$1
shift

if [ "$#" -lt 1 ]; then
    echo "Please provide at least one country."
    exit 1
fi

countries=("$@")
command_template='export DEVICE_MAP="cuda:%d" && python PPL_Positivity_Mixtral.py --data_tsv Eurotweets_English_val_without_line_return.tsv_clean_test --model_name mistralai/Mixtral-8x7B-Instruct-v0.1 --list_gender %s --list_countries "%s" --verbose'

index=0

for country in "${countries[@]}"; do
    conda init && conda activate biases && new-session -d -s "country_${index}" "$(printf "$command_template" "$index" "$list_gender" "$country")"
    ((index++))
done
