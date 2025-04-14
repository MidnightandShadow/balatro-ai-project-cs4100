#! /usr/bin/env bash

set -eu

DATE=$(date --iso-8601=seconds)
RESULTS_DIR="../results"
EXE="python3 ./entrypoints/train_dqn.py"

printf "Title: "; read title
OUTPUT_DIR="$RESULTS_DIR/$title-$DATE"

mkdir -p "$OUTPUT_DIR"
# zip the results and delete folder
function exit_handler() {
    echo -e "\n\nZipping $OUTPUT_DIR..."
    zip "$OUTPUT_DIR.zip" "$OUTPUT_DIR" 2> /dev/null 1>&2
    rm -r "$OUTPUT_DIR"
}
trap exit_handler EXIT

printf "Description:\n"; 
while read line && [ "$line" != "" ]; do
    echo $line
done > "$OUTPUT_DIR/description.md"

exit 0;

source ../env/bin/activate 
cp ../entrypoints/train_dqn.py "$OUTPUT_DIR/entrypoint.py"
echo "$EXE $@" > "$OUTPUT_DIR/cmd.txt"
(cd ..; unbuffer python3 ./entrypoints/train_dqn.py $@) | tee "$OUTPUT_DIR/output.txt" || :

