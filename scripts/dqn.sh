#! /usr/bin/env bash

set -eu

DATE=$(date --iso-8601=seconds)
RESULTS_DIR="../results"
EXE="python3 ./entrypoints/train_dqn.py"

printf "Title: "; read title
DIR_NAME="$title-$DATE"
OUTPUT_DIR="$RESULTS_DIR/$DIR_NAME"

mkdir -p "$OUTPUT_DIR"
# zip the results and delete folder
function exit_handler() {
    echo -e "\n\nZipping $OUTPUT_DIR..."
    cd $RESULTS_DIR
    zip "$DIR_NAME.zip" "$DIR_NAME" 2> /dev/null 1>&2
    rm -r "$DIR_NAME"
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

