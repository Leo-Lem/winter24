DIR=${1:-"/workspaces/w24/bai/paper/src"}
FILE=$DIR/${2:-"__assembled__"}.py

# 'compile' all python files to the model.py file
if [ -f $FILE ]; then
    rm $FILE
fi
cat $DIR/**/*.py >> $FILE
cat $DIR/__main__.py >> $FILE

# remove all relative imports
sed -i 's/from \..*//g' $FILE
