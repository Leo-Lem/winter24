FILE=model.py

# 'compile' all python files to the model.py file
rm $FILE
cat **/*.py >> $FILE
cat __main__.py >> $FILE

# remove all relative imports
sed -i 's/from \..*//g' $FILE
