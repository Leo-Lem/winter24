DEST=${1:-/workspaces/w24/BAI/paper/src/data/flickr8k}

# Download flickr8k dataset#
if [ ! -d $DEST ]; then
  mkdir -p $DEST
fi
curl -L -o $DEST/flickr8k.zip https://www.kaggle.com/api/v1/datasets/download/adityajn105/flickr8k
unzip $DEST/flickr8k.zip -d $DEST
rm $DEST/flickr8k.zip

# Download spacy model
spacy download en_core_web_sm