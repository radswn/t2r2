from snorkel.preprocess import preprocessor
from snorkel.slicing import slicing_function
from textblob import TextBlob


@slicing_function()
def short(x):
    return len(x.text.split()) < 60

@slicing_function()
def long(x):
    return len(x.text.split()) > 100

@preprocessor(memoize=True)
def textblob_sentiment(x):
    scores = TextBlob(x.text)
    x.polarity = scores.sentiment.polarity
    return x


@slicing_function(pre=[textblob_sentiment])
def textblob_polarity(x):
    return x.polarity > 0.1
