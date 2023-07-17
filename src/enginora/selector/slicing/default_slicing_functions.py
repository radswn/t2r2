from snorkel.slicing import SlicingFunction, slicing_function
from textblob import TextBlob
from snorkel.preprocess import preprocessor


@slicing_function()
def short(x):
    return len(x.text.split()) < 60


@preprocessor(memoize=True)
def textblob_sentiment(x):
    scores = TextBlob(x.text)
    x.polarity = scores.sentiment.polarity
    return x


@slicing_function(pre=[textblob_sentiment])
def textblob_polarity(x):
    return x.polarity > 0.1
