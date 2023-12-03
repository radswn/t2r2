from snorkel.preprocess import preprocessor
from snorkel.slicing import slicing_function
from textblob import TextBlob


@slicing_function()
def short(x):
    '''Short texts, below 60 characters'''
    return len(x.text.split()) < 60

@slicing_function()
def long(x):
    '''Long texts, above 100 characters'''
    return len(x.text.split()) > 100

@preprocessor(memoize=True)
def textblob_sentiment(x):
    scores = TextBlob(x.text)
    x.polarity = scores.sentiment.polarity
    return x

@slicing_function(pre=[textblob_sentiment])
def textblob_polarity(x):
    '''Slightly more positive sentiment(-1 is negative 1 is positive)'''
    return x.polarity > 0.1
