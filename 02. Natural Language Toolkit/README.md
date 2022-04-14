# NLTK: A Natural Language Toolkit


```python
from nltk import download, word_tokenize
# download()
download('punkt')
```

    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Unzipping tokenizers/punkt.zip.

    True

## Tokenize
Using the word_tokenize method from nltk to tokenize the data in the input file. Symbols are included when it will be tokenized.


```python
with open('/content/example.txt') as file:
  raw_text = file.read()

tokenized_text = word_tokenize(raw_text)
tokenized_text
```

    ['This',
     'is',
     'a',
     'short',
     'example',
     'text',
     'to',
     'work',
     'with',
     'natural',
     'languge',
     'toolkit',
     '.',
     'It',
     'is',
     'a',
     'really',
     'nice',
     'toolkit',
     'for',
     'NLP',
     '.']

## Vectorize
Create a vector with all the different words in the data. get_feature_names ust extract the words, ut Symbols won't be included.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

vectorized_text = CountVectorizer()
vectorized_text.fit(tokenized_text)
vectorized_text.get_feature_names()
```

    /usr/local/lib/python3.7/dist-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.
      warnings.warn(msg, category=FutureWarning)

    ['example',
     'for',
     'is',
     'it',
     'languge',
     'natural',
     'nice',
     'nlp',
     'really',
     'short',
     'text',
     'this',
     'to',
     'toolkit',
     'with',
     'work']

## Encode the text
vocabulary_ method is a way to encode the text. It returns a dictionary with a encoded value for each word.

Example: [3 2 8 9] means "it is really short"

```python
vectorized_text.vocabulary_
```

    {'example': 0,
     'for': 1,
     'is': 2,
     'it': 3,
     'languge': 4,
     'natural': 5,
     'nice': 6,
     'nlp': 7,
     'really': 8,
     'short': 9,
     'text': 10,
     'this': 11,
     'to': 12,
     'toolkit': 13,
     'with': 14,
     'work': 15}

## Generate a One Hot Encoder for each value in the array based on the vocabulary dictionary

```python
X = vectorized_text.transform(tokenized_text)
X.toarray()
```

    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

## TfidF

Transform a count matrix to a normalized tf or tf-idf representation.

Tf means term-frequency while tf-idf means term-frequency times inverse document-frequency. This is a common term weighting scheme in information retrieval, that has also found good use in document classification.

The goal of using tf-idf instead of the raw frequencies of occurrence of a token in a given document is to scale down the impact of tokens that occur very frequently in a given corpus and that are hence empirically less informative than features that occur in a small fraction of the training corpus.

The formula that is used to compute the tf-idf for a term t of a document d in a document set is tf-idf(t, d) = tf(t, d) * idf(t), and the idf is computed as idf(t) = log [ n / df(t) ] + 1 (if smooth_idf=False), where n is the total number of documents in the document set and df(t) is the document frequency of t; the document frequency is the number of documents in the document set that contain the term t. The effect of adding “1” to the idf in the equation above is that terms with zero idf, i.e., terms that occur in all documents in a training set, will not be entirely ignored. (Note that the idf formula above differs from the standard textbook notation that defines the idf as idf(t) = log [ n / (df(t) + 1) ]).

If smooth_idf=True (the default), the constant “1” is added to the numerator and denominator of the idf as if an extra document was seen containing every term in the collection exactly once, which prevents zero divisions: idf(t) = log [ (1 + n) / (1 + df(t)) ] + 1.

Furthermore, the formulas used to compute tf and idf depend on parameter settings that correspond to the SMART notation used in IR as follows:

Tf is “n” (natural) by default, “l” (logarithmic) when sublinear_tf=True. Idf is “t” when use_idf is given, “n” (none) otherwise. Normalization is “c” (cosine) when norm='l2', “n” (none) when norm=None.

```python
tfidf = TfidfTransformer()
tfidf.fit(X)
```

    TfidfTransformer()

```python
tfidf_text = tfidf.transform(X)
tfidf_text
```

    <22x16 sparse matrix of type '<class 'numpy.float64'>'
    	with 18 stored elements in Compressed Sparse Row format>

