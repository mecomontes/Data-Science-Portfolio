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

## Rating bot

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

df = pd.read_csv('/content/reviews.csv')
df.head(20)
```
  <div id="df-f0a326f7-fcf2-4aca-880c-fa008b6c65c9">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Reviews</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The product is fairly good but it has scratche...</td>
      <td>Average</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Not same as told in the description</td>
      <td>Poor</td>
    </tr>
    <tr>
      <th>2</th>
      <td>It is worth the money!</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Delivered product is not genuine</td>
      <td>Poor</td>
    </tr>
    <tr>
      <th>4</th>
      <td>I'm not satisifed with the build quality</td>
      <td>Poor</td>
    </tr>
    <tr>
      <th>5</th>
      <td>This is considerably good for the price range</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>6</th>
      <td>The product is fine but the packaging isn't good</td>
      <td>Average</td>
    </tr>
    <tr>
      <th>7</th>
      <td>I am satisfied</td>
      <td>Average</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Certainly a better version is availble but it ...</td>
      <td>Average</td>
    </tr>
    <tr>
      <th>9</th>
      <td>The product is damaged</td>
      <td>Poor</td>
    </tr>
    <tr>
      <th>10</th>
      <td>I will recommend everyone to go for this</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>11</th>
      <td>It was not worth the money</td>
      <td>Poor</td>
    </tr>
    <tr>
      <th>12</th>
      <td>The product is a fake copy of the genuine</td>
      <td>Poor</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Evrything is fine except for the packaging</td>
      <td>Average</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Does not come with warranty card</td>
      <td>Poor</td>
    </tr>
    <tr>
      <th>15</th>
      <td>The build quality is awesome for the price</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Cannot find any better</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Complete satisfactory</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Packaging was torn off but the product is fine</td>
      <td>Average</td>
    </tr>
    <tr>
      <th>19</th>
      <td>This is not good for the price range</td>
      <td>Poor</td>
    </tr>
  </tbody>
</table>
</div>
 
```python
X = df['Reviews']
vectorized_features = CountVectorizer()
vectorized_features.fit(X)
vectorized_X = vectorized_features.transform(X)
vectorized_features.get_feature_names()
```

    ['am',
     'any',
     'as',
     'availble',
     'awesome',
     'better',
     'box',
     'build',
     'but',
     'cannot',
     'card',
     'certainly',
     'come',
     'complete',
     'considerably',
     'copy',
     'damaged',
     'delivered',
     'description',
     'does',
     'everyone',
     'evrything',
     'except',
     'fairly',
     'fake',
     'find',
     'fine',
     'for',
     'genuine',
     'go',
     'good',
     'has',
     'in',
     'is',
     'isn',
     'it',
     'money',
     'not',
     'of',
     'off',
     'on',
     'packaging',
     'price',
     'product',
     'quality',
     'range',
     'recommend',
     'same',
     'satisfactory',
     'satisfied',
     'satisifed',
     'scratches',
     'the',
     'this',
     'to',
     'told',
     'torn',
     'version',
     'warranty',
     'was',
     'will',
     'with',
     'worth']

```python
tfidf = TfidfTransformer()
tfidf.fit(vectorized_X)
X_reviews = tfidf.transform(vectorized_X)
X_reviews
```

    <20x63 sparse matrix of type '<class 'numpy.float64'>'
    	with 127 stored elements in Compressed Sparse Row format>

```python
y = df['Rating'].tolist()
y
```

    ['Average',
     'Poor',
     'Good',
     'Poor',
     'Poor',
     'Good',
     'Average',
     'Average',
     'Average',
     'Poor',
     'Good',
     'Poor',
     'Poor',
     'Average',
     'Poor',
     'Good',
     'Good',
     'Good',
     'Average',
     'Poor']

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_reviews, y)
```

    DecisionTreeClassifier()

```python
text = ['This product is in a good condition']
vectorized_text = vectorized_features.transform(text)
tfidf_text = tfidf.transform(vectorized_text)
model.predict(tfidf_text)
```

    array(['Poor'], dtype='<U7')

## Creating the function to rate reviews

```python
def rate(*comment):
  vectorized_text = vectorized_features.transform(comment)
  tfidf_text = tfidf.transform(vectorized_text)
  pred = model.predict(tfidf_text)

  for review, rating in zip(comment, pred):
    print(f'{review}\n Rating: {rating}')
```

```python
rate('Not in good condition', 'It is satisfactory', 'Too late')
```

    Not in good condition
     Rating: Poor
    It is satisfactory
     Rating: Good
    Too late
     Rating: Good

