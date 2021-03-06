# Flavor Recommender

Create a machine learning model to predict which flavor does a person likes or prefers if the person's age and gender is given as input. Here is a sample dataset of 20 customers with their age, gender and flavor preference

## Import libraries
Import pandas and Scikit-Learn libraries to load the dataframe and and the LabelEncoder class


```python
import pandas
df = pandas.read_csv('/content/flavour.csv')
df
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>Flavour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>Male</td>
      <td>Chocolate</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>Female</td>
      <td>Strawberry</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>Male</td>
      <td>Chocolate</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>Female</td>
      <td>Strawberry</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>Male</td>
      <td>Butterscotch</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10</td>
      <td>Female</td>
      <td>Butterscotch</td>
    </tr>
    <tr>
      <th>6</th>
      <td>12</td>
      <td>Male</td>
      <td>Butterscotch</td>
    </tr>
    <tr>
      <th>7</th>
      <td>14</td>
      <td>Female</td>
      <td>Vanilla</td>
    </tr>
    <tr>
      <th>8</th>
      <td>15</td>
      <td>Male</td>
      <td>Mango</td>
    </tr>
    <tr>
      <th>9</th>
      <td>15</td>
      <td>Female</td>
      <td>Vanilla</td>
    </tr>
    <tr>
      <th>10</th>
      <td>17</td>
      <td>Male</td>
      <td>Mango</td>
    </tr>
    <tr>
      <th>11</th>
      <td>16</td>
      <td>Female</td>
      <td>Butterscotch</td>
    </tr>
    <tr>
      <th>12</th>
      <td>19</td>
      <td>Male</td>
      <td>Almond &amp; Chocolate</td>
    </tr>
    <tr>
      <th>13</th>
      <td>18</td>
      <td>Female</td>
      <td>Butterscotch</td>
    </tr>
    <tr>
      <th>14</th>
      <td>20</td>
      <td>Male</td>
      <td>Almond &amp; Chocolate</td>
    </tr>
    <tr>
      <th>15</th>
      <td>20</td>
      <td>Female</td>
      <td>Butterscotch</td>
    </tr>
    <tr>
      <th>16</th>
      <td>21</td>
      <td>Male</td>
      <td>Coffe</td>
    </tr>
    <tr>
      <th>17</th>
      <td>22</td>
      <td>Female</td>
      <td>Coffe</td>
    </tr>
    <tr>
      <th>18</th>
      <td>24</td>
      <td>Male</td>
      <td>Almond &amp; Chocolate</td>
    </tr>
    <tr>
      <th>19</th>
      <td>24</td>
      <td>Female</td>
      <td>Coffe</td>
    </tr>
  </tbody>
</table>
</div>

```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(['Male', 'Female'])
```




    LabelEncoder()




```python
df['Gender'] = encoder.transform(df['Gender'])
df['Gender']
```




    0     1
    1     0
    2     1
    3     0
    4     1
    5     0
    6     1
    7     0
    8     1
    9     0
    10    1
    11    0
    12    1
    13    0
    14    1
    15    0
    16    1
    17    0
    18    1
    19    0
    Name: Gender, dtype: int64



## Split the data in X: features and Y: Labels


```python
X = df.drop(columns=['Flavour'])
y = df.drop(columns=['Age', 'Gender'])
```

# Import and create a DecisionTreeClassifier from Scikit-Learn


```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X, y)
```




    DecisionTreeClassifier()



## Prediction:
Use the trained model to predict the recommended flavor for a 18 years old Male 


```python
age = 18
gender = encoder.transform(['Male'])
model.predict([ [age, gender] ])
```




    array(['Mango'], dtype=object)



# Generalize a way to input age and gender to use the classifier


```python
def flav_pred():
    age = int(input('Age:'))
    gen = input('Gender:').capitalize()
    gender = encoder.transform([gen])
    flav = model.predict([[age, gender]])
    print('Recommended Flavour:', flav[0])
```


```python
flav_pred()
```

    Age:37
    Gender:Male
    Recommended Flavour: Almond & Chocolate

