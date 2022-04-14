---
jupyter:
  colab:
    name: FlavourRecommender.ipynb
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.8.5
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown id="sY_Ong6SX-j3"}
# Flavor Recommender

Create a machine learning model to predict which flavor does a person
likes or prefers if the person\'s age and gender is given as input. Here
is a sample dataset of 20 customers with their age, gender and flavor
preference

## Import libraries

Import pandas and Scikit-Learn libraries to load the dataframe and and
the LabelEncoder class
:::

::: {.cell .code execution_count="1" colab="{\"height\":677,\"base_uri\":\"https://localhost:8080/\"}" id="t5uLqvdKXrWS" outputId="69026f3a-9daf-4c47-a470-096cbd2c1f43"}
``` {.python}
import pandas
df = pandas.read_csv('/content/flavour.csv')
df
```

::: {.output .execute_result execution_count="1"}
```{=html}
  <div id="df-09b60851-64d3-499d-ba09-bb906a8db0b3">
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
      <button class="colab-df-convert" onclick="convertToInteractive('df-09b60851-64d3-499d-ba09-bb906a8db0b3')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-09b60851-64d3-499d-ba09-bb906a8db0b3 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-09b60851-64d3-499d-ba09-bb906a8db0b3');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .code execution_count="2" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="zs0aFDywXrWU" outputId="7fc9dc8a-2c15-467a-e662-e9354d24fb0f" scrolled="true"}
``` {.python}
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(['Male', 'Female'])
```

::: {.output .execute_result execution_count="2"}
    LabelEncoder()
:::
:::

::: {.cell .code execution_count="3" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="6IftYd6ZXrWW" outputId="b5aec670-cbc5-4422-80b7-928728f0cd8b"}
``` {.python}
df['Gender'] = encoder.transform(df['Gender'])
df['Gender']
```

::: {.output .execute_result execution_count="3"}
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
:::
:::

::: {.cell .markdown id="uTaJ2x8pZdPz"}
## Split the data in X: features and Y: Labels
:::

::: {.cell .code execution_count="4" id="mDVSuIA1XrWY"}
``` {.python}
X = df.drop(columns=['Flavour'])
y = df.drop(columns=['Age', 'Gender'])
```
:::

::: {.cell .markdown id="0NGGDlkBZnp2"}
# Import and create a DecisionTreeClassifier from Scikit-Learn
:::

::: {.cell .code execution_count="5" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="NZq4F_iQXrWZ" outputId="2853601a-dc48-4325-e050-8311c8ffddee"}
``` {.python}
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X, y)
```

::: {.output .execute_result execution_count="5"}
    DecisionTreeClassifier()
:::
:::

::: {.cell .markdown id="gwE8vzRWaahg"}
## Prediction:

Use the trained model to predict the recommended flavor for a 18 years
old Male
:::

::: {.cell .code id="03PSFy1RXrWb" outputId="0fd46d45-19bc-42a6-f544-bd45774a24e9"}
``` {.python}
age = 18
gender = encoder.transform(['Male'])
model.predict([ [age, gender] ])
```

::: {.output .execute_result execution_count="18"}
    array(['Mango'], dtype=object)
:::
:::

::: {.cell .markdown id="jmTND9k-axM-"}
# Generalize a way to input age and gender to use the classifier
:::

::: {.cell .code execution_count="7" id="ny_oFyWpXrWc"}
``` {.python}
def flav_pred():
    age = int(input('Age:'))
    gen = input('Gender:').capitalize()
    gender = encoder.transform([gen])
    flav = model.predict([[age, gender]])
    print('Recommended Flavour:', flav[0])
```
:::

::: {.cell .code execution_count="8" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="E7vkGZeGXrWd" outputId="7d00564f-9db4-44e7-f26b-29b42f3ec34e"}
``` {.python}
flav_pred()
```

::: {.output .stream .stdout}
    Age:37
    Gender:Male
    Recommended Flavour: Almond & Chocolate
:::

::: {.output .stream .stderr}
    /usr/local/lib/python3.7/dist-packages/sklearn/base.py:451: UserWarning: X does not have valid feature names, but DecisionTreeClassifier was fitted with feature names
      "X does not have valid feature names, but"
:::
:::
