#### import all necessary libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

#### as data is in the form of excel, use read_excel command


```python
df = pd.read_excel('/content/Bank_Personal_Loan_Modelling.xlsx', 1)
df.head()
```





  <div id="df-067bfe51-92f0-4b4e-a34f-9a0cd2033f8e">
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
      <th>ID</th>
      <th>Age</th>
      <th>Experience</th>
      <th>Income</th>
      <th>ZIP Code</th>
      <th>Family</th>
      <th>CCAvg</th>
      <th>Education</th>
      <th>Mortgage</th>
      <th>Personal Loan</th>
      <th>Securities Account</th>
      <th>CD Account</th>
      <th>Online</th>
      <th>CreditCard</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>25</td>
      <td>1</td>
      <td>49</td>
      <td>91107</td>
      <td>4</td>
      <td>1.6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>45</td>
      <td>19</td>
      <td>34</td>
      <td>90089</td>
      <td>3</td>
      <td>1.5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>39</td>
      <td>15</td>
      <td>11</td>
      <td>94720</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>35</td>
      <td>9</td>
      <td>100</td>
      <td>94112</td>
      <td>1</td>
      <td>2.7</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>35</td>
      <td>8</td>
      <td>45</td>
      <td>91330</td>
      <td>4</td>
      <td>1.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-067bfe51-92f0-4b4e-a34f-9a0cd2033f8e')"
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
          document.querySelector('#df-067bfe51-92f0-4b4e-a34f-9a0cd2033f8e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-067bfe51-92f0-4b4e-a34f-9a0cd2033f8e');
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




### Perform Exploratory Data Analysis


```python
df.shape
```




    (5000, 14)




```python
df.isnull().sum()
```




    ID                    0
    Age                   0
    Experience            0
    Income                0
    ZIP Code              0
    Family                0
    CCAvg                 0
    Education             0
    Mortgage              0
    Personal Loan         0
    Securities Account    0
    CD Account            0
    Online                0
    CreditCard            0
    dtype: int64




```python
# ID and Zipcode might be removed as they may not be useful for our analysis
df.drop(['ID', 'ZIP Code'], axis=1, inplace=True)
```


```python
df.columns
```




    Index(['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education',
           'Mortgage', 'Personal Loan', 'Securities Account', 'CD Account',
           'Online', 'CreditCard'],
          dtype='object')




```python
import plotly.express as px
```


```python
fig = px.box(df, y=['Age', 'Experience', 'Income', 'Family', 'Education'])
fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-2.8.3.min.js"></script>                <div id="b4744e25-35e2-480f-a3d4-972b0ef64c6c" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("b4744e25-35e2-480f-a3d4-972b0ef64c6c")) {                    Plotly.newPlot(                        "b4744e25-35e2-480f-a3d4-972b0ef64c6c",                        [{"alignmentgroup":"True","hovertemplate":"variable=%{x}<br>value=%{y}<extra></extra>","legendgroup":"","marker":{"color":"#636efa"},"name":"","notched":false,"offsetgroup":"","orientation":"v","showlegend":false,"x":["Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Age","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Experience","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Income","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Family","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education","Education"],"x0":" ","xaxis":"x","y":[25,45,39,35,35,37,53,50,35,34,65,29,48,59,67,60,38,42,46,55,56,57,29,44,36,43,40,46,56,38,59,40,53,30,31,48,59,51,42,38,57,34,32,39,46,57,39,37,56,40,32,61,30,50,29,41,55,56,28,31,49,47,42,42,47,59,62,53,47,53,42,53,44,41,28,31,58,46,54,50,60,47,41,33,46,27,40,48,65,25,55,35,43,60,65,38,41,54,49,66,48,61,53,43,56,24,43,42,33,43,41,62,40,58,39,65,54,58,41,32,54,52,58,37,39,60,31,34,38,41,28,58,31,30,53,58,59,49,59,59,51,35,33,25,49,59,46,50,52,48,46,26,57,60,54,24,26,41,32,61,29,61,38,28,53,27,25,33,50,27,27,52,38,58,42,45,52,29,53,62,51,36,24,29,52,39,48,46,64,55,60,51,50,48,53,34,48,55,27,36,32,35,30,58,56,38,49,34,40,64,51,44,46,57,54,38,27,39,44,56,32,45,26,55,52,39,24,47,47,48,47,35,46,62,26,38,43,62,57,28,51,48,41,65,41,35,38,53,55,26,30,54,65,47,65,66,26,66,35,56,51,42,49,27,45,49,63,47,64,43,60,40,29,41,30,49,30,29,50,39,33,57,34,61,44,40,51,37,44,42,51,43,30,45,35,60,34,55,43,41,34,65,45,49,48,60,55,42,32,62,57,52,36,34,63,24,57,40,27,65,60,44,63,59,56,54,52,58,60,28,54,32,59,63,48,56,36,57,29,39,59,31,43,35,54,51,44,25,40,26,39,28,52,53,44,43,56,38,30,32,35,43,58,25,54,57,50,32,63,31,36,58,56,49,30,33,45,30,47,25,63,55,65,44,51,35,30,31,54,45,45,58,54,53,33,60,50,26,54,28,36,29,54,55,61,36,45,64,60,49,47,60,45,32,52,35,40,53,27,58,47,28,46,43,56,28,42,32,62,37,51,39,43,52,30,52,61,36,58,47,64,52,58,48,64,58,47,49,31,61,51,28,39,54,50,30,64,29,48,35,60,55,29,48,43,66,25,45,34,48,32,50,43,64,60,43,60,64,44,60,54,33,56,29,25,60,55,39,37,53,34,42,60,50,41,25,50,48,32,50,59,50,44,31,40,36,51,64,47,52,51,31,39,30,27,41,53,54,28,34,61,48,36,56,24,64,26,33,64,39,54,32,62,27,53,51,25,44,31,57,25,30,40,63,35,43,27,44,49,61,60,59,28,52,28,34,60,39,30,49,43,63,28,51,33,55,53,34,34,40,49,35,39,55,29,54,25,52,46,57,52,28,44,24,40,34,39,50,41,31,29,30,44,33,50,42,48,24,56,28,56,58,29,63,28,57,34,28,27,37,52,49,65,60,37,63,40,46,63,57,33,41,41,44,33,52,30,45,49,45,32,45,57,61,57,60,40,53,42,62,43,35,50,45,52,35,58,62,50,25,47,28,34,49,54,50,37,38,52,63,39,63,65,46,54,54,52,63,66,56,23,65,51,34,49,29,47,46,52,55,61,34,58,40,43,35,24,48,44,54,59,45,26,40,32,29,51,42,64,44,37,44,35,41,56,62,58,47,35,29,43,62,41,34,50,47,29,59,56,61,58,49,45,50,64,39,58,62,45,58,43,28,26,49,66,33,61,64,36,49,52,61,32,61,45,30,62,57,41,58,29,53,64,49,38,56,56,52,64,53,29,48,32,54,37,47,37,64,43,33,26,42,54,41,55,65,50,55,62,53,32,56,54,44,48,46,45,45,58,29,55,55,41,24,54,57,30,42,29,29,31,47,36,52,54,55,53,52,64,54,32,63,36,50,33,62,49,41,51,56,51,39,61,35,39,37,48,63,35,55,29,61,36,61,36,58,42,30,45,39,27,57,34,64,47,44,51,40,57,33,46,41,33,27,52,59,62,49,45,63,57,60,50,54,28,60,44,61,40,54,43,54,32,24,30,61,40,35,33,63,57,44,51,51,38,31,54,41,57,24,55,32,38,58,29,43,50,62,57,30,30,57,57,43,28,46,29,64,66,23,60,47,35,57,65,53,44,45,41,51,27,31,32,55,55,42,33,65,35,55,28,27,51,50,58,46,62,38,62,56,61,38,55,30,41,57,36,55,30,40,32,59,44,51,37,59,37,56,55,49,46,35,47,39,27,62,57,55,55,41,57,43,40,43,63,63,54,54,52,50,38,36,58,46,50,46,53,62,63,42,34,40,34,41,32,28,33,46,52,60,59,57,46,25,53,38,63,44,50,28,27,52,50,39,50,60,30,31,39,29,54,35,27,45,58,62,28,32,29,41,61,56,37,60,49,43,53,35,56,41,36,56,64,51,49,43,51,55,62,35,53,33,43,58,63,31,36,30,59,28,59,47,47,56,41,25,30,50,34,44,36,39,54,33,39,41,40,29,51,54,47,53,30,28,60,51,63,38,59,53,31,41,25,27,50,50,43,50,46,30,42,42,29,38,51,35,37,47,55,65,58,54,52,28,32,51,43,42,42,41,34,30,32,46,38,37,32,35,30,30,58,57,34,31,39,57,47,51,30,55,32,32,44,33,49,32,31,37,41,56,55,49,63,55,59,41,49,48,41,50,28,36,38,34,41,43,30,37,62,40,35,64,49,24,36,29,29,28,33,36,42,25,28,50,34,43,62,61,45,42,39,29,45,58,29,32,37,48,40,29,36,35,35,62,26,32,63,38,50,46,50,61,34,27,61,45,50,44,62,45,41,30,61,45,59,30,60,39,56,56,27,66,43,53,44,54,31,38,28,51,52,64,29,34,33,46,48,52,44,51,47,39,42,57,36,27,31,63,34,52,57,63,26,35,58,32,64,50,34,36,43,28,64,60,62,27,42,45,36,48,65,39,51,30,65,38,29,42,63,46,62,58,56,56,34,42,30,61,38,50,61,41,42,29,51,32,34,26,54,38,62,37,46,52,32,49,28,55,52,32,31,27,32,52,52,50,32,61,60,28,34,31,31,62,47,50,36,26,51,52,35,42,36,41,49,57,44,60,38,26,29,59,51,50,35,61,42,55,50,64,54,50,31,32,44,60,60,62,46,57,30,58,39,60,59,50,63,27,54,62,60,38,34,65,55,57,27,35,52,45,29,44,47,62,52,47,42,65,42,40,32,40,55,32,58,46,53,63,40,41,60,65,59,48,59,33,40,42,65,30,30,42,32,55,29,64,37,31,25,31,32,58,26,51,65,43,46,28,63,59,42,58,39,36,60,47,29,52,41,63,59,44,54,29,51,63,36,42,51,47,40,54,47,35,28,45,33,62,45,59,58,52,34,65,48,44,61,40,65,28,67,35,60,58,55,34,35,28,38,62,30,38,33,58,59,52,36,45,49,52,54,30,65,34,30,51,52,43,35,56,57,58,53,45,44,54,41,52,43,63,54,33,25,41,40,43,36,57,34,38,47,39,45,62,59,61,36,58,55,29,34,61,50,52,39,55,33,47,57,57,40,50,29,46,42,59,31,51,35,59,35,46,34,55,64,34,61,63,59,51,41,37,64,44,62,50,43,34,38,29,39,53,43,61,46,57,59,52,29,57,49,39,56,63,37,56,45,66,40,50,60,31,40,36,55,54,35,55,36,66,38,58,41,60,47,62,48,61,29,45,39,31,39,63,28,56,31,46,42,53,41,61,31,62,59,49,65,30,32,56,36,58,27,41,59,56,52,35,47,29,31,62,48,26,60,35,40,31,50,33,37,38,63,57,61,37,51,44,63,43,38,34,48,29,37,60,46,34,56,57,62,32,51,55,60,40,62,63,60,59,26,56,58,57,48,30,45,64,44,51,43,29,56,65,46,48,56,61,46,58,31,27,44,44,51,39,32,33,40,36,52,54,26,39,46,57,59,52,52,50,41,43,25,40,35,60,57,44,61,33,45,45,64,50,28,37,62,29,49,46,60,55,33,53,50,28,42,33,40,31,41,52,65,48,45,26,64,41,43,60,62,46,36,31,43,46,50,52,27,34,49,52,37,53,54,29,35,32,38,44,44,48,46,35,56,49,57,35,44,38,57,35,29,58,40,51,61,46,55,35,60,28,43,61,48,65,45,36,45,60,47,32,48,33,49,56,59,56,30,59,38,47,54,34,41,47,44,43,31,28,55,42,53,30,65,43,56,25,35,50,36,34,32,51,52,65,51,37,35,67,30,62,42,48,61,36,48,65,25,55,63,31,43,28,37,27,62,51,56,56,44,46,56,56,57,31,65,31,36,56,52,42,55,49,51,26,32,54,50,59,61,43,39,56,38,25,42,42,50,56,43,60,42,57,48,37,57,62,39,38,54,45,39,45,62,43,30,35,58,44,56,28,64,63,44,34,50,51,30,55,57,43,61,49,52,57,53,52,39,58,36,45,30,49,44,43,42,29,28,50,44,52,28,62,34,45,52,43,54,64,27,42,28,47,39,29,39,41,37,41,45,52,58,31,26,31,42,56,52,59,32,46,52,30,32,35,49,54,56,48,28,44,30,44,30,47,64,48,63,25,61,46,57,40,49,30,41,42,63,43,59,46,33,55,36,47,59,38,42,30,63,60,62,49,59,36,46,35,50,51,41,45,41,57,51,52,43,63,28,43,41,34,28,58,39,49,33,37,33,28,54,63,57,56,54,29,41,58,61,30,62,52,29,46,52,40,49,34,35,26,65,52,32,31,36,50,36,51,39,53,50,31,53,48,57,47,55,37,59,53,31,35,25,37,40,31,62,41,56,47,28,60,27,57,62,57,44,31,31,39,41,41,55,28,35,44,44,40,65,35,55,55,59,39,50,45,50,65,36,57,53,28,55,56,33,57,27,27,54,48,62,41,62,40,32,62,35,25,50,61,43,52,39,33,27,27,32,65,55,52,39,35,39,34,30,37,41,31,37,49,58,45,40,34,62,54,26,54,29,48,27,42,25,45,34,51,51,60,59,49,50,41,49,50,63,63,33,38,64,36,58,39,46,61,53,28,64,48,38,52,65,59,45,53,38,54,25,61,48,46,36,46,59,59,36,63,51,30,48,55,41,26,41,54,57,54,35,60,63,41,46,31,58,59,46,33,56,47,59,24,39,30,55,47,35,47,38,38,27,42,26,60,27,27,40,40,29,30,30,47,33,57,38,54,47,48,62,30,35,59,38,47,57,42,39,53,27,59,48,62,66,38,42,47,27,32,37,56,39,36,32,62,48,58,27,52,54,31,60,34,46,41,62,31,41,55,45,51,27,30,31,61,40,45,45,37,34,43,42,56,33,36,62,58,65,65,52,64,51,59,52,55,46,61,35,56,31,44,53,36,27,36,40,39,59,43,30,26,48,50,33,32,34,33,32,55,58,47,30,42,40,33,46,63,62,43,31,28,64,27,33,39,44,53,42,44,34,47,53,62,61,42,48,38,41,57,31,39,48,55,29,47,61,60,34,45,60,25,41,63,63,43,58,50,38,54,61,29,39,33,23,56,54,37,38,34,53,65,62,51,31,64,39,28,60,47,25,44,51,31,32,51,25,43,54,34,54,42,46,62,31,30,52,35,60,58,24,40,49,43,33,36,62,57,64,52,54,40,30,55,39,55,28,44,46,61,61,45,38,29,52,38,28,34,35,46,63,33,38,53,28,44,58,38,48,33,66,59,40,36,62,51,58,52,41,31,28,57,61,60,56,49,63,49,49,32,26,27,57,29,56,59,53,54,56,50,51,53,37,32,49,34,54,64,58,25,50,38,48,41,32,46,39,39,63,58,28,36,43,36,44,31,45,39,43,40,30,58,46,35,30,34,62,58,45,42,60,55,45,52,31,60,33,37,59,51,47,60,61,64,46,31,57,48,48,35,33,47,46,44,42,50,52,53,37,64,46,57,61,42,40,33,50,52,35,57,56,44,23,33,48,45,54,42,47,61,53,56,33,44,63,47,50,49,34,40,38,51,28,52,39,29,54,63,40,36,45,61,26,33,64,43,24,30,60,50,40,31,30,60,39,66,65,54,54,35,32,63,60,43,59,62,30,54,30,31,44,32,63,57,60,37,53,51,30,28,50,52,56,40,64,61,46,55,45,40,63,57,38,37,31,50,42,51,38,53,43,35,49,28,51,39,31,44,46,42,41,23,59,47,48,58,58,54,49,63,62,45,39,58,56,29,33,55,56,36,53,57,35,43,54,29,39,32,51,50,54,38,32,62,57,47,51,54,26,37,27,63,62,59,32,35,56,55,31,54,58,48,48,33,36,41,55,63,53,38,46,59,56,59,39,47,47,53,36,34,36,60,45,27,47,44,54,57,52,51,57,65,58,64,52,58,52,43,56,37,53,27,53,42,58,36,53,48,54,26,50,31,35,63,29,57,30,33,62,35,37,37,35,35,59,53,45,46,40,42,25,39,30,58,41,37,36,27,60,62,67,44,24,45,58,61,54,28,49,35,36,34,36,35,27,42,65,29,65,59,46,59,52,60,43,65,50,48,49,58,24,58,45,42,64,49,55,40,28,57,50,40,55,53,48,59,58,38,49,60,54,28,27,42,52,54,56,58,56,64,35,45,57,35,46,30,44,39,42,34,46,55,28,35,60,50,52,51,51,59,53,43,31,32,41,31,41,47,37,53,49,62,33,54,27,60,29,56,55,45,33,45,44,37,42,26,33,61,31,54,62,61,66,38,55,60,23,29,35,53,32,38,44,43,54,52,31,47,42,57,33,35,56,57,25,53,59,56,54,48,55,33,46,42,49,42,46,65,50,54,42,57,61,60,40,33,37,52,33,62,62,63,55,33,25,55,29,45,60,44,48,57,58,58,44,54,59,63,61,58,44,53,63,41,50,51,47,43,46,50,33,27,34,34,28,29,52,47,41,58,37,57,63,60,50,60,54,28,45,28,54,42,30,61,64,38,39,54,59,39,63,31,56,47,28,32,54,29,39,26,29,46,38,55,48,36,39,40,26,55,61,57,56,31,61,58,43,29,50,49,43,58,41,65,52,55,49,52,56,30,36,41,42,60,44,34,56,31,29,31,36,42,64,61,35,28,38,44,45,46,57,40,38,39,23,47,32,30,54,25,60,61,36,52,33,57,34,50,43,34,26,26,48,49,47,43,40,31,27,55,54,23,54,57,41,28,33,51,28,63,29,62,51,52,43,39,35,34,49,43,48,30,46,43,27,39,58,44,39,35,41,43,55,32,56,30,65,31,41,55,37,34,34,33,48,28,30,44,61,59,33,56,53,42,43,35,61,39,61,40,34,65,40,39,61,40,49,43,45,52,32,31,27,33,65,62,55,46,37,60,44,35,52,30,62,41,38,52,48,47,41,44,31,50,36,52,62,55,61,34,34,59,41,33,55,64,44,32,67,40,57,59,43,58,50,52,35,40,31,32,55,43,31,26,58,51,45,56,25,38,62,39,56,50,52,53,25,44,42,42,63,57,56,60,62,48,37,55,42,39,47,34,48,52,53,49,47,48,38,48,56,65,46,60,50,41,41,60,57,48,53,42,45,35,34,67,36,37,40,35,60,59,35,27,29,35,38,62,43,35,41,65,61,55,28,52,34,49,42,49,49,32,59,43,48,31,30,58,41,38,33,51,45,34,39,44,55,28,57,43,46,35,25,65,64,39,62,46,36,42,35,63,45,27,29,55,32,37,25,41,52,31,40,54,48,39,64,54,39,55,42,58,45,29,36,63,55,54,61,36,61,39,57,35,66,49,48,61,44,23,31,39,45,39,64,64,47,34,56,33,56,57,43,43,26,64,43,44,60,37,56,54,43,57,29,54,61,29,47,43,46,55,48,26,63,57,58,28,61,65,33,63,43,26,57,50,54,59,49,54,65,34,31,31,64,52,57,60,45,39,25,29,40,36,33,51,35,54,29,32,37,55,30,49,51,65,32,29,46,64,27,50,33,38,38,37,46,31,35,50,45,30,60,31,60,36,64,29,58,59,58,35,43,33,54,38,38,57,34,52,50,60,26,56,39,45,30,37,45,48,65,46,40,33,40,60,51,41,37,35,35,46,60,51,31,30,32,53,33,40,57,51,30,41,54,42,30,60,56,63,56,39,29,28,41,28,49,30,63,45,40,28,62,38,32,58,33,60,34,38,44,56,37,45,44,37,47,51,63,61,43,41,59,29,32,64,50,35,34,58,41,37,35,45,53,53,54,28,58,47,24,27,42,50,41,46,46,51,59,58,37,39,47,51,64,59,55,57,59,42,34,41,43,53,47,49,35,52,53,48,35,52,60,33,38,29,35,26,48,43,60,27,38,40,38,50,38,34,42,60,62,59,49,49,36,33,43,53,57,53,60,34,51,36,63,37,57,52,38,61,31,39,38,46,48,58,50,67,36,30,58,43,31,37,49,27,50,46,49,29,55,61,45,33,63,32,42,51,44,33,39,56,28,43,30,34,26,58,43,40,54,44,54,39,59,53,32,40,54,27,63,26,33,43,57,26,55,30,63,55,35,45,47,31,56,49,53,62,63,26,59,40,42,29,40,31,35,62,51,32,27,62,66,53,49,65,30,60,30,54,54,37,32,51,46,41,62,54,52,51,24,61,55,37,64,34,31,42,47,29,34,36,34,26,48,47,39,62,34,43,55,65,26,57,32,35,63,49,23,30,43,39,31,65,34,37,62,33,48,33,45,44,37,31,56,30,61,32,51,26,31,43,57,42,48,46,33,45,31,42,56,63,42,50,31,65,60,34,62,56,51,44,41,43,25,40,53,54,36,26,35,29,35,28,48,46,31,40,27,32,67,24,45,26,42,65,59,30,32,36,48,56,44,60,51,34,45,47,29,52,61,40,24,33,33,52,40,45,27,38,50,41,60,64,34,30,31,41,61,42,48,59,57,37,53,53,26,39,35,59,43,39,49,47,41,57,42,61,56,29,25,32,37,31,38,40,61,50,32,62,62,40,59,43,62,48,29,58,43,39,33,40,28,38,65,35,29,61,46,50,60,54,43,38,46,64,24,39,34,65,38,62,59,49,57,64,47,30,42,53,50,62,34,47,62,61,59,47,65,56,56,31,61,42,44,47,30,62,56,25,53,26,59,62,58,40,35,51,41,51,27,46,46,31,58,42,59,54,35,34,46,52,55,34,57,45,29,49,36,57,25,43,27,54,53,55,43,35,59,42,51,57,39,53,31,33,38,47,63,44,61,52,59,56,58,30,42,51,60,30,49,26,36,65,27,60,32,46,60,28,50,52,29,30,42,32,40,49,53,42,38,60,27,61,27,45,41,44,38,39,48,47,64,27,66,43,34,28,52,45,24,39,40,30,49,53,56,50,53,60,58,43,46,29,56,48,61,41,35,48,43,37,47,29,63,43,57,55,47,58,53,59,46,41,46,44,44,50,51,55,37,34,59,45,30,32,61,54,35,63,66,48,60,41,31,58,67,35,40,42,44,47,59,29,36,47,55,41,51,26,33,30,30,45,40,42,50,62,63,43,49,51,61,43,43,61,35,59,40,61,48,37,56,35,35,40,50,49,46,64,60,45,52,58,54,48,51,53,57,43,37,32,34,54,62,56,39,32,50,27,37,60,43,62,39,34,46,46,51,44,60,65,58,37,52,42,54,54,51,59,41,43,52,52,57,53,58,59,57,27,42,52,49,47,45,25,47,44,30,63,50,40,56,39,42,28,26,58,38,23,53,54,42,54,66,46,63,63,58,65,35,33,43,30,61,49,52,45,64,26,35,45,44,34,65,32,41,52,35,51,30,58,49,63,40,27,38,52,49,59,32,30,64,59,62,61,53,51,55,36,44,26,54,35,34,28,32,38,53,26,45,58,59,45,64,30,40,61,40,40,43,39,35,38,67,55,28,30,59,26,52,40,31,50,27,64,34,30,39,34,40,33,38,42,37,33,60,28,45,56,53,37,47,58,52,46,52,24,57,66,30,48,63,48,34,60,55,50,29,61,50,37,64,43,39,23,34,29,33,60,49,54,59,42,62,63,57,61,35,26,33,31,51,55,38,38,53,62,35,46,60,63,43,33,43,62,48,38,36,49,61,49,59,30,44,67,59,37,50,56,29,55,48,32,47,46,33,39,60,39,34,55,67,40,44,56,50,31,66,43,58,33,33,32,55,25,40,54,36,35,44,38,30,39,35,41,56,52,29,38,51,45,51,53,50,59,57,45,27,40,39,26,27,55,64,41,46,43,24,29,58,45,53,45,32,56,31,29,48,36,36,41,48,27,33,31,48,59,41,42,62,62,51,48,56,62,53,62,28,61,48,50,58,53,65,27,50,50,41,43,55,33,44,47,43,59,65,28,58,33,24,46,26,47,32,58,32,46,35,53,55,63,45,58,50,37,25,52,26,35,58,37,35,31,58,43,43,54,53,32,37,34,51,49,54,37,57,37,32,48,44,50,44,54,37,34,32,63,56,37,66,38,35,61,52,57,47,50,36,45,58,27,27,48,46,32,54,31,42,30,41,44,37,51,30,36,65,33,58,34,38,59,37,59,47,48,38,34,44,33,47,41,36,28,59,43,56,28,62,40,34,52,40,27,52,39,52,50,40,35,39,25,33,26,46,27,55,52,59,63,61,58,29,51,59,41,59,52,39,45,59,49,48,61,31,42,35,57,54,61,60,59,62,26,41,65,25,25,27,65,60,29,32,32,41,52,40,39,34,34,34,41,59,40,52,37,39,49,63,34,51,61,56,62,56,58,58,50,44,49,31,49,43,31,66,41,39,46,57,59,30,26,46,66,50,61,37,51,56,58,41,35,38,26,35,36,26,53,56,44,47,32,52,39,47,35,26,43,52,30,36,48,36,58,35,59,36,45,56,46,26,37,44,44,33,34,35,48,58,30,62,40,42,43,58,36,29,49,60,58,50,46,45,32,42,30,60,46,32,56,56,30,52,31,37,30,29,49,49,65,54,36,56,34,33,59,49,61,31,45,35,37,58,49,63,55,38,45,44,58,56,37,50,34,51,49,33,61,41,50,41,38,51,63,55,46,27,59,26,61,44,53,34,40,56,57,43,38,60,54,51,41,25,58,61,56,43,42,48,45,40,43,52,54,26,26,33,40,64,62,54,34,40,41,48,46,51,30,65,49,29,36,50,41,42,37,31,40,36,64,37,43,57,62,63,57,59,47,26,59,45,33,61,54,46,28,52,26,49,42,51,39,44,29,47,53,29,47,45,63,39,29,50,51,58,39,46,32,53,29,41,41,58,45,37,58,58,31,59,38,29,40,57,50,29,34,36,51,27,48,32,48,34,24,55,51,30,45,64,29,30,63,65,28,1,19,15,9,8,13,27,24,10,9,39,5,23,32,41,30,14,18,21,28,31,27,5,18,11,19,16,20,30,13,35,16,28,6,5,24,35,25,18,13,32,9,7,15,20,31,14,12,26,16,8,37,6,26,5,17,30,31,2,5,24,21,18,17,23,35,36,23,21,29,18,29,20,16,3,7,32,20,30,26,36,22,16,9,22,2,16,22,41,-1,30,10,19,34,39,12,15,28,23,41,23,36,23,18,32,0,17,18,7,17,14,38,15,34,14,40,29,33,16,7,29,26,32,13,15,35,5,8,14,16,4,34,1,4,29,33,32,25,34,35,25,11,9,1,23,35,19,25,28,22,22,0,32,36,29,0,0,17,6,35,0,35,12,4,27,1,1,9,26,1,1,28,13,34,17,20,25,3,29,37,27,12,0,3,26,14,23,21,40,29,36,25,23,24,29,10,24,31,3,11,6,9,3,34,31,12,25,10,16,39,26,18,22,33,28,14,3,14,20,30,6,19,2,25,27,13,-1,23,22,24,22,10,19,37,1,8,18,38,32,3,26,22,16,39,17,11,14,29,29,1,6,28,40,21,41,40,0,41,9,30,27,16,23,1,19,23,38,22,39,13,36,14,3,16,5,24,5,2,26,14,8,31,9,36,19,13,25,12,19,15,25,16,5,19,9,34,9,25,19,15,9,39,21,25,23,35,29,18,8,38,32,26,6,9,37,-2,31,16,2,39,34,20,39,34,30,28,27,32,35,4,30,6,33,38,23,32,12,27,3,13,34,6,19,5,24,27,19,0,15,2,14,4,28,29,20,19,30,14,6,6,10,19,32,0,24,32,24,8,37,6,12,34,30,25,5,7,21,5,23,0,33,29,41,18,25,9,5,5,30,20,19,32,29,28,9,35,24,2,30,3,10,2,28,30,36,11,19,40,36,22,23,36,20,7,28,8,15,29,0,33,22,3,20,19,30,3,18,7,38,13,26,13,17,28,6,26,35,9,32,23,39,27,28,24,40,32,23,23,4,37,25,-2,13,28,24,4,39,3,24,10,36,30,4,22,19,42,0,20,10,23,6,24,19,39,34,19,34,39,20,36,29,9,32,5,1,34,30,13,13,28,10,18,36,24,17,0,24,22,8,25,33,26,19,5,10,12,25,40,22,28,26,5,14,6,1,16,27,27,4,9,37,24,11,31,-1,38,2,8,39,15,30,6,37,2,28,25,-1,20,5,32,-1,6,14,38,10,19,2,14,23,36,34,34,3,28,2,8,34,15,5,25,18,33,3,27,7,29,28,8,9,14,25,9,15,28,5,30,-1,28,19,33,22,3,18,-1,16,4,14,24,17,7,3,5,20,7,26,18,22,-2,31,4,30,32,5,38,3,31,8,3,2,11,28,23,40,35,12,37,14,20,37,27,8,17,17,19,6,28,6,19,24,19,7,18,32,31,31,35,16,28,16,36,18,10,24,21,27,9,33,38,25,-1,22,4,9,23,29,25,11,8,26,39,14,38,41,20,30,24,26,39,41,29,-1,41,27,10,23,2,23,21,27,31,36,9,34,16,17,8,-1,22,20,30,34,18,2,15,7,4,27,17,38,20,11,19,9,17,32,36,34,20,10,4,17,37,16,9,23,23,5,34,31,35,32,24,21,24,38,15,33,37,20,28,18,3,1,24,42,7,35,37,10,23,27,37,6,37,20,4,37,32,17,34,5,28,39,23,14,30,31,28,39,28,3,24,6,28,13,21,12,39,19,6,2,18,28,16,29,39,26,31,36,28,7,32,30,20,22,22,21,15,34,3,30,29,16,-2,29,32,6,17,2,3,7,23,12,27,28,29,27,27,39,29,6,33,10,25,8,38,23,15,27,30,25,13,35,9,15,11,21,37,9,30,5,31,10,35,12,33,17,4,20,15,3,33,10,39,23,17,27,14,32,8,20,16,7,2,28,33,38,25,19,37,31,30,23,30,2,34,20,35,15,30,19,28,7,0,4,36,14,11,3,37,31,19,25,26,13,5,29,16,33,-2,29,6,12,32,4,17,24,37,32,3,4,33,33,18,2,22,3,40,36,-1,36,21,10,32,41,28,20,20,16,27,1,5,6,30,30,18,9,40,10,30,4,3,27,23,33,20,32,13,37,32,36,13,29,4,15,32,11,29,4,16,6,34,20,26,12,35,11,32,29,24,22,9,21,14,1,36,32,30,31,15,32,19,16,18,38,38,30,30,26,26,13,12,33,22,25,22,27,36,39,16,10,14,9,15,8,3,6,20,27,35,35,32,20,1,23,12,38,19,24,3,3,27,25,13,26,34,5,5,15,3,29,8,3,20,33,37,4,7,4,17,35,32,12,34,23,17,28,11,30,16,6,32,34,27,24,18,26,31,37,10,27,7,17,32,38,6,6,0,34,2,34,22,21,30,17,1,5,24,9,18,9,14,24,8,14,15,13,3,27,30,22,28,5,3,35,26,37,13,35,29,5,17,1,3,24,25,18,24,20,6,16,17,3,14,25,10,13,21,29,41,33,28,28,2,8,24,18,18,17,16,8,6,7,20,12,13,8,9,5,4,32,27,9,4,14,33,21,26,6,28,6,7,20,7,24,6,7,13,15,32,31,23,37,30,35,16,25,23,16,26,1,11,14,9,17,19,5,12,37,16,10,40,24,-1,10,4,3,3,7,11,17,0,2,25,9,19,38,36,19,17,15,5,20,32,3,7,13,23,14,4,12,8,11,37,1,7,37,12,26,21,24,36,8,2,36,20,25,20,36,18,17,5,37,19,35,6,36,13,30,32,1,41,19,29,18,28,6,13,2,26,27,38,4,10,8,21,22,28,19,27,20,13,17,33,12,1,7,37,8,27,31,39,1,5,33,2,39,23,9,11,18,4,39,35,37,2,16,20,10,22,40,15,26,6,40,13,3,18,38,21,38,34,30,31,10,17,6,35,14,25,36,17,16,5,26,6,9,2,24,14,36,11,21,27,6,25,3,30,26,6,7,3,5,26,28,24,5,35,36,4,9,7,5,38,22,24,12,0,27,25,11,16,12,17,25,32,20,34,14,2,2,35,27,25,10,37,16,29,25,40,28,26,5,8,19,35,34,38,21,33,5,32,13,35,34,26,39,3,29,37,34,12,8,41,31,31,3,10,28,15,3,18,23,37,27,23,18,41,18,16,8,15,29,6,28,22,23,39,14,17,35,39,33,24,33,8,15,18,41,4,4,17,8,30,3,38,11,5,-1,5,7,34,2,25,41,17,21,3,37,29,15,33,13,12,33,21,4,28,16,37,34,20,28,5,25,39,11,16,25,20,16,28,21,10,4,19,9,36,18,35,28,26,8,35,23,19,37,14,39,4,42,9,35,32,30,9,9,4,12,38,4,12,8,34,35,28,12,21,23,26,28,4,39,8,6,25,27,18,10,26,32,32,28,21,20,28,17,26,17,38,30,8,-1,16,16,18,10,33,9,14,21,13,20,37,34,37,12,34,30,5,8,35,20,26,15,29,9,21,32,31,14,25,5,22,15,33,1,25,10,35,10,20,9,29,40,9,35,39,33,27,16,13,40,20,37,26,18,8,13,5,14,29,19,36,20,31,33,28,3,32,23,13,31,38,12,26,20,41,15,24,36,7,14,6,29,28,10,29,10,41,14,32,17,34,23,36,23,36,3,21,14,6,14,38,2,30,6,20,18,29,17,36,5,38,34,24,39,6,7,31,10,34,3,16,35,32,26,5,21,4,6,36,18,1,34,11,15,5,25,7,11,14,38,32,35,12,25,20,37,18,14,9,23,5,11,35,20,10,30,31,36,8,26,29,34,16,38,39,34,34,1,32,32,31,23,6,21,38,20,25,16,3,30,41,22,24,31,37,20,34,5,3,20,20,27,13,8,7,16,12,28,29,2,15,19,32,33,26,26,20,17,19,0,16,10,36,31,19,36,7,20,21,38,24,3,12,36,5,23,22,34,31,8,29,24,3,17,9,14,6,16,27,35,24,21,0,38,14,18,36,37,21,11,5,18,22,26,27,3,9,24,26,11,27,29,3,11,6,13,20,20,22,20,9,32,24,32,10,20,14,33,10,3,32,16,26,36,20,31,10,34,3,19,36,22,39,19,11,20,34,22,7,23,8,23,32,33,30,4,29,13,22,29,9,16,23,19,18,7,2,25,17,29,6,40,18,32,0,10,26,10,8,6,25,25,39,24,13,11,41,6,38,17,22,36,6,24,39,1,30,37,5,17,4,11,3,38,24,30,30,19,19,32,30,33,6,41,7,10,30,27,18,30,24,25,2,7,29,24,33,36,19,14,26,14,-1,17,18,26,30,18,35,16,33,24,11,32,32,9,13,28,21,15,19,38,19,6,10,34,19,29,2,39,39,20,9,24,25,4,31,33,19,36,23,28,33,23,28,15,34,12,21,5,25,20,17,18,4,2,24,19,26,4,38,10,20,26,18,24,38,3,17,2,22,13,3,13,15,11,17,19,26,33,5,1,7,17,31,28,35,8,22,28,5,8,11,24,30,32,22,2,17,4,20,4,23,39,21,38,0,36,21,31,15,19,5,17,15,39,17,34,20,3,29,12,20,33,12,17,3,38,35,37,23,33,10,19,8,24,25,16,20,17,32,25,28,16,38,4,18,15,8,3,32,15,23,8,12,7,3,29,38,31,30,29,5,16,32,37,4,37,28,3,20,27,16,23,9,11,2,40,27,7,7,9,24,12,27,9,29,25,4,23,23,31,21,29,11,35,29,6,5,-1,13,14,5,38,17,32,23,4,34,2,33,36,31,17,7,5,13,17,17,29,2,9,20,19,14,40,10,31,31,35,15,24,15,26,40,11,32,27,4,31,31,6,32,3,3,30,22,38,16,38,14,8,38,11,0,25,35,17,28,13,3,3,0,8,40,29,27,13,11,15,10,5,12,14,7,13,23,33,15,14,8,36,30,2,30,4,23,3,18,1,19,9,27,24,35,35,24,25,16,24,25,37,37,7,12,40,10,33,14,22,37,27,3,40,24,13,22,40,33,20,28,12,24,1,35,23,22,11,20,33,35,12,37,24,5,22,29,17,0,17,28,31,28,11,34,37,14,22,5,32,35,22,9,31,23,33,0,14,3,29,21,11,23,13,13,3,18,2,34,3,1,15,16,3,6,4,23,7,32,14,28,23,22,36,6,11,35,13,23,33,17,15,23,3,35,24,37,41,13,17,21,2,7,13,31,13,12,6,37,22,32,2,26,30,5,34,9,22,15,37,7,11,30,19,25,2,4,5,37,16,21,21,13,8,16,18,31,9,10,37,34,40,40,26,39,25,35,28,31,19,36,9,31,5,19,27,12,1,12,15,13,35,18,4,1,22,24,7,6,10,9,5,30,33,23,5,18,16,9,20,39,37,17,5,2,39,1,9,12,20,28,18,17,10,22,29,36,36,17,21,13,15,32,7,15,22,31,4,22,36,34,10,21,35,0,17,37,39,19,32,25,12,30,36,5,12,7,-1,31,30,11,12,9,29,40,37,25,5,38,15,3,35,23,1,19,26,7,7,25,1,19,29,8,30,17,20,37,5,5,28,9,36,34,-2,16,23,18,7,11,36,32,38,26,28,14,5,30,13,30,4,18,21,36,36,20,14,3,28,14,3,9,9,22,37,9,14,27,2,18,31,14,24,7,42,34,15,11,38,27,32,26,16,5,3,31,37,36,29,23,37,23,25,8,1,1,31,5,30,35,28,29,30,25,25,27,12,7,23,8,30,39,34,-1,25,13,24,15,8,22,15,15,39,32,4,11,19,12,18,5,21,13,16,15,5,33,21,10,4,8,32,34,18,16,36,29,21,27,7,34,9,11,34,26,23,33,36,40,20,7,32,23,23,10,8,23,20,20,18,24,26,27,10,40,22,33,35,18,16,8,26,26,11,32,31,20,-3,8,22,18,28,17,21,36,27,30,6,18,37,20,24,25,9,14,13,26,4,26,13,5,29,38,14,12,20,37,0,8,40,17,0,5,36,26,15,6,4,35,14,41,41,28,29,9,7,39,35,18,33,37,5,30,6,1,20,8,38,32,30,11,27,25,5,2,24,27,30,16,38,36,20,29,19,15,37,32,14,11,5,26,18,27,13,26,17,9,24,4,27,14,7,18,20,18,17,-2,33,21,23,33,34,29,24,38,37,19,13,34,30,5,9,29,31,12,29,31,9,18,29,3,14,8,27,25,29,12,7,37,33,23,25,27,1,11,0,38,36,33,7,8,31,31,5,29,32,24,22,9,6,16,31,37,29,13,20,33,31,35,13,21,22,26,9,9,10,34,20,3,22,20,30,33,26,25,32,39,33,39,26,34,22,18,32,12,27,2,27,16,34,12,28,22,28,1,26,7,9,37,4,32,5,7,36,10,11,11,10,9,35,27,21,22,16,16,1,14,6,33,15,11,11,3,34,36,43,18,-1,21,34,36,29,3,24,11,10,8,11,11,2,18,39,5,41,33,22,33,26,35,17,39,26,23,25,33,-2,32,19,15,40,25,31,14,2,31,25,16,28,29,24,33,32,14,25,36,28,2,1,18,28,30,30,34,32,40,8,20,32,8,22,4,20,12,18,9,20,31,3,10,35,24,26,27,26,35,28,17,6,6,17,4,16,22,13,23,22,36,9,24,3,35,5,32,30,19,9,21,18,11,16,2,8,36,7,29,38,36,41,12,29,36,-2,3,10,27,7,14,18,18,30,25,7,22,18,33,8,10,30,32,-1,28,33,32,28,23,30,7,21,18,25,17,21,40,24,24,18,33,35,34,14,7,13,28,7,36,37,37,25,9,1,31,3,21,34,18,23,32,32,33,20,28,33,37,35,32,20,29,39,17,26,25,22,19,21,24,9,2,8,9,2,5,26,22,16,32,12,33,39,35,25,35,30,4,21,2,29,18,4,36,38,13,14,29,33,15,33,5,26,20,3,8,30,5,15,0,-1,21,13,31,22,10,13,16,0,29,35,33,31,5,31,32,18,5,23,25,18,32,16,40,27,31,25,22,31,4,10,17,15,34,20,9,32,5,4,5,10,16,39,36,11,2,14,17,20,20,32,14,12,14,-2,22,7,5,30,0,34,36,11,26,7,31,8,24,18,10,1,0,22,25,22,19,15,5,1,29,30,-1,28,33,15,4,7,27,4,37,4,38,25,28,16,12,9,10,24,18,24,4,21,17,3,15,33,17,15,10,16,18,25,6,26,5,39,7,15,29,7,10,9,9,23,3,4,20,35,33,7,32,29,16,19,9,35,9,37,15,8,39,16,15,35,16,23,18,21,28,8,7,2,9,40,37,25,20,12,35,14,9,28,4,36,15,14,26,24,22,17,20,6,25,11,26,38,30,37,7,9,35,17,8,30,40,19,8,41,14,31,35,17,34,23,27,9,15,5,8,31,19,6,-1,33,25,21,30,-1,13,36,13,30,25,27,28,-1,20,12,16,37,32,32,34,38,22,11,29,17,13,22,10,23,27,29,25,22,24,13,22,26,41,20,35,25,15,16,35,31,23,27,18,20,10,9,42,9,13,14,10,34,29,9,1,3,9,13,37,19,11,15,41,35,25,3,26,4,23,18,23,23,6,35,19,24,5,4,34,15,8,9,27,18,10,13,18,29,2,31,18,21,10,0,41,38,15,36,22,11,17,10,37,21,3,3,29,7,11,-1,16,28,6,14,29,22,15,40,29,14,30,12,32,21,5,11,37,29,24,36,12,37,12,31,10,41,23,23,35,19,-1,5,15,21,14,38,38,23,9,31,8,29,31,17,17,1,40,18,18,35,13,32,29,18,32,4,30,37,3,21,19,22,31,23,1,37,27,33,3,37,41,6,37,19,2,31,25,27,34,24,30,39,10,6,6,39,26,33,36,18,13,1,4,15,12,9,27,9,28,2,8,13,31,6,23,26,39,8,3,20,39,1,23,9,12,11,11,20,4,9,26,21,6,36,5,35,10,40,4,33,34,33,10,17,7,28,12,12,32,10,27,24,34,0,30,15,20,5,11,19,22,40,20,16,7,10,35,27,16,13,9,11,20,34,25,5,6,8,27,7,15,33,26,4,15,29,18,6,36,30,38,30,9,5,2,16,4,25,3,37,18,15,4,38,12,7,32,3,34,8,14,20,26,11,19,20,13,21,25,38,31,18,15,35,5,6,39,25,11,10,34,15,11,8,20,29,27,29,3,28,21,-3,1,18,26,16,21,20,25,35,33,11,14,22,26,34,35,29,32,33,17,9,14,13,29,21,23,9,27,28,22,8,26,35,7,12,4,9,2,24,19,35,3,13,15,14,25,13,9,16,34,37,33,25,23,11,9,17,27,31,27,35,10,26,12,39,13,33,27,8,35,5,13,12,22,22,33,25,41,11,4,33,18,1,11,22,1,25,20,23,5,29,37,19,8,39,6,17,27,20,6,13,30,3,17,6,8,1,32,19,14,30,19,28,14,35,29,8,14,29,3,39,0,7,19,32,2,30,4,37,25,11,21,23,4,26,24,27,36,37,0,35,16,16,4,16,7,10,36,26,6,3,37,41,27,25,40,5,34,6,28,28,12,7,27,22,17,36,28,27,25,-2,35,25,11,38,8,7,18,22,5,8,11,10,2,24,23,13,37,9,19,30,40,0,27,7,9,33,25,-1,6,19,14,6,39,8,12,38,9,22,9,19,19,11,5,31,4,35,7,27,1,5,18,32,18,23,21,7,21,6,18,30,39,18,24,6,40,34,10,32,30,25,19,16,16,0,16,29,30,12,2,11,4,11,4,24,20,7,16,1,2,43,-2,18,0,17,40,33,5,6,12,24,31,20,34,27,10,21,23,5,28,35,14,0,7,8,26,14,20,3,13,26,15,34,34,8,6,4,15,37,18,23,34,33,13,27,27,2,14,11,33,18,15,24,23,17,33,17,36,26,3,-1,8,12,5,14,14,36,26,7,36,37,15,34,19,37,22,5,32,18,15,7,15,3,11,40,11,5,35,21,23,33,27,18,14,22,39,0,13,8,40,14,36,35,25,32,38,22,6,18,28,24,38,10,21,37,35,34,21,39,32,32,7,31,18,19,21,6,38,32,-1,28,0,35,36,32,14,5,25,15,27,1,21,20,5,32,18,35,24,11,9,21,28,30,9,32,19,3,23,11,31,0,17,2,28,26,29,19,11,34,18,25,32,15,27,6,3,14,21,39,19,37,28,34,32,33,6,17,27,35,4,23,0,12,40,0,35,6,20,36,2,26,28,-1,4,18,6,15,24,23,17,14,34,3,35,2,21,16,20,14,15,22,22,39,0,41,17,9,2,28,20,-2,14,16,5,23,27,30,24,29,34,33,19,21,3,30,23,36,17,11,23,19,12,22,3,38,19,32,31,23,34,28,35,22,15,20,18,18,26,25,30,12,10,34,20,4,8,37,28,10,38,40,24,34,17,7,31,42,9,14,17,18,23,35,3,6,22,29,17,25,2,9,5,4,19,16,15,26,37,37,19,25,25,36,19,19,36,9,33,16,36,23,11,32,9,8,16,23,25,22,40,35,21,27,34,30,22,25,26,27,18,13,7,10,24,36,32,15,7,24,1,12,34,19,36,14,9,21,22,26,20,35,39,33,12,28,16,29,28,27,35,17,17,26,26,31,28,33,33,31,2,16,26,23,23,19,1,22,19,3,38,24,16,31,15,18,1,0,32,13,-3,29,28,17,28,42,21,37,38,34,41,9,9,19,5,37,24,27,21,39,1,11,19,20,8,41,8,15,28,9,26,6,33,23,38,15,0,14,28,24,35,8,4,38,33,37,37,26,26,29,10,19,2,30,11,10,3,7,14,28,1,21,33,33,18,39,3,16,36,16,10,19,14,11,12,43,30,2,4,35,2,28,15,7,25,3,39,10,6,15,10,15,8,12,17,13,8,34,4,20,32,27,12,21,32,26,22,27,0,31,41,5,23,37,23,10,35,25,24,5,35,25,13,40,19,14,-2,10,2,8,35,25,28,34,17,38,38,31,36,10,0,8,7,27,29,12,12,27,38,9,21,35,38,18,7,19,36,23,14,10,25,35,22,34,6,20,41,35,11,24,31,3,29,22,8,22,21,7,13,35,15,10,30,42,14,20,30,25,5,41,18,32,9,9,8,30,-2,14,28,11,9,19,14,4,13,9,16,26,28,4,14,25,21,26,26,26,33,32,21,1,15,13,1,2,30,39,17,22,19,-3,3,32,18,28,21,7,31,5,4,24,11,9,16,23,0,9,7,22,34,17,18,37,36,24,24,32,38,29,38,4,35,24,25,33,29,40,0,23,25,16,19,29,9,19,20,18,33,40,2,32,8,0,20,0,21,6,28,7,20,11,27,30,37,20,32,24,13,-1,26,0,11,32,11,10,7,34,16,18,30,27,7,13,10,26,25,24,12,32,12,7,22,20,23,19,28,13,7,6,38,30,12,41,13,9,36,26,32,20,25,11,21,34,1,1,24,21,8,29,5,17,5,16,19,13,25,6,11,40,7,34,10,13,35,11,35,23,24,12,10,17,7,21,16,11,4,35,19,31,3,37,16,9,28,14,1,26,14,26,23,14,11,13,0,7,0,21,3,25,28,34,39,35,34,3,27,34,17,35,28,13,19,35,22,22,36,7,16,5,27,28,37,36,35,37,1,17,40,0,1,3,39,34,5,6,8,15,26,16,15,8,8,10,17,35,14,27,11,13,23,39,9,25,36,32,38,30,32,33,26,20,23,7,25,18,5,41,17,14,21,33,35,4,2,21,41,25,35,7,25,32,34,15,9,14,2,5,11,2,28,32,14,23,8,27,14,20,9,0,19,28,5,12,22,10,34,11,35,10,21,30,21,0,11,20,20,7,10,11,24,32,6,37,14,16,19,34,12,4,23,34,32,24,22,19,6,17,6,36,21,6,32,31,6,28,7,12,6,4,24,23,39,24,10,30,8,9,35,23,34,6,21,10,11,32,25,39,31,12,19,20,32,31,13,24,8,25,24,7,35,16,24,17,12,27,39,30,22,3,35,0,36,19,29,9,15,32,32,19,13,34,30,26,15,1,28,35,31,19,12,22,20,15,18,26,29,1,0,8,15,40,37,28,9,16,16,22,22,26,4,39,24,5,10,25,16,16,11,5,15,12,39,13,19,33,36,38,27,35,23,0,33,20,8,35,29,22,4,26,0,24,18,26,13,20,5,23,27,3,21,19,37,13,-1,26,27,28,14,20,6,27,5,17,16,32,19,13,28,32,1,33,11,5,15,27,26,5,9,10,26,1,23,6,23,8,0,25,25,5,21,40,3,4,39,40,4,49,34,11,100,45,29,72,22,81,180,105,45,114,40,112,22,130,81,193,21,25,63,62,43,152,29,83,158,48,119,35,29,41,18,50,81,121,71,141,80,84,60,132,45,104,52,43,194,81,49,8,131,72,190,44,139,29,131,93,188,39,125,22,32,105,131,105,45,60,20,115,69,130,85,135,135,12,29,133,19,41,40,82,50,18,109,42,78,51,113,118,29,34,64,121,48,80,161,94,15,74,30,44,22,38,35,69,43,32,49,9,45,82,92,39,81,35,61,73,112,12,38,73,84,78,80,115,82,74,70,81,149,51,39,98,45,49,128,42,18,31,58,48,54,70,124,84,83,163,42,118,132,24,22,58,60,15,83,79,41,134,80,52,70,92,43,21,23,13,112,138,11,171,42,168,85,44,65,24,11,38,10,135,148,63,115,45,159,169,112,93,29,85,21,144,13,165,9,59,158,29,20,68,65,61,91,31,71,73,172,20,55,69,155,94,92,125,74,72,61,25,83,104,41,58,93,39,148,53,71,92,61,38,58,80,71,89,83,28,52,70,71,75,170,78,25,60,120,99,55,29,170,53,138,134,42,99,18,24,55,58,111,33,74,38,23,61,81,129,33,63,70,45,65,74,50,22,30,21,155,64,65,71,40,69,69,45,62,172,24,80,8,38,93,55,64,122,70,81,159,70,150,152,195,22,22,79,33,128,91,39,121,21,41,45,51,165,119,110,20,64,101,101,99,158,89,80,114,49,191,78,28,42,140,45,122,65,68,153,89,91,55,118,22,63,12,50,43,173,60,113,155,91,55,173,71,24,42,141,32,55,153,113,30,29,174,35,98,30,58,25,19,44,20,98,90,61,40,38,28,34,73,133,53,21,40,41,82,100,155,45,9,48,18,80,64,29,48,23,84,179,30,93,39,60,133,125,58,89,82,110,54,89,42,41,38,85,83,33,50,58,115,145,161,38,28,75,35,24,78,113,75,91,31,45,80,50,31,113,29,59,43,122,29,91,65,22,71,60,60,69,48,21,53,29,60,42,69,20,200,141,81,183,149,83,35,13,39,21,10,84,30,31,182,114,152,53,24,150,132,68,53,173,30,113,15,84,88,43,43,90,34,38,173,160,44,83,94,43,42,34,39,70,39,44,69,44,32,15,118,62,82,54,48,74,113,81,43,34,48,54,75,72,39,75,79,205,65,122,82,21,50,39,101,41,132,43,131,11,21,109,141,81,54,164,28,68,44,61,35,54,14,52,101,149,35,21,118,38,24,59,41,85,12,32,79,175,28,41,70,161,53,128,50,80,93,48,149,49,88,22,55,72,38,114,83,101,94,40,128,39,51,79,48,85,41,152,125,11,103,141,38,135,28,70,41,81,170,55,24,81,32,129,108,180,139,33,74,42,73,115,114,92,34,54,64,42,70,51,71,35,40,165,18,32,35,120,31,35,32,85,139,103,152,33,84,61,64,34,82,122,58,92,78,129,13,81,23,35,79,165,52,185,49,64,61,112,72,18,41,61,105,23,22,59,33,11,204,61,103,51,164,12,82,164,48,38,65,71,18,52,48,30,40,112,115,63,85,59,68,84,62,109,141,129,30,148,25,21,72,59,83,10,84,98,32,31,94,21,110,38,39,132,61,92,119,53,18,114,90,140,90,85,80,53,49,152,138,80,82,195,22,81,40,154,49,85,21,14,60,138,98,22,49,102,45,82,81,35,59,52,84,85,65,89,109,81,38,72,78,172,71,165,120,39,23,135,12,92,192,42,158,194,160,98,164,42,202,10,31,58,65,98,150,44,15,82,61,38,39,173,8,51,62,34,132,44,162,64,111,41,52,65,130,45,35,65,38,42,45,145,33,60,45,72,34,23,45,28,81,72,49,31,63,150,142,74,24,29,79,94,121,54,73,71,29,93,73,19,58,39,23,29,155,90,113,42,30,19,124,30,28,15,70,10,22,70,61,161,29,35,48,44,88,40,21,58,59,74,84,58,154,185,78,55,30,74,118,182,82,29,120,53,43,59,84,161,21,63,172,51,24,95,59,51,28,154,15,55,149,79,68,78,34,195,184,85,200,64,88,42,91,38,28,32,31,22,95,74,22,43,43,112,9,81,131,19,62,19,8,193,129,30,80,22,33,64,60,81,45,112,83,180,28,169,14,43,88,78,68,144,30,120,24,78,135,44,73,90,58,75,174,50,85,54,83,24,45,68,48,114,142,52,142,15,118,101,84,32,64,81,28,93,185,42,45,49,69,94,18,8,55,85,62,65,138,103,99,152,25,98,39,40,58,192,62,69,40,61,30,29,41,118,109,122,50,43,108,110,20,112,25,42,29,84,81,55,40,145,175,78,51,50,21,79,84,34,73,90,23,145,54,49,51,8,62,25,63,24,11,23,33,83,39,138,113,125,195,105,75,40,61,75,122,75,59,24,175,39,145,24,20,85,65,191,11,40,54,95,94,79,48,70,40,44,43,29,188,114,52,13,95,84,49,181,182,70,79,61,121,34,183,51,70,39,84,122,145,98,79,38,49,38,91,29,172,104,58,171,73,191,90,55,28,42,9,65,134,83,38,13,143,75,120,91,99,71,111,108,158,81,12,21,55,42,81,13,132,99,23,40,181,112,138,94,113,112,190,38,32,104,43,45,35,42,58,103,71,14,98,90,65,19,35,71,31,43,24,58,115,168,128,138,81,41,123,71,8,42,62,22,38,24,50,190,94,165,43,48,52,84,131,44,78,15,38,84,122,98,80,165,121,20,11,45,118,14,30,45,80,25,144,84,22,33,60,81,169,63,12,15,39,44,110,130,41,59,39,35,80,81,31,93,45,40,80,20,41,31,35,40,84,53,85,138,71,113,23,62,14,60,94,83,130,61,92,20,194,74,84,98,52,55,64,128,113,50,54,129,82,100,44,164,81,71,28,80,90,74,14,23,153,38,112,145,28,31,195,50,71,21,35,42,78,73,53,51,40,178,35,192,123,48,45,15,79,63,30,145,32,64,84,21,99,35,180,42,179,42,180,82,55,79,48,93,23,50,85,35,171,29,84,20,14,179,48,74,53,83,171,85,38,85,79,69,43,33,42,40,43,20,65,139,135,84,179,45,109,34,162,105,22,82,105,62,82,72,38,25,20,80,84,33,55,33,190,43,45,141,69,78,84,172,51,75,183,20,101,129,63,44,184,100,12,68,48,82,52,154,39,40,54,32,64,92,40,60,85,21,35,52,128,195,68,55,55,80,123,90,61,41,43,71,25,154,141,22,145,49,109,80,82,52,85,148,160,39,25,33,38,85,48,15,94,120,60,145,29,78,59,80,180,8,23,79,78,64,64,160,43,32,179,8,63,40,99,141,159,39,99,18,38,133,84,60,178,18,73,125,91,74,35,113,52,191,18,25,50,75,92,33,65,44,183,175,28,49,45,64,22,120,175,101,104,155,58,80,45,134,58,20,25,55,155,30,39,73,41,34,21,11,154,19,101,24,131,105,52,21,45,39,192,195,83,34,49,60,41,72,102,31,73,89,19,63,104,40,92,72,44,114,73,63,69,42,88,98,65,12,122,12,24,170,184,25,131,50,21,55,124,58,72,192,83,93,38,55,11,85,124,129,180,74,138,111,83,33,21,35,105,103,75,33,52,89,63,84,44,29,29,22,53,24,153,31,21,180,82,90,154,99,153,93,53,18,70,100,193,125,68,55,152,84,13,33,89,93,68,85,73,83,158,182,24,102,53,175,28,14,139,34,64,84,42,63,100,190,22,20,21,25,20,173,81,139,119,74,42,73,114,44,141,14,33,83,89,39,83,108,21,102,48,32,43,35,184,140,32,149,15,71,108,122,40,198,79,84,31,12,88,29,201,20,15,155,25,200,101,19,188,8,59,72,55,24,19,71,54,28,25,51,125,88,125,79,31,131,70,38,83,59,121,42,32,29,40,25,21,79,52,61,25,155,25,80,55,23,60,54,44,33,45,55,134,44,149,22,74,128,62,9,9,15,28,83,73,42,34,32,68,82,19,60,192,119,190,34,44,23,171,43,139,118,113,98,70,42,143,185,28,45,78,121,59,64,15,10,61,50,79,35,11,128,55,79,18,91,9,62,59,25,54,112,125,194,161,35,113,25,45,119,30,79,178,23,171,74,103,99,43,23,91,93,154,21,65,15,52,30,42,20,60,54,60,41,30,21,105,65,20,179,161,82,43,61,90,90,21,118,44,110,99,98,69,82,112,123,78,59,78,49,82,125,185,163,19,115,81,93,111,184,50,55,13,29,72,83,98,43,34,10,201,85,50,91,112,98,115,22,101,83,52,191,134,54,69,64,53,118,19,31,63,25,22,78,81,41,62,35,30,51,140,73,40,69,191,82,181,38,23,55,58,29,39,39,30,58,62,62,19,38,84,78,22,81,32,89,121,42,130,30,114,155,50,34,94,114,89,49,115,148,72,114,11,63,113,80,54,32,11,141,84,18,20,55,31,114,52,18,55,29,30,38,122,183,41,38,61,103,80,22,128,142,124,44,170,75,78,31,99,41,39,51,52,169,141,93,14,160,44,33,103,71,55,113,79,80,179,9,61,111,80,32,83,91,29,19,52,150,32,91,180,121,25,102,44,161,134,43,94,29,38,120,85,89,25,20,125,18,173,34,159,55,32,65,83,30,180,13,35,95,83,39,54,81,53,119,160,21,40,69,45,55,38,44,45,84,188,29,95,79,41,19,75,64,174,54,14,94,10,145,203,81,153,58,49,132,85,85,178,104,40,103,25,69,30,70,15,125,50,44,38,64,9,44,93,83,179,40,58,74,15,11,41,68,28,115,83,40,113,89,38,62,65,168,40,30,20,58,150,54,19,30,123,45,154,93,71,83,99,55,38,74,69,104,38,25,162,64,30,52,42,79,34,123,160,51,108,158,68,42,32,22,29,183,69,92,40,9,128,110,171,13,25,123,33,189,34,58,51,29,111,43,130,20,101,48,180,92,33,51,31,83,45,89,193,89,162,9,58,80,73,41,74,29,25,98,59,43,72,183,134,140,39,35,141,23,134,35,42,81,14,45,79,53,33,190,60,8,38,154,54,41,25,53,79,13,130,93,82,15,150,131,28,9,88,143,168,105,62,51,101,90,83,21,115,172,32,204,34,30,31,90,79,22,114,42,29,72,68,78,90,170,14,129,39,82,31,9,15,70,84,155,89,170,185,82,60,58,29,32,115,83,54,112,182,112,129,23,198,84,39,129,113,35,85,73,70,130,39,72,68,35,61,69,59,99,201,130,72,44,91,92,55,20,89,59,8,85,94,22,74,59,12,8,74,184,34,63,123,85,109,31,61,88,22,63,80,78,45,51,111,45,184,41,69,23,160,61,110,50,49,185,52,53,125,72,51,23,41,41,138,138,14,145,25,43,93,90,41,169,63,23,140,75,13,10,100,85,73,130,65,59,31,134,11,32,53,28,44,40,40,163,82,89,78,55,34,108,58,73,54,45,123,93,102,39,114,29,30,22,38,41,161,38,25,70,201,42,64,28,119,28,60,23,164,39,19,72,41,32,69,23,44,32,25,80,83,133,89,81,44,119,39,40,79,30,179,178,82,50,64,129,68,30,48,130,40,105,41,168,80,134,49,63,70,32,14,111,38,121,90,178,20,61,43,39,60,63,8,52,92,111,71,25,34,74,120,50,10,45,29,145,100,24,60,50,43,79,44,31,14,19,111,90,21,104,75,175,98,41,171,79,24,90,39,9,15,80,63,20,43,65,91,53,110,82,23,172,88,71,180,39,94,25,10,42,142,34,139,154,40,33,80,10,41,30,78,164,23,149,25,42,71,114,70,149,55,29,123,152,8,69,161,79,105,39,163,9,71,51,32,161,60,35,41,73,49,79,120,60,78,40,110,160,68,49,65,55,62,152,42,39,111,82,108,59,61,78,18,113,62,81,13,75,33,179,69,45,59,81,133,81,14,28,93,191,155,155,68,52,51,44,121,49,42,144,72,44,43,74,145,158,78,154,105,100,58,113,10,142,128,131,88,101,70,122,70,148,43,31,35,81,19,98,101,81,135,34,104,29,119,82,62,85,8,78,44,122,22,39,55,144,71,191,22,158,131,75,69,39,34,32,129,158,54,34,45,68,53,32,173,31,72,30,138,18,69,58,63,43,28,38,72,64,70,12,159,103,42,48,49,89,22,10,38,49,30,82,82,24,32,34,195,61,22,40,50,35,64,49,44,65,13,84,28,108,59,163,183,69,115,130,185,118,163,140,91,61,168,69,22,53,25,115,31,83,152,30,34,44,182,44,122,35,91,30,53,28,85,28,41,154,41,33,182,59,129,35,185,45,62,183,14,53,48,128,105,40,10,102,31,30,21,44,48,21,84,64,82,39,45,133,152,12,32,74,54,181,75,95,190,90,20,64,85,105,21,78,158,125,81,183,54,79,38,172,184,158,188,20,60,113,70,84,23,141,110,31,22,91,82,23,35,114,18,80,74,122,73,40,55,69,92,48,113,58,109,39,33,18,63,43,70,19,39,81,34,140,114,55,21,50,41,190,8,55,40,22,44,102,54,130,75,42,133,40,34,142,64,44,95,49,12,98,42,44,124,175,22,78,54,154,42,195,80,81,89,61,25,43,122,160,83,79,30,145,151,14,19,55,132,182,78,42,44,195,53,65,43,38,50,81,41,73,31,84,95,162,60,121,43,38,82,52,43,82,161,24,102,53,85,111,90,94,63,153,39,205,142,163,113,64,20,179,91,103,80,38,44,164,81,95,44,88,148,169,11,92,74,72,63,172,183,40,68,78,68,49,39,151,159,89,105,78,128,81,50,38,119,110,29,19,130,38,31,14,158,39,141,33,92,78,42,15,48,63,149,49,125,58,48,75,114,134,111,62,45,113,128,168,81,75,21,83,121,40,101,90,68,74,74,51,149,31,85,62,44,63,23,40,78,40,78,129,71,23,15,28,23,19,42,113,34,19,43,179,44,21,115,81,91,30,55,48,23,21,55,21,40,30,78,65,50,55,111,21,65,114,54,75,13,54,22,198,18,74,61,64,10,82,61,83,73,22,91,65,13,103,95,31,131,175,38,104,114,38,30,19,25,124,20,83,18,99,62,24,13,64,62,158,88,28,52,82,140,80,58,180,55,65,62,23,35,35,74,14,83,71,53,103,109,60,12,141,128,98,41,90,31,74,83,35,140,65,35,73,29,55,20,70,81,25,119,49,38,80,84,61,173,60,83,59,32,33,19,14,94,154,33,28,44,81,29,58,38,82,18,45,64,48,24,65,111,114,39,19,22,49,40,63,55,33,31,24,81,81,113,92,81,101,78,78,35,9,82,41,84,42,54,84,131,85,84,114,61,39,21,111,68,179,93,85,180,110,65,159,81,132,44,98,62,91,29,101,65,58,32,140,44,113,38,13,62,29,141,132,23,11,90,43,59,28,28,108,78,65,25,108,43,95,24,190,24,41,80,63,79,105,153,114,120,104,20,41,35,174,164,22,132,32,21,49,79,30,118,11,61,43,141,54,33,84,125,110,14,65,78,18,95,95,191,19,19,39,93,115,112,40,45,133,85,18,54,41,21,152,53,163,84,59,33,81,182,61,88,170,83,44,83,21,143,103,135,162,73,142,25,115,88,73,94,58,81,113,35,65,170,62,54,39,28,95,82,21,50,34,19,71,113,9,118,79,72,18,93,62,23,40,34,114,125,41,38,45,12,115,175,24,28,32,63,32,60,53,58,42,39,72,80,39,18,30,54,128,38,120,25,85,135,14,70,23,31,132,28,125,91,191,88,84,64,28,149,172,42,53,149,113,79,175,38,120,52,42,13,141,131,133,64,49,34,91,195,53,39,20,104,51,154,38,12,28,33,31,44,49,134,182,114,90,105,58,53,15,103,58,83,125,61,69,89,70,39,41,148,38,95,129,84,29,30,90,150,15,13,9,24,41,25,49,58,141,50,61,65,112,19,23,60,30,144,118,194,109,174,34,84,41,92,19,55,22,155,72,81,30,54,60,68,65,31,169,139,29,43,80,43,194,24,32,153,30,165,64,15,70,32,128,84,62,33,65,33,15,45,132,29,65,52,64,73,20,44,79,104,88,51,61,23,38,75,42,45,59,130,9,62,202,162,93,145,99,148,154,149,69,30,48,42,132,81,60,45,70,71,28,83,131,82,79,51,111,93,73,24,64,104,38,191,53,74,21,80,41,79,141,32,38,85,93,140,69,32,61,125,30,104,24,22,59,120,164,60,43,70,51,59,129,22,29,18,65,65,38,110,22,43,30,134,32,139,45,62,51,93,122,45,179,64,41,58,64,28,21,60,78,59,59,83,128,95,160,78,184,30,51,35,74,43,23,20,112,74,65,124,65,73,8,53,131,13,60,45,39,78,43,31,118,82,112,10,18,72,72,78,78,30,45,80,174,51,181,78,79,119,49,83,100,70,52,12,82,34,112,42,83,142,199,29,70,25,84,81,15,54,108,83,62,134,75,109,152,83,52,31,135,98,14,64,109,118,80,51,115,83,90,28,82,24,71,80,109,140,39,39,50,31,35,44,35,20,10,83,203,84,41,164,152,62,12,28,52,19,35,28,70,140,102,50,61,188,178,44,12,69,132,128,44,44,34,132,158,83,28,131,31,40,71,42,35,81,91,129,75,54,43,94,84,34,15,99,15,83,83,143,81,39,158,62,64,29,64,21,142,64,40,61,44,78,25,125,63,54,92,119,40,41,81,101,25,55,43,98,112,69,79,118,81,19,139,63,102,40,44,59,224,64,129,43,12,53,39,65,18,55,60,42,44,111,62,44,69,62,35,91,12,89,65,179,82,48,20,91,122,22,43,38,61,33,145,170,55,40,68,53,63,123,13,12,53,79,89,188,62,123,40,119,123,23,62,69,124,52,134,58,45,75,23,123,48,145,31,38,78,94,84,22,78,75,71,24,112,53,41,25,42,51,19,90,89,22,119,93,18,32,182,19,85,90,59,84,95,13,88,34,11,80,41,90,93,81,60,39,22,32,28,35,154,189,40,88,124,23,23,139,173,42,161,28,191,42,81,175,82,53,142,42,64,90,44,29,93,69,82,180,13,72,54,104,44,40,190,64,9,38,72,82,48,82,14,162,54,90,64,65,113,38,65,39,174,59,43,24,50,68,61,21,21,31,70,25,78,19,23,40,22,71,58,75,40,155,83,99,59,53,11,179,71,85,49,122,171,138,8,59,49,92,75,60,41,40,81,52,25,139,54,81,73,30,59,21,65,41,52,84,135,18,34,85,23,65,195,40,141,95,23,82,53,10,75,23,133,129,85,168,83,52,114,81,32,63,70,20,138,63,85,180,80,53,72,123,91,148,163,28,193,22,74,70,11,61,41,108,23,135,30,144,139,143,44,49,75,43,59,154,75,75,88,91,11,22,49,140,99,82,10,109,45,93,89,39,21,31,31,52,13,21,50,81,74,89,82,88,61,139,29,51,58,21,43,104,9,39,89,21,173,29,43,30,39,83,58,8,39,204,45,111,83,83,115,60,53,134,91,91,128,78,161,60,161,40,68,74,55,145,24,10,138,63,54,62,81,61,68,78,165,48,155,158,52,18,42,18,40,44,11,194,108,12,19,150,89,83,79,102,155,138,51,80,135,34,195,62,173,149,20,42,28,95,95,34,191,41,150,91,84,73,122,73,20,130,85,134,98,54,41,128,132,188,170,14,93,79,79,62,95,60,75,85,143,34,44,31,13,52,42,102,143,10,44,158,12,59,92,82,44,182,121,38,92,53,45,63,181,184,33,22,99,44,101,32,59,25,140,29,35,141,75,58,41,42,55,18,75,85,43,149,25,19,18,13,41,139,62,51,71,145,45,53,64,39,38,85,61,23,122,72,123,40,62,113,81,59,25,25,14,19,61,21,44,42,52,112,34,83,24,71,181,75,153,75,19,31,178,65,8,92,145,85,149,9,164,40,54,164,140,18,12,140,24,60,50,44,51,34,33,63,22,104,75,75,62,48,73,135,61,78,40,44,45,18,53,11,38,28,35,81,90,115,78,30,39,69,29,54,60,99,51,53,111,79,90,18,73,59,40,41,53,128,145,35,28,155,195,50,48,81,50,21,142,64,91,74,182,82,45,85,133,22,24,38,80,33,41,90,89,8,85,53,20,9,25,114,41,49,12,50,30,32,41,25,29,50,79,110,40,18,48,40,19,35,133,19,83,39,38,63,85,14,64,124,20,33,80,35,74,32,73,41,18,28,64,44,109,71,79,30,82,101,13,59,64,188,28,120,131,19,44,49,99,95,81,73,193,115,41,80,90,41,102,59,69,83,49,180,61,59,85,13,151,44,53,133,31,101,61,68,21,149,75,55,81,179,81,42,199,18,28,80,79,52,18,52,15,84,114,41,29,23,84,60,13,45,83,102,58,134,130,148,92,142,62,50,29,85,78,85,89,33,42,31,143,35,22,45,119,43,75,121,63,58,184,155,69,188,38,9,69,199,38,129,59,115,83,65,72,72,63,64,194,104,180,18,93,32,68,38,115,161,154,68,44,149,103,41,113,48,69,43,19,65,32,20,25,70,70,103,162,61,170,49,108,62,102,141,8,91,10,35,71,59,14,122,81,35,83,121,35,140,88,70,63,125,21,75,38,58,31,18,29,29,69,121,64,84,65,85,44,174,178,55,25,21,72,129,18,91,38,21,38,154,178,85,93,151,78,135,40,80,18,74,94,173,88,82,54,45,39,20,93,85,95,48,22,33,40,30,22,20,49,25,150,32,9,23,18,42,39,84,101,43,28,59,29,39,42,24,62,33,73,88,58,48,40,160,39,53,32,32,11,123,184,60,41,99,83,134,85,41,44,62,149,115,25,84,81,181,62,11,60,42,83,109,70,25,72,183,44,52,18,40,174,41,81,128,135,65,145,65,119,124,33,41,105,130,80,115,62,165,34,18,44,25,52,133,71,61,62,33,28,53,69,165,75,54,142,53,41,43,79,24,73,129,50,28,64,49,121,58,51,61,35,39,74,201,81,44,19,85,74,54,58,18,88,19,49,101,138,25,120,153,28,110,94,48,123,33,42,68,28,42,28,73,89,82,83,121,28,39,110,55,111,94,85,81,94,162,80,70,19,112,109,12,33,49,42,41,43,64,19,65,53,32,22,39,59,50,19,55,81,108,122,98,110,33,34,69,41,60,95,73,41,68,64,29,31,54,63,92,135,195,45,72,98,30,78,43,85,38,58,92,13,218,75,40,15,24,49,83,4,3,1,1,4,4,2,1,3,1,4,3,2,4,1,1,4,4,2,1,4,3,1,2,2,3,4,1,1,1,1,1,2,3,4,3,1,1,3,4,3,3,4,1,1,4,3,4,2,1,4,1,1,3,1,2,3,2,2,2,3,1,1,4,2,1,2,4,3,4,1,4,1,1,2,4,3,3,2,2,4,3,1,1,1,4,4,3,2,4,4,4,3,2,1,4,1,1,1,3,1,3,3,2,4,3,4,1,1,1,3,4,3,2,3,3,1,2,3,1,2,3,2,1,4,3,2,1,2,3,3,4,2,3,3,4,4,2,3,1,2,3,1,4,2,1,1,4,2,3,2,3,1,2,4,4,4,4,2,4,4,2,1,4,2,1,3,3,4,4,2,3,2,4,2,4,3,4,4,1,2,4,1,3,2,1,1,3,2,2,1,1,1,4,2,4,1,4,4,1,1,2,4,4,2,4,1,4,4,4,2,1,2,1,1,2,2,3,3,1,2,2,3,3,4,1,2,2,4,2,1,4,3,4,1,4,3,1,3,4,1,1,1,3,4,2,2,4,2,3,3,2,3,1,3,2,4,3,4,1,1,2,1,4,2,4,2,1,1,1,4,4,4,3,4,4,4,4,4,2,4,4,4,3,3,3,3,3,2,3,1,3,4,4,1,2,1,3,2,1,4,2,2,4,1,1,3,1,2,1,4,1,1,3,2,3,1,2,4,3,1,3,1,1,4,1,1,2,3,1,4,3,1,1,1,2,3,1,2,4,1,4,2,1,3,4,3,2,4,2,1,4,4,4,2,3,2,1,2,1,4,2,2,2,3,1,1,2,2,2,4,4,4,4,2,3,3,4,2,2,1,2,4,1,4,3,4,4,4,1,1,3,4,4,4,2,4,3,2,4,3,4,1,2,3,1,1,1,2,1,2,4,4,3,3,4,2,4,4,4,4,4,2,2,1,4,3,3,2,4,1,3,1,4,1,3,3,4,2,4,2,1,1,1,2,3,1,1,4,3,1,2,3,4,3,4,3,3,1,2,2,2,3,2,4,1,2,1,1,2,4,4,4,1,2,3,1,4,2,2,3,1,1,3,2,1,2,4,3,2,2,4,4,1,2,4,2,4,1,3,2,3,1,4,3,3,3,3,2,2,1,4,3,1,3,1,3,3,1,1,4,4,1,4,4,2,1,2,4,1,2,4,2,1,2,1,3,1,1,1,4,2,3,3,2,4,3,3,1,3,3,1,4,2,2,4,3,2,4,4,1,4,1,4,1,3,3,3,2,3,3,4,1,3,2,1,3,2,3,1,4,4,1,2,1,4,2,1,2,1,4,1,4,4,1,1,1,2,2,2,2,1,2,2,4,1,3,1,4,4,3,4,1,1,1,2,2,4,2,1,4,2,2,1,2,2,1,3,1,4,3,3,4,1,3,3,2,4,3,2,1,2,1,2,2,4,4,4,1,4,1,3,1,2,4,2,1,1,3,4,2,2,3,3,1,4,1,3,3,4,4,1,1,1,3,1,1,2,4,3,3,1,1,1,1,2,4,2,4,1,2,3,1,2,1,1,2,1,1,1,2,3,3,2,1,3,1,3,3,4,3,3,2,3,3,1,4,3,2,3,1,1,3,4,1,2,4,3,2,1,1,2,1,1,2,4,3,2,4,2,1,1,1,4,2,3,4,3,1,4,2,1,4,1,2,4,4,3,1,2,3,3,1,1,2,4,1,1,2,4,2,3,1,2,1,3,2,2,2,2,4,3,3,2,2,2,2,3,4,4,4,4,1,2,2,2,2,3,1,4,1,4,3,4,4,3,4,1,3,1,2,4,4,1,2,1,3,2,4,4,1,4,3,1,4,3,3,2,4,4,3,4,4,3,4,2,3,1,4,4,2,1,3,3,1,3,1,1,4,2,1,2,1,1,1,2,1,4,3,3,2,1,2,1,1,3,4,1,2,2,3,4,4,3,4,4,2,4,4,4,1,2,1,1,2,4,3,2,2,2,4,4,2,3,4,4,3,4,4,3,4,4,2,1,4,1,2,2,4,1,1,4,1,2,3,1,2,2,3,1,3,3,1,1,4,1,4,3,4,1,3,1,3,2,2,1,1,3,4,2,1,4,4,4,4,1,4,3,1,3,4,2,2,4,2,4,1,2,4,1,2,4,1,2,4,2,2,4,1,3,2,3,2,1,3,2,4,4,1,4,1,3,3,2,1,1,4,2,1,3,4,4,1,1,1,2,2,3,1,1,4,4,2,4,4,2,1,3,1,2,2,2,1,3,2,1,1,4,2,4,1,2,1,1,4,3,3,1,4,1,4,1,3,2,3,4,4,1,4,2,4,4,4,3,1,1,4,4,3,4,2,4,3,1,4,1,1,2,2,1,1,1,1,3,3,3,4,1,3,2,2,3,2,1,3,4,3,3,2,2,4,3,1,3,4,2,2,3,1,4,4,3,4,3,4,1,1,3,1,3,1,3,1,1,1,1,2,3,4,2,3,1,1,3,1,3,1,1,2,1,2,3,1,4,4,2,2,1,2,3,1,2,3,1,1,1,1,4,4,3,4,3,4,1,1,2,1,1,1,1,3,2,2,4,1,2,4,1,1,4,1,3,2,2,3,1,4,2,1,1,3,1,3,2,4,1,4,1,1,3,1,4,4,3,4,3,4,1,2,2,2,1,1,3,4,2,2,1,2,2,4,4,4,3,4,2,4,4,1,3,4,1,4,4,4,4,1,1,2,3,2,2,3,1,1,2,2,3,1,3,4,1,4,2,3,4,4,1,2,2,4,1,4,1,3,1,1,2,4,1,1,2,4,4,2,2,1,1,1,4,3,1,2,4,2,2,1,2,3,4,2,3,3,3,4,2,2,2,1,2,3,3,1,4,1,4,3,4,1,4,4,4,4,2,1,1,3,4,1,2,2,1,3,2,1,2,4,1,3,4,3,2,1,4,4,2,2,2,1,2,1,1,2,3,1,1,4,2,4,3,2,1,1,4,2,1,1,4,4,2,4,2,2,2,1,2,3,2,1,3,4,1,4,1,1,1,1,1,1,2,4,4,3,1,4,3,2,3,4,1,4,3,3,3,3,1,4,2,4,1,2,3,2,4,1,2,4,4,4,1,4,3,1,3,2,4,1,3,1,4,4,1,1,4,1,1,4,2,1,2,4,1,2,3,2,1,2,2,2,1,1,4,2,2,2,1,3,2,4,1,2,1,1,2,2,1,4,1,4,1,3,2,3,4,1,1,2,2,3,1,3,4,4,3,1,2,4,2,1,3,4,1,4,1,2,1,2,2,3,4,2,1,3,1,2,2,1,4,4,1,1,2,1,1,2,4,2,1,2,4,4,2,1,2,1,3,1,1,1,1,2,2,1,4,2,1,2,4,4,2,2,3,3,2,2,4,4,4,4,3,1,2,4,1,4,1,4,1,1,4,1,4,1,1,1,4,4,4,4,3,4,2,4,2,1,2,1,1,4,3,1,2,1,3,3,4,4,4,3,4,3,1,1,4,4,3,1,2,2,3,4,4,4,1,3,1,2,1,2,4,2,2,4,4,4,2,2,2,3,1,4,2,1,3,1,3,4,3,2,1,2,1,4,1,2,3,4,4,2,1,1,2,1,4,1,1,4,4,3,1,2,4,2,1,2,2,2,3,4,4,2,1,2,1,3,1,4,3,1,2,1,4,3,2,4,4,1,4,2,1,4,2,4,2,2,3,2,3,4,1,3,1,4,3,1,3,2,1,1,2,4,4,3,2,2,2,4,1,2,4,2,2,2,2,2,4,4,3,2,3,1,1,3,1,4,1,2,3,1,4,3,4,2,3,2,4,1,3,4,2,2,1,1,1,2,3,2,1,4,2,1,2,4,1,3,1,2,2,1,4,2,3,2,2,4,3,2,1,3,1,1,1,2,4,4,2,3,4,3,2,4,2,1,2,4,4,2,1,2,4,4,2,4,1,3,2,2,3,4,4,1,2,2,4,3,1,4,2,3,1,1,2,2,1,3,2,1,4,4,4,1,1,1,3,3,1,2,1,3,4,3,1,2,1,4,2,1,2,4,4,1,4,1,3,3,2,2,4,2,1,1,4,1,1,4,1,1,2,2,3,2,4,4,1,2,2,4,3,1,4,4,1,1,1,3,2,1,3,3,1,4,4,4,4,3,3,2,1,3,2,3,1,1,1,2,4,1,2,1,2,1,4,3,3,3,3,1,1,3,3,4,3,3,1,1,4,4,2,1,4,1,4,3,1,4,4,1,1,4,1,4,2,3,3,2,2,2,1,4,3,2,3,3,4,1,3,3,4,2,2,2,1,1,1,1,1,1,3,1,1,3,2,3,4,1,1,3,1,1,2,1,2,2,4,3,3,4,1,4,1,1,4,1,4,4,3,4,2,1,1,4,1,1,2,1,3,2,3,1,1,1,4,4,2,4,2,2,3,2,1,1,3,3,2,4,4,2,3,1,1,1,3,2,2,1,1,1,1,3,2,1,2,3,1,1,2,4,3,1,1,2,4,1,3,3,1,4,3,2,1,3,4,4,1,4,3,2,2,2,4,2,3,3,1,4,4,3,2,1,2,3,3,1,2,1,4,3,3,1,4,2,4,1,2,2,1,3,2,1,2,4,4,3,2,4,3,2,1,2,4,3,1,4,1,1,4,1,4,2,1,4,1,4,1,4,3,1,2,3,1,1,1,4,4,3,4,2,3,1,2,1,1,2,2,4,4,1,4,3,1,3,1,1,2,2,3,3,3,2,3,1,4,3,1,3,4,4,2,1,4,3,1,2,1,1,1,1,4,2,1,1,4,4,3,3,3,1,4,2,1,1,2,3,1,1,1,2,4,4,1,3,4,3,4,2,4,3,1,4,2,3,1,2,1,2,2,3,1,2,1,2,1,2,2,2,1,2,2,4,1,2,2,4,2,1,4,4,1,1,1,2,2,4,1,2,2,4,1,2,3,2,2,1,1,1,3,1,4,2,4,1,2,1,3,2,4,1,4,4,4,1,1,2,1,4,2,1,1,1,4,4,4,1,3,1,2,3,1,1,3,1,4,3,2,2,1,4,2,2,3,2,4,2,3,4,4,1,2,1,3,4,3,3,4,2,1,4,2,2,4,2,3,2,4,4,1,1,1,3,1,4,1,2,2,2,3,2,3,4,3,3,4,1,2,3,2,2,3,2,1,2,4,2,3,4,2,2,3,1,2,4,3,1,2,1,1,4,1,3,1,4,4,1,3,4,1,2,1,2,3,1,1,3,1,3,1,2,3,4,1,2,2,3,4,1,3,4,4,2,2,2,3,1,2,2,2,4,4,3,4,4,2,3,2,2,4,3,2,1,2,2,1,4,2,2,4,3,4,3,1,3,1,2,4,2,3,1,1,2,2,2,2,3,4,1,4,1,1,3,3,2,3,4,2,1,4,1,2,1,2,1,3,2,1,4,4,3,4,4,4,4,4,4,1,1,4,3,4,1,3,1,2,2,4,3,1,4,2,2,1,4,1,1,3,1,4,2,2,2,3,2,1,3,4,3,2,2,1,1,1,2,2,2,4,2,3,4,1,2,4,2,2,1,4,1,3,1,1,2,4,3,2,2,1,2,4,3,3,2,4,2,4,2,4,2,4,1,4,3,4,4,4,2,1,2,1,3,2,4,2,4,4,2,2,3,1,4,3,3,3,4,2,4,1,1,2,1,1,2,1,4,2,4,4,1,3,2,2,1,1,1,2,1,2,1,3,3,4,2,2,2,1,2,1,3,2,1,2,2,1,2,2,1,2,3,3,4,1,1,1,1,4,1,1,3,1,4,4,1,1,1,4,4,4,1,1,3,1,1,4,3,1,4,1,4,4,2,4,2,2,2,3,4,2,4,4,2,2,3,3,1,3,4,3,3,4,2,3,4,2,4,1,1,3,1,2,1,3,1,3,2,1,2,4,2,2,3,1,2,4,2,1,4,2,4,2,1,1,1,3,1,1,2,4,1,2,3,1,1,2,2,1,3,4,2,2,1,3,2,2,1,2,4,1,4,4,2,1,2,1,3,2,2,1,3,2,4,3,2,3,4,1,2,1,2,1,1,3,1,3,2,4,1,1,3,1,1,4,2,1,1,2,4,2,4,1,1,1,3,4,2,1,1,3,4,3,1,2,2,1,2,1,2,4,2,1,2,2,1,1,1,4,3,2,4,3,4,2,2,3,3,4,2,4,1,4,3,1,1,2,4,1,4,2,2,1,2,2,4,4,3,1,1,2,4,4,1,2,1,1,1,2,3,2,4,1,2,1,1,4,2,4,1,1,3,3,2,2,3,3,4,4,3,2,1,3,2,1,3,2,2,3,4,1,2,1,3,4,3,1,4,2,4,2,1,3,4,2,3,2,3,4,1,4,3,2,4,1,2,2,1,3,1,4,4,1,4,3,2,1,4,4,4,4,2,2,2,3,4,1,1,4,1,1,1,3,2,2,4,3,2,4,1,4,4,4,1,2,3,3,4,1,1,4,1,4,3,4,1,1,2,1,4,3,3,2,2,2,3,2,3,4,4,3,2,1,1,3,1,3,2,1,1,2,1,4,2,1,2,4,4,2,1,3,2,1,3,2,3,4,3,1,1,4,2,3,1,2,2,3,1,2,1,3,1,2,4,3,3,4,4,1,2,1,3,2,3,2,4,2,2,3,1,1,4,3,1,1,2,1,3,1,3,2,2,4,2,2,2,1,2,3,1,3,1,2,1,4,2,1,1,3,2,3,3,3,3,1,1,3,3,3,2,2,1,4,4,2,1,2,1,2,4,4,2,2,4,4,3,2,3,1,1,2,1,1,4,3,4,2,3,2,3,3,2,1,3,1,1,2,1,3,2,2,4,4,4,1,2,1,4,1,1,4,3,1,4,1,2,1,2,4,2,1,1,2,2,1,3,2,3,2,3,4,3,3,3,2,1,3,1,3,2,4,1,1,3,3,2,1,4,2,4,3,2,4,3,2,3,3,4,3,2,4,4,2,3,3,1,3,2,4,2,3,2,1,4,4,2,2,2,3,2,3,2,2,3,3,1,1,4,3,4,4,3,3,4,4,2,4,4,3,1,4,1,3,1,4,2,3,1,1,4,2,4,2,1,4,1,1,3,3,4,1,4,4,3,2,4,2,2,1,2,3,2,1,3,1,2,1,1,4,1,1,3,3,4,1,4,4,1,4,2,2,4,1,2,1,4,2,2,2,1,3,3,4,4,4,2,1,2,1,2,1,3,4,3,1,2,4,3,1,4,1,3,2,1,3,4,1,1,1,1,4,4,4,1,2,2,1,3,3,2,1,2,4,1,1,4,3,2,1,2,4,4,3,1,2,4,2,4,4,2,1,2,1,1,2,1,1,1,2,1,4,4,3,1,2,4,1,1,1,4,2,3,3,1,1,2,4,1,4,2,2,3,1,1,4,2,4,4,1,2,1,4,3,3,3,4,3,1,1,1,1,1,2,1,1,1,1,4,3,2,4,2,2,4,3,4,4,4,2,3,1,1,4,2,3,3,2,2,1,1,4,1,1,3,1,4,3,1,3,2,4,2,1,1,1,3,4,3,3,4,3,1,4,2,4,4,3,1,1,2,3,1,4,4,2,2,3,3,1,1,3,4,2,1,2,2,2,1,1,2,3,1,2,3,4,4,2,4,1,4,3,1,2,1,3,3,4,1,1,1,2,4,4,2,2,1,2,2,2,1,4,2,4,4,4,1,3,4,2,1,1,1,1,1,4,4,3,4,1,3,4,1,3,2,2,2,2,4,3,2,1,4,1,2,2,2,2,2,1,4,4,2,1,2,2,1,2,2,1,1,1,3,2,4,4,2,3,1,2,1,2,4,1,4,2,4,1,1,4,3,4,4,1,4,1,1,3,3,1,4,2,4,2,3,1,2,4,3,2,2,1,4,4,2,4,3,4,4,3,2,4,3,1,1,3,1,1,4,4,4,2,3,1,1,3,1,3,1,3,4,1,4,1,1,4,3,1,4,3,1,1,1,2,2,1,2,2,2,3,1,3,3,4,3,3,2,1,2,3,1,4,1,3,2,3,1,2,2,2,1,2,1,2,1,3,3,2,2,3,4,4,3,2,2,3,1,1,3,1,1,1,3,2,1,2,2,1,2,3,4,4,1,4,4,2,1,1,1,4,2,1,4,2,3,3,4,1,4,4,4,3,4,1,3,3,3,3,2,1,4,2,4,1,4,2,1,1,2,1,2,4,3,3,2,4,4,2,4,1,3,4,1,2,1,2,4,2,3,1,2,3,4,1,1,1,1,3,2,2,4,3,1,1,2,3,3,4,1,1,4,4,2,3,2,1,4,4,2,4,1,3,2,2,3,3,2,3,3,4,1,1,2,4,3,2,3,3,3,1,2,3,4,1,3,1,3,2,1,2,1,4,3,1,4,3,1,3,3,4,4,2,4,3,2,1,2,4,3,2,2,3,2,3,4,4,1,2,3,2,2,4,4,2,3,1,4,4,2,1,4,3,4,1,1,4,2,3,2,1,4,1,4,1,3,2,3,4,1,2,4,4,2,3,4,2,2,4,4,4,1,3,3,4,3,1,4,4,3,2,4,4,4,4,1,2,4,1,2,2,2,1,1,3,3,4,2,2,3,2,2,4,1,4,1,4,4,1,4,2,1,4,4,1,4,1,2,1,2,1,3,4,2,3,4,3,1,3,2,2,2,1,2,2,1,1,2,3,4,1,1,2,4,4,4,2,3,1,2,3,2,4,1,3,2,2,2,1,3,3,2,1,1,1,1,2,3,2,1,2,1,3,1,1,1,3,4,2,2,1,4,1,2,1,2,4,3,4,1,1,3,1,2,3,3,2,2,4,2,1,1,1,3,3,3,4,4,1,1,3,2,4,2,1,3,4,4,3,3,1,1,3,4,2,1,1,4,3,3,1,2,1,1,1,4,3,2,2,4,3,1,4,4,1,3,2,1,4,2,1,2,3,4,2,4,4,1,2,3,4,2,3,3,2,1,2,3,2,4,2,2,2,4,2,4,4,1,3,4,2,4,3,3,1,3,4,4,2,1,1,2,3,2,2,2,1,3,1,2,1,2,4,2,4,3,1,2,4,2,1,2,2,4,3,2,2,1,4,2,2,2,2,1,1,3,4,2,4,4,3,1,1,1,4,1,4,1,3,1,1,4,3,1,3,4,4,2,4,3,2,2,2,4,4,3,2,4,1,2,3,2,1,4,1,1,1,3,4,3,3,3,3,4,4,2,4,1,4,1,4,1,4,2,2,3,2,4,4,1,4,1,1,4,2,4,2,1,1,1,3,3,2,2,4,1,2,4,1,2,3,2,2,2,4,1,3,4,4,1,3,1,4,4,2,2,4,4,2,1,4,4,2,3,3,1,1,2,1,1,1,2,1,2,2,4,4,4,2,3,3,1,3,1,4,1,2,4,4,4,3,3,4,3,2,1,3,1,2,2,2,1,3,1,2,3,1,4,3,4,2,3,1,3,2,1,1,2,1,2,1,1,2,1,2,2,3,2,3,1,4,1,3,3,1,1,1,2,4,4,3,1,3,1,2,2,3,3,2,1,4,4,2,2,4,1,2,2,4,3,1,4,2,4,4,1,1,3,2,2,4,3,3,2,2,1,4,1,1,2,1,2,2,1,2,4,4,1,1,2,4,3,2,3,2,3,4,3,1,4,3,1,1,4,4,2,4,1,4,1,1,2,4,4,4,1,4,2,4,1,2,2,1,1,2,4,2,2,1,2,2,3,1,1,2,3,3,2,3,4,1,4,2,3,1,1,4,4,2,2,2,3,4,4,2,4,2,2,4,1,4,1,1,3,3,3,1,4,4,1,4,3,3,3,1,1,4,1,1,4,2,4,4,2,4,4,1,1,1,1,3,1,1,1,2,4,2,2,4,2,2,4,4,3,2,1,2,4,1,1,4,2,3,1,2,1,1,4,2,2,2,2,1,4,2,2,2,2,2,3,2,4,1,2,2,3,3,3,1,2,2,4,3,3,1,2,3,2,1,1,4,4,3,3,4,3,1,3,3,2,3,1,2,3,3,2,1,1,2,2,4,2,1,2,4,3,1,1,3,2,4,1,1,1,4,4,2,1,1,4,3,2,3,4,4,1,2,1,3,3,1,4,1,4,4,4,3,4,4,4,3,1,4,3,1,2,1,1,4,2,1,2,2,4,3,3,1,3,4,4,1,1,3,3,2,3,2,4,2,1,4,1,2,3,3,2,2,2,3,2,3,2,2,2,2,1,3,4,3,2,1,4,2,2,2,2,2,4,1,2,1,3,1,3,1,3,4,4,1,3,3,3,1,3,4,3,3,4,2,1,1,2,1,2,1,2,1,1,4,1,2,2,4,4,4,1,3,3,1,1,3,3,2,1,4,4,1,3,2,1,4,4,1,1,2,4,4,2,3,1,4,4,4,1,1,2,4,3,1,1,2,3,2,3,2,3,2,4,1,1,1,2,1,4,2,3,1,3,4,1,1,1,2,1,1,3,2,1,2,2,4,3,2,1,4,2,4,4,4,1,3,3,1,2,1,4,3,3,2,1,1,2,3,3,3,2,2,3,2,3,3,2,2,3,1,2,1,3,4,1,2,1,1,4,2,1,4,3,4,1,2,3,1,1,4,3,3,3,1,1,4,4,1,3,1,1,4,1,3,1,1,1,1,2,2,4,1,1,4,2,2,4,1,4,1,3,3,1,4,4,3,3,2,1,1,2,4,4,1,3,3,2,4,2,2,4,3,1,4,3,4,4,2,4,2,4,4,3,4,1,2,3,2,1,1,3,1,3,3,2,2,4,3,3,2,3,2,2,3,1,1,1,2,3,2,2,1,1,1,4,1,4,1,1,2,3,2,4,4,4,4,3,4,2,1,3,2,2,2,1,2,4,1,1,2,4,1,2,1,3,1,1,1,2,4,2,4,2,3,2,4,3,2,3,3,1,4,1,4,3,2,3,3,3,2,4,3,1,3,3,4,1,3,1,2,2,1,1,4,4,3,2,2,1,3,1,3,2,2,1,2,1,1,1,1,2,4,2,3,4,3,1,3,2,4,2,1,1,2,3,1,1,1,2,2,3,1,4,3,3,2,1,1,3,2,1,2,1,4,1,1,4,3,3,2,4,2,1,1,1,3,3,2,1,1,1,1,4,2,2,1,3,4,4,4,1,3,4,1,3,2,4,1,4,3,1,3,1,1,4,1,4,2,3,1,4,2,3,3,1,1,1,2,2,2,2,3,2,3,3,2,3,2,1,3,3,1,3,2,2,3,1,1,1,1,3,1,3,2,3,2,3,3,3,1,1,3,3,3,3,1,2,1,1,1,2,3,3,1,2,1,1,3,3,1,2,3,1,1,2,1,1,2,1,1,1,3,1,1,1,2,1,3,1,2,3,2,3,1,1,2,3,2,3,3,2,1,1,3,2,3,2,3,1,3,1,1,1,3,1,2,3,2,1,2,1,1,3,1,2,2,1,1,3,2,2,3,1,1,3,3,2,2,1,1,1,2,1,3,1,2,3,2,2,1,2,1,1,3,3,1,3,1,2,1,2,3,1,2,1,3,1,1,3,1,1,1,3,2,3,1,1,1,1,1,2,3,1,3,1,1,1,1,2,2,2,2,1,1,3,2,1,1,2,3,1,3,1,1,1,3,2,1,1,1,1,1,1,3,3,3,2,1,2,2,1,2,1,1,1,1,1,1,2,1,1,1,3,3,1,2,1,3,3,3,2,1,3,1,2,3,2,1,2,3,3,3,1,2,1,3,3,3,1,2,3,2,1,3,2,2,1,1,3,3,3,1,1,1,1,3,3,2,1,3,1,1,1,3,1,2,1,3,1,3,3,3,1,1,3,3,3,2,3,2,2,2,3,2,1,2,1,3,1,3,1,3,1,3,1,1,1,1,1,1,1,3,1,1,1,1,3,1,3,3,2,2,3,2,2,2,1,1,1,2,3,1,2,1,2,1,2,1,2,1,2,3,1,2,1,2,1,3,3,2,2,3,1,1,3,2,2,2,1,3,3,2,1,3,1,1,1,2,3,2,3,2,3,1,1,1,2,1,1,1,1,2,3,2,3,3,1,1,3,1,2,2,3,1,1,3,3,3,1,1,1,2,1,1,1,2,1,2,2,1,1,3,1,2,1,3,3,1,2,2,1,2,3,1,3,2,1,1,1,1,1,1,1,2,3,1,1,1,2,1,3,2,2,1,1,2,3,1,3,3,1,3,2,3,1,3,2,2,3,2,2,3,1,1,1,1,3,2,3,2,3,1,3,3,2,1,1,2,1,2,2,2,1,1,3,1,2,2,1,2,1,2,1,3,1,1,1,1,1,2,3,3,3,2,1,2,2,3,3,2,3,3,3,1,2,2,1,3,3,3,2,2,2,3,2,2,1,1,1,1,1,1,1,2,1,3,1,1,3,1,2,3,2,3,3,1,1,1,1,1,3,3,3,2,2,1,1,1,1,1,3,1,2,1,3,3,2,1,3,1,3,1,1,1,2,2,1,2,2,2,3,1,2,2,3,1,3,2,1,3,1,1,1,1,3,1,3,2,1,1,3,1,3,1,1,1,1,2,1,1,2,3,2,3,2,1,1,1,1,1,3,3,3,3,2,2,1,3,2,3,3,1,2,1,1,2,3,3,1,1,1,3,1,3,3,1,1,1,2,2,1,1,2,3,3,1,1,2,3,3,3,3,3,2,1,1,2,2,3,3,1,3,1,2,1,2,1,3,1,2,3,1,3,2,1,3,2,2,1,2,2,2,1,1,3,2,3,3,1,1,3,3,3,3,1,3,1,1,1,2,1,2,1,1,3,2,2,2,2,1,2,1,3,1,3,3,2,1,1,1,1,2,2,3,1,1,1,1,1,2,3,2,2,2,1,3,1,1,1,2,2,2,1,2,1,3,2,2,1,2,3,2,2,2,2,1,2,1,2,1,1,3,1,2,2,1,3,1,1,2,3,1,2,3,1,3,3,1,1,1,1,3,1,2,1,2,3,1,3,3,1,3,2,3,1,1,1,3,2,2,3,1,1,3,3,3,2,2,1,2,2,1,2,3,1,3,3,1,1,2,2,3,1,2,2,3,2,3,1,1,1,2,3,1,1,1,1,1,3,2,1,3,2,1,1,2,3,1,2,1,3,1,2,2,2,3,2,3,2,3,1,1,1,1,1,2,1,1,3,2,1,2,3,2,1,1,1,1,2,2,2,1,3,3,3,1,2,1,1,3,1,2,1,2,1,1,1,1,2,1,1,3,3,1,3,3,2,2,1,1,1,1,3,2,2,1,1,2,3,3,2,2,1,2,2,3,2,2,3,1,3,2,2,1,3,1,2,2,3,2,3,3,3,2,2,3,1,2,3,1,1,1,3,2,1,1,1,2,3,2,3,2,2,3,1,3,1,3,1,1,1,2,3,3,3,2,2,1,3,1,1,1,2,3,1,3,1,2,2,3,2,1,2,2,1,1,3,1,3,1,1,3,1,3,1,2,2,1,3,2,2,3,3,2,2,2,2,1,2,1,2,1,3,3,3,2,2,1,1,3,1,2,2,3,3,1,3,1,2,3,2,2,1,3,3,1,2,2,2,1,3,3,3,2,1,3,1,1,2,1,3,1,3,1,2,3,3,1,1,3,2,3,1,1,2,2,2,3,3,1,3,2,1,2,2,3,1,2,1,3,3,1,1,2,1,3,3,1,1,2,1,1,2,1,1,1,3,3,1,2,1,1,1,3,3,2,1,3,1,1,1,1,2,1,1,3,1,2,3,2,2,3,1,2,1,3,1,3,2,3,1,2,1,1,1,3,1,1,3,1,1,1,3,1,2,1,1,1,2,1,2,1,3,3,2,2,2,2,2,2,1,2,2,3,2,1,1,2,1,1,2,2,3,1,1,1,1,2,3,2,1,3,1,1,1,2,2,1,2,1,3,2,2,2,1,1,1,1,2,3,3,1,1,2,1,1,2,1,3,2,1,2,1,1,2,1,3,1,1,3,2,3,3,3,1,3,2,1,2,3,3,1,2,2,2,3,3,1,3,3,3,3,1,2,3,3,3,3,1,3,3,2,1,3,3,1,2,3,1,1,3,1,1,3,2,1,2,1,1,2,2,2,1,1,2,1,2,1,1,3,1,2,2,3,2,1,3,2,1,2,3,1,3,3,3,3,1,3,2,3,3,3,1,1,2,1,2,2,1,1,2,3,1,2,1,1,2,2,1,1,1,2,3,2,1,1,1,3,1,3,1,1,2,2,2,3,2,3,1,1,1,1,2,1,1,2,1,1,1,3,1,3,3,3,3,1,1,3,3,3,2,1,3,1,1,2,1,2,3,1,1,1,1,3,2,1,1,3,2,3,3,3,1,1,2,3,2,1,3,2,3,3,3,1,1,3,1,1,3,1,1,1,2,1,1,3,2,3,1,3,3,1,1,3,1,1,1,2,1,1,1,1,2,3,2,3,1,1,2,1,3,1,3,1,1,2,1,1,1,1,1,3,3,2,3,1,3,1,1,3,1,3,2,2,3,1,2,3,3,3,2,3,1,3,3,1,1,1,3,2,2,1,1,2,2,1,1,1,2,3,3,3,1,2,1,2,1,2,2,3,1,3,3,3,2,3,1,1,1,3,2,1,3,3,1,1,3,3,3,1,2,3,1,3,1,3,1,1,1,1,3,1,3,3,1,1,3,2,1,1,1,2,2,1,1,1,1,1,2,2,3,1,1,2,1,2,1,3,2,3,2,3,2,2,1,1,2,3,1,1,1,1,2,1,3,1,1,2,3,1,3,1,2,2,1,1,3,2,1,3,3,3,1,1,1,1,2,3,3,1,1,3,1,3,2,3,1,2,2,3,2,1,1,3,2,1,2,1,2,2,1,2,1,3,2,2,2,2,2,1,3,3,1,3,1,3,1,3,1,2,1,3,2,2,2,3,3,3,1,1,1,1,1,3,1,3,2,2,1,3,2,1,2,2,2,3,1,1,2,1,3,2,1,1,3,1,1,3,1,1,1,3,1,2,1,3,1,1,2,1,1,2,1,3,2,2,1,3,1,2,3,2,3,1,2,1,3,1,1,2,1,2,2,1,1,3,1,3,2,1,3,2,3,3,3,1,2,1,3,2,1,2,3,1,1,1,3,1,1,2,3,2,2,3,2,1,3,1,1,2,3,1,1,1,2,2,3,1,2,1,3,2,1,1,1,3,2,1,3,3,2,2,2,1,1,2,1,1,1,1,2,1,3,1,3,1,1,3,3,1,3,1,1,2,1,2,1,1,1,1,1,3,1,3,1,3,1,3,2,1,2,3,3,1,1,1,2,1,2,3,1,1,2,3,1,2,2,3,2,1,3,1,1,3,3,2,2,1,3,3,2,3,1,2,1,2,3,1,2,2,2,2,1,1,1,1,3,2,1,3,2,1,1,2,3,2,1,2,2,1,1,2,1,2,3,2,1,1,1,3,2,2,3,1,1,2,1,3,1,1,2,3,1,1,2,1,3,3,1,1,1,1,2,2,3,2,2,1,2,1,3,3,3,2,3,3,3,1,3,1,2,1,3,3,2,1,1,2,3,3,2,3,3,1,3,1,2,1,2,1,3,3,1,3,2,3,2,1,3,2,1,1,1,1,2,1,1,1,2,3,2,3,3,1,1,2,3,1,3,1,1,3,3,1,1,2,2,1,2,2,1,2,1,3,3,1,1,1,1,2,3,3,3,2,2,2,1,1,3,2,2,2,3,1,1,2,1,3,3,1,1,2,1,1,1,3,3,1,1,2,2,1,3,2,1,1,1,1,2,1,2,2,1,3,3,2,1,3,1,2,2,2,2,2,3,1,1,1,1,3,1,1,3,1,3,2,1,3,3,1,2,1,3,1,1,1,1,3,3,3,1,3,2,3,2,2,2,3,2,2,2,2,3,2,1,2,3,3,1,3,3,2,1,1,1,3,3,1,3,3,1,3,1,3,1,1,1,1,3,1,2,3,1,3,3,2,1,2,1,1,1,1,3,3,1,1,1,2,2,3,3,2,2,3,3,1,1,1,2,1,3,1,3,1,1,2,3,1,2,1,1,1,1,3,2,2,2,3,1,3,3,1,3,2,3,3,2,1,1,1,2,1,2,2,3,3,3,3,3,3,1,3,3,1,1,1,1,3,2,1,2,3,2,2,1,1,3,1,1,1,2,3,3,1,2,2,1,1,3,2,1,2,2,3,3,3,1,2,3,3,1,2,3,2,3,1,1,3,2,1,2,1,3,1,3,1,1,2,2,2,2,3,2,2,1,2,1,1,1,1,2,1,2,1,2,2,2,1,2,2,1,3,2,3,1,1,1,3,3,1,1,2,1,1,3,2,1,3,1,3,3,3,2,3,3,1,2,2,1,3,1,2,2,3,1,3,1,3,1,1,1,1,2,1,1,2,2,2,3,2,1,3,2,1,3,3,1,2,3,3,1,3,1,1,2,3,3,1,1,2,3,2,1,2,1,3,1,1,1,1,2,3,1,1,1,3,2,2,2,3,2,2,3,2,3,3,3,2,1,2,2,1,1,1,3,3,1,2,2,3,1,1,2,1,3,2,2,1,1,2,1,3,1,3,1,2,1,1,1,2,1,2,1,3,3,3,2,1,3,2,3,2,1,1,1,2,1,3,1,1,3,3,1,3,1,2,1,3,2,2,3,2,2,2,1,1,2,2,1,2,1,2,3,2,3,1,1,3,2,3,1,1,3,1,3,1,3,3,2,3,1,3,3,3,2,2,1,1,2,1,3,2,1,2,3,2,1,1,3,1,2,3,1,2,2,2,1,1,3,1,1,3,1,1,3,1,1,3,1,1,2,3,1,2,1,3,2,2,1,1,2,1,1,3,2,1,1,3,2,3,3,1,2,1,2,1,1,1,1,1,2,2,1,2,3,1,1,2,2,3,2,2,1,1,1,2,2,2,2,2,1,2,1,3,3,3,3,2,1,2,1,3,1,1,3,3,3,3,2,3,1,1,1,1,1,3,3,1,2,3,1,3,1,3,1,3,3,1,2,2,1,2,1,1,3,2,1,3,2,1,1,1,1,3,3,3,1,1,2,2,1,1,1,1,1,2,2,3,3,1,3,1,2,1,1,2,2,1,1,2,3,1,2,3,3,2,1,1,1,3,1,2,3,2,1,1,3,2,3,1,3,1,1,1,1,3,1,1,1,3,1,2,1,3,2,1,1,1,2,3,2,2,1,2,1,1,3,2,1,3,1,1,3,3,3,3,3,3,3,3,3,1,2,1,2,2,2,1,1,2,3,2,1,2,2,2,1,2,1,2,1,3,2,2,1,1,1,1,1,1,2,2,3,2,1,3,1,1,2,1,2,1,1,1,1,3,3,2,1,2,1,1,3,3,3,1,1,2,1,2,3,3,2,3,2,1,3,2,1,2,3,1,3,2,1,2,1,1,3,2,1,1,1,3,3,1,1,2,1,1,1,2,3,2,2,1,1,2,2,3,3,1,1,3,3,1,1,1,1,3,2,3,3,2,2,1,3,3,1,3,3,1,3,2,2,2,3,3,3,3,1,1,1,1,1,2,1,3,1,3,2,2,2,1,1,3,3,1,1,1,1,2,2,3,1,2,1,1,3,3,2,1,2,1,1,2,1,2,3,2,2,1,1,3,3,3,3,1,3,3,1,1,1,1,1,3,1,1,3,2,3,3,2,2,1,3,2,2,3,3,2,1,2,2,1,1,2,3,3,2,1,3,2,1,3,2,2,1,1,3,2,1,2,2,3,3,2,2,1,1,2,1,1,1,3,1,3,1,3,1,3,1,1,1,1,2,1,2,1,3,2,1,2,3,2,1,1,2,3,2,1,1,3,2,3,3,1,1,2,3,3,2,1,1,3,3,2,3,1,3,2,1,1,3,3,2,2,1,1,1,2,1,2,1,2,1,1,1,1,3,2,3,1,1,1,1,1,3,1,3,1,2,1,1,2,1,2,2,1,2,2,2,3,2,1,1,1,1,3,1,3,1,1,3,1,1,2,3,1,1,1,3,3,1,2,3,3,3,1,3,2,2,1,3,1,1,2,2,2,2,1,1,1,1,2,1,3,1,2,3,2,1,1,1,3,3,2,3,1,1,3,3,3,2,2,2,3,1,3,3,3,1,3,1,2,3,3,1,2,2,3,1,1,1,3,3,1,2,2,2,1,3,2,1,2,1,2,3,3,2,2,1,2,3,1,2,3,3,2,1,2,3,3,3,2,1,3,2,1,1,1,3,1,2,2,2,1,3,1,2,3,1,1,1,1,1,3,1,1,3,2,3,3,1,3,1,3,2,3,2,2,1,3,1,2,1,1,3,1,1,2,2,1,2,2,2,1,1,2,1,1,3,3,1,1,1,2,2,1,1,1,3,3,1,3,3,3,1,3,1,3,2,2,2,2,1,1,1,2,1,3,1,2,2,1,3,2,3,3,3,1,1,1,2,3,1,3,3,3,2,1,3,2,1,1,3,3,1,2,3,3,2,2,1,1,1,2,3,1,3,2,1,1,1,3,2,3,2,3,2,3,3,2,1,3,1,3,1,3,3,3,1,3,1,2,2,1,1,3,1,3,1,1,2,1,2,1,2,3,2,2,3,1,1,3,2,1,1,3,2,2,3,1,2,2,1,3,2,1,3,1,1,3,2,2,1,1,3,1,1,3,3,1,3,1,1,3,1,1,1,2,3,1,1,3,3,2,2,3,1,3,3,2,3,3,1,3,3,1,1,1,3,3,3,2,1,3,3,3,1,3,2,1,2,1,1,2,2,2,3,1,2,2,2,3,2,1,2,3,2,1,3,2,3,2,1,1,1,1,1,3,3,1,1,3,3,1,2,3,2,1,1,2,3,2,3,3,2,3,3,3,3,1,2,3,1,1,3,1,1,2,2,3,1,3,1,3,3,2,1,3,1,3,2,1,3,1,1,3,1,2,2,1,1,2,3,3,1,1,2,1,3,3,2,3,3,2,2,1,1,3,1,1,2,3,3,2,1,1,2,2,3,2,1,1,3,2,1,2,1,1,1,3,1,1,3,1,1,2,3,1,3,2,1,1,1,3,3,2,2,3,1,3,1,2,3,1,3,3,3,2,2,3,3,2,3,1,2,2,1,2,3,1,2,2,3,3,1,2,2,3,3,2,1,2,3,1,1,1,1,1,3,2,3,1,1,3,1,1,1,3,2,2,1,3,2,2,1,3,2,1,1,1,2,2,2,3,3,3,2,1,2,1,3,3,1,3,1,2,2,2,3,2,3,3,2,1,2,2,2,1,2,2,2,1,3,3,1,1,2,1,3,2,3,3,2,1,1,2,3,3,3,1,1,1,1,1,3,2,1,3,1,2,3,1,2,2,2,3,1,3,1,1,1,3,3,1,1,3,2,2,1,3,1,3,3,3,3,1,3,2,2,3,2,3,3,2,2,1,1,2,2,1,2,2,2,1,1,2,1,1,1,3,2,2,3,3,1,1,1,2,1,1,1,1,1,1,2,2,3,1,1,1,1,1,3,1,2,3,2,1,3,3,2,1,1,3,3,2,1,3,3,3,2,2,1,1,2,3,1,1,3,3,1,3,1,1,2,3,2,3,1,2,1,1,2,3,2,3,1,2,3,1,1,1,3,1,2,3,1,2,2,2,1,2,1,3,2,3,3,1,2,1,2,3,3,1,1,1,3,2,1,3,1,1,2,1,3,3,3,3,1,2,1,1,1,2,3,2,1,1,1,1,1,1,1,3,1,2,2,2,1,3,1,3,1,2,3,1,1,1,1,1,1,1,1,1,2,1,1,2,1,3,3,1,2,3,2,1,1,1,1,1,3,3,1,1,1,3,3,2,2,1,3,2,1,1,2,1,2,1,2,3,2,1,3,2,3,3,1,3,2,2,3,1,2,1,2,3,1,1,2,3,2,3,1,3,3,1,1,2,1,1,1,2,2,3,2,3,3,2,1,2,2,3,2,1,2,1,1,1,3,1,1,1,2,3,1,3,3,3,1,2,1,1,3,3,3,2,1,1,1,2,3,3,1,2,3,1,3,1,1,1,2,1,1,2,3,1,1,3,1,1,1,1,2,2,2,3,3,2,3,2,2,3,1,1,3,1,2,2,2,2,1,1,1,3,3,3,1,3,3,1,2,1,3,1,2,3,2,2,2,2,1,2,1,1,3,1,3,3,3,3,2,1,1,2,3,1,3,3,2,1,3,3,2,1,2,2,1,3,2,3,1,3,2,1,3,1,1,1,3,1,2,3,1,3,1,1,3,1,3,1,1,2,3,1,3,2,3,3,1,1,1,2,1,1,1,2,2,3,1,3,3,2,1,1,1,1,3,1,3,3,3,1,3,1,3,2,3,3,2,2,1,1,2,3,1,3,1,3,2,2,1,2,3,2,1,2,2,1,3,3,3,3,1,1,2,3,1,1,1,1,2,2,2,1,3,1,1,2,1,1,3,1,1,2,2,3,2,1,3,1,3,3,2,1,1,3,2,2,2,3,1,2,2,1,1,2,1,3,2,1,3,2,3,3,1,1,2,3,2,3,3,2,1,3,1,3,3,1,3,1,3,2,2,1,2,3,3,3,2,1,3,1,2,2,1,3,1,3,2,1,3,3,3,3,1,2,1,1,2,3,2,2,2,1,1,3,2,3,3,1,3,1,1,2,2,3,1,2,1,3,3,2,2,2,2,1,3,3,3,2,2,3,3,2,2,3,1,1,1,2,3,1,3,1,1,3,1,1,1,1,3,1,2,2,1,2,1,2,2,1,3,3,3,3,2,3,1,1,2,2,3,1,1,1,3,1,2,1,2,2,2,1,1,3,3,2,2,1,1,1,2,1,2,3,3,1,1,1,1,3,2,3,1,3,1,1,3,2,1,1,2,3,1,3,1,2,3,1,3,1,3,2,3,3,2,2,3,1,3,2,2,1,2,1,2,1,1,3,3,3,3,3,1,2,1,2,1,2,2,1,1,1,3,1,2,1,3,1,1,1,2,2,2,1,2,1,3,1,1,3,1,1,2,2,2,3,1,2,1,2,2,3,2,2,1,1,1,1,3,2,3,3,1,3,1,2,3,1,3,1,1,2,1,2,1,2,2,3,1,3,1,1,1,3,1,3,3,2,2,2,3,1,2,2,2,1,1,1,1,1,3,1,2,3,2,3,3,1,3,1,1,3,1,3,2,2,1,2,1,3,3,3,3,3,1,1,1,3,2,3,3,1,1,3,2,2,2,1,2,3,1,2,3,1,1,1,2,1,3,1,3,2,1,2,1,1,3,1,2,3,1,3,3,3,1,1,2,3,2,1,2,1,1,3,2,3,1,1,1,1,2,2,3,3,1,3,2,3,3,1,1,3,1,1,1,3,3,3,2,2,1,2,2,1,2,1,1,2,3,1,3,2,1,2,2,1,3,2,3,2,1,1,1,1,3,1,3,1,2,1,1,2,2,2,2,1,3,1,2,1,2,2,3,1,1,1,3,1,2,3,2,3,3,3,3,1,3,3,3,2,1,1,3,3,1,2,3,3,3,3,3,2,2,3,2,3,2,1,2,1,1,3,3,1,1,2,2,1,2,3,3,2,2,2,1,3,3,3,3,1,1,1,1,2,3,1,2,3,2,2,1,2,2,1,1,3,1,1,3,2,3,3,1,2,1,2,1,1,1,2,1,3,1,1,1,2,2,2,2,3,1,1,1,2,1,2,1,1,3,3,2,1,3,1,2,3,1,1,3,2,1,1,1,3,1,1,2,1,2,2,2,3,3,2,1,3,1,3,2,1,1,1,1,3,3,2,1,1,1,1,2,3,1,3,1,1,3,1,1,1,3,3,1,3,1,1,1,1,1,2,3,2,3,1,1,3,2,2,2,1,1,1,1,2,1,3,1,3,2,3,1,3,1,1,2,1,1,2,1,1,2,1,2,3,2,2,1,1,1,2,1,3,3,1,2,3,3,3,2,1,3,2,2,1,1,1,3,1,2,1,2,3,1,1,1,3,1,3,1,3,1,1,1,1,3,2,1,1,1,2,2,1,3,3,2,3,1,2,1,2,1,3,1,1,3,1,1,3,1,3,3,3,3,2,1,1,1,3,1,1,1,2,1,2,2,1,1,2,2,1,3,1,1,3,1,1,1,3,2,3,3,2,3,1,2,3,2,1,3,3,3,2,1,2,1,3,1,3,3,1,1,1,3,1,3,3,1,3,3,1,3,3,1,3,3,3,1,1,1,3,3,1,2,1,2,1,1,3,2,1,2,1,1,2,3,1,2,1,2,2,1,3,3,1,1,1,2,2,2,3,1,2,3,1,1,3,1,3,1,3,2,1,2,1,2,2,1,2,1,1,1,1,3,3,1,1,3,3,3,3,2,3,3,3,3,1,2,1,2,1,3,2,3,3,3,2,2,2,3,3,2,1,1,1,1,3,2,3,2,1,3,3,2,3,1,3,3,1,3,2,1],"y0":" ","yaxis":"y","type":"box"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"variable"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"value"}},"legend":{"tracegroupgap":0},"margin":{"t":60},"boxmode":"group"},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('b4744e25-35e2-480f-a3d4-972b0ef64c6c');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>


#### Five point summary suggest that Experience has negative value(This should be fixed).

we can see the Min, Max, mean and std deviation for all key attributes of the dataset

Income has too much noise and slightly skewed right, Age and exp are equally distributed.

#### check if there is skewness in data or not!!


```python
df.skew()
```




    Age                  -0.029341
    Experience           -0.026325
    Income                0.841339
    Family                0.155221
    CCAvg                 1.598457
    Education             0.227093
    Mortgage              2.104002
    Personal Loan         2.743607
    Securities Account    2.588268
    CD Account            3.691714
    Online               -0.394785
    CreditCard            0.904589
    dtype: float64




```python
df.dtypes
```




    Age                     int64
    Experience              int64
    Income                  int64
    Family                  int64
    CCAvg                 float64
    Education               int64
    Mortgage                int64
    Personal Loan           int64
    Securities Account      int64
    CD Account              int64
    Online                  int64
    CreditCard              int64
    dtype: object



#### now visualise Skewness by distribution


```python
df.hist(figsize=(20, 20))
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7f3932109d10>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f39320e3450>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f393209ca50>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f3932047b90>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f3932017690>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f3931fccc90>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f3931f8d350>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f3931fc4890>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f3931fc48d0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f3931f7cfd0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f3931ef5b10>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f3931eb7150>]],
          dtype=object)




    
![png](Bank_Loan_files/Bank_Loan_16_1.png)
    


#### INFERENCE from Histogram
    1.Age & Experience are to an extent equally distributed
    2.Income & Credit card spending are skewed to the left
    3.We have more Undergraduates than Graduate and Advanced & Professional
    4.60% of customers have enabled online banking and went digital


```python
import seaborn as sns
sns.distplot(df['Experience'])
```

    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2619: FutureWarning:
    
    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
    





    <matplotlib.axes._subplots.AxesSubplot at 0x7f39315e2fd0>




    
![png](Bank_Loan_files/Bank_Loan_18_2.png)
    



```python
df['Experience'].mean()
```




    20.1046




```python
Negative_exp = df[df['Experience'] < 0]
Negative_exp.head()
```





  <div id="df-5a7f2bda-8b03-4587-9c3f-53ad5d34bac2">
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
      <th>Experience</th>
      <th>Income</th>
      <th>Family</th>
      <th>CCAvg</th>
      <th>Education</th>
      <th>Mortgage</th>
      <th>Personal Loan</th>
      <th>Securities Account</th>
      <th>CD Account</th>
      <th>Online</th>
      <th>CreditCard</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>89</th>
      <td>25</td>
      <td>-1</td>
      <td>113</td>
      <td>4</td>
      <td>2.30</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>226</th>
      <td>24</td>
      <td>-1</td>
      <td>39</td>
      <td>2</td>
      <td>1.70</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>315</th>
      <td>24</td>
      <td>-2</td>
      <td>51</td>
      <td>3</td>
      <td>0.30</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>451</th>
      <td>28</td>
      <td>-2</td>
      <td>48</td>
      <td>2</td>
      <td>1.75</td>
      <td>3</td>
      <td>89</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>524</th>
      <td>24</td>
      <td>-1</td>
      <td>75</td>
      <td>4</td>
      <td>0.20</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-5a7f2bda-8b03-4587-9c3f-53ad5d34bac2')"
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
          document.querySelector('#df-5a7f2bda-8b03-4587-9c3f-53ad5d34bac2 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-5a7f2bda-8b03-4587-9c3f-53ad5d34bac2');
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





```python
sns.distplot(Negative_exp['Age'])
```

    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2619: FutureWarning:
    
    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
    





    <matplotlib.axes._subplots.AxesSubplot at 0x7f39315b52d0>




    
![png](Bank_Loan_files/Bank_Loan_21_2.png)
    



```python
Negative_exp['Experience'].mean()
```




    -1.4423076923076923




```python
Negative_exp.size
```




    624




```python
print('There are {} records which has negative values for experience, approx {} %'.format(Negative_exp.size , ((Negative_exp.size / df.size) * 100)))
```

    There are 624 records which has negative values for experience, approx 1.04 %



```python
data = df.copy()
data.head()
```





  <div id="df-ab815e39-93d7-4957-a9a2-eef24e67d90c">
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
      <th>Experience</th>
      <th>Income</th>
      <th>Family</th>
      <th>CCAvg</th>
      <th>Education</th>
      <th>Mortgage</th>
      <th>Personal Loan</th>
      <th>Securities Account</th>
      <th>CD Account</th>
      <th>Online</th>
      <th>CreditCard</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25</td>
      <td>1</td>
      <td>49</td>
      <td>4</td>
      <td>1.6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45</td>
      <td>19</td>
      <td>34</td>
      <td>3</td>
      <td>1.5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>39</td>
      <td>15</td>
      <td>11</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35</td>
      <td>9</td>
      <td>100</td>
      <td>1</td>
      <td>2.7</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35</td>
      <td>8</td>
      <td>45</td>
      <td>4</td>
      <td>1.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ab815e39-93d7-4957-a9a2-eef24e67d90c')"
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
          document.querySelector('#df-ab815e39-93d7-4957-a9a2-eef24e67d90c button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ab815e39-93d7-4957-a9a2-eef24e67d90c');
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




#### use numpy where function to change the negative values to mean value derived from data with the same age group


```python
data['Experience'] = np.where(data['Experience'] < 0, data['Experience'].mean(), data['Experience'])
data[data['Experience'] < 0]
```





  <div id="df-b1b450eb-4af5-4b52-997a-f880d200626c">
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
      <th>Experience</th>
      <th>Income</th>
      <th>Family</th>
      <th>CCAvg</th>
      <th>Education</th>
      <th>Mortgage</th>
      <th>Personal Loan</th>
      <th>Securities Account</th>
      <th>CD Account</th>
      <th>Online</th>
      <th>CreditCard</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-b1b450eb-4af5-4b52-997a-f880d200626c')"
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
          document.querySelector('#df-b1b450eb-4af5-4b52-997a-f880d200626c button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-b1b450eb-4af5-4b52-997a-f880d200626c');
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





```python
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f392f4a1190>




    
![png](Bank_Loan_files/Bank_Loan_28_1.png)
    


#### We could see that Age & Experience are very strongly correlated,
#### Hence it is fine for us to go with Age and drop Experience to avoid multi-colinearity issue.


```python
data = data.drop(['Experience'], axis=1)
data.head()
```





  <div id="df-7e84d62b-d202-44f8-97e5-7359ebef6d23">
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
      <th>Income</th>
      <th>Family</th>
      <th>CCAvg</th>
      <th>Education</th>
      <th>Mortgage</th>
      <th>Personal Loan</th>
      <th>Securities Account</th>
      <th>CD Account</th>
      <th>Online</th>
      <th>CreditCard</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25</td>
      <td>49</td>
      <td>4</td>
      <td>1.6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45</td>
      <td>34</td>
      <td>3</td>
      <td>1.5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>39</td>
      <td>11</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35</td>
      <td>100</td>
      <td>1</td>
      <td>2.7</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35</td>
      <td>45</td>
      <td>4</td>
      <td>1.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-7e84d62b-d202-44f8-97e5-7359ebef6d23')"
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
          document.querySelector('#df-7e84d62b-d202-44f8-97e5-7359ebef6d23 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-7e84d62b-d202-44f8-97e5-7359ebef6d23');
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





```python
data['Education'].unique()
```




    array([1, 2, 3])




```python
def mark(x):
    if x == 1:
        return 'Undergrad'
    elif x == 2:
        return 'Graduate'
    else:
        return 'Advanced/Professional'
```


```python
data['Edu_mark'] = data['Education'].apply(mark)
data.head()
```





  <div id="df-ab5867d3-9ec5-4722-9673-9c22ec92151c">
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
      <th>Income</th>
      <th>Family</th>
      <th>CCAvg</th>
      <th>Education</th>
      <th>Mortgage</th>
      <th>Personal Loan</th>
      <th>Securities Account</th>
      <th>CD Account</th>
      <th>Online</th>
      <th>CreditCard</th>
      <th>Edu_mark</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25</td>
      <td>49</td>
      <td>4</td>
      <td>1.6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Undergrad</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45</td>
      <td>34</td>
      <td>3</td>
      <td>1.5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Undergrad</td>
    </tr>
    <tr>
      <th>2</th>
      <td>39</td>
      <td>11</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Undergrad</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35</td>
      <td>100</td>
      <td>1</td>
      <td>2.7</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Graduate</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35</td>
      <td>45</td>
      <td>4</td>
      <td>1.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Graduate</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ab5867d3-9ec5-4722-9673-9c22ec92151c')"
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
          document.querySelector('#df-ab5867d3-9ec5-4722-9673-9c22ec92151c button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ab5867d3-9ec5-4722-9673-9c22ec92151c');
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





```python
EDU_dis = data.groupby('Edu_mark')['Age'].count()
EDU_dis
```




    Edu_mark
    Advanced/Professional    1501
    Graduate                 1403
    Undergrad                2096
    Name: Age, dtype: int64




```python
fig = px.pie(data, values=EDU_dis, names=EDU_dis.index, title='Pie CHart')
fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-2.8.3.min.js"></script>                <div id="394670e5-f491-4e1c-86ae-889975673792" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("394670e5-f491-4e1c-86ae-889975673792")) {                    Plotly.newPlot(                        "394670e5-f491-4e1c-86ae-889975673792",                        [{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"hovertemplate":"label=%{label}<br>value=%{value}<extra></extra>","labels":["Advanced/Professional","Graduate","Undergrad"],"legendgroup":"","name":"","showlegend":true,"values":[1501,1403,2096],"type":"pie"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"legend":{"tracegroupgap":0},"title":{"text":"Pie CHart"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('394670e5-f491-4e1c-86ae-889975673792');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>


#### Inference :We could see that We have more Undergraduates 41.92% than graduates(28.06%) & Advanced Professional(30.02%)


```python
data.columns
```




    Index(['Age', 'Income', 'Family', 'CCAvg', 'Education', 'Mortgage',
           'Personal Loan', 'Securities Account', 'CD Account', 'Online',
           'CreditCard', 'Edu_mark'],
          dtype='object')



#### Lets Explore the account holder's distribution


```python
def Security_CD(row):
    if (row['Securities Account'] == 1) & (row['CD Account'] == 1):
        return 'Holds Securites & Deposit'
    elif (row['Securities Account'] == 0) & (row['CD Account'] == 0):
        return 'Does not Holds Securites or Deposit'
    elif (row['Securities Account'] == 1) & (row['CD Account'] == 0):
        return ' Holds only Securites '
    elif (row['Securities Account'] == 0) & (row['CD Account'] == 1):
        return ' Holds only Deposit'
```


```python
data['Account_holder_category'] = data.apply(Security_CD, axis=1)
data.head()
```





  <div id="df-066f47a3-3a51-4289-920a-a8d79e6d4f54">
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
      <th>Income</th>
      <th>Family</th>
      <th>CCAvg</th>
      <th>Education</th>
      <th>Mortgage</th>
      <th>Personal Loan</th>
      <th>Securities Account</th>
      <th>CD Account</th>
      <th>Online</th>
      <th>CreditCard</th>
      <th>Edu_mark</th>
      <th>Account_holder_category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25</td>
      <td>49</td>
      <td>4</td>
      <td>1.6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Undergrad</td>
      <td>Holds only Securites</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45</td>
      <td>34</td>
      <td>3</td>
      <td>1.5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Undergrad</td>
      <td>Holds only Securites</td>
    </tr>
    <tr>
      <th>2</th>
      <td>39</td>
      <td>11</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Undergrad</td>
      <td>Does not Holds Securites or Deposit</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35</td>
      <td>100</td>
      <td>1</td>
      <td>2.7</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Graduate</td>
      <td>Does not Holds Securites or Deposit</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35</td>
      <td>45</td>
      <td>4</td>
      <td>1.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Graduate</td>
      <td>Does not Holds Securites or Deposit</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-066f47a3-3a51-4289-920a-a8d79e6d4f54')"
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
          document.querySelector('#df-066f47a3-3a51-4289-920a-a8d79e6d4f54 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-066f47a3-3a51-4289-920a-a8d79e6d4f54');
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





```python
values = data['Account_holder_category'].value_counts()
values.index
```




    Index(['Does not Holds Securites or Deposit', ' Holds only Securites ',
           ' Holds only Deposit', 'Holds Securites & Deposit'],
          dtype='object')




```python
fig = px.pie(data, values=values, names=values.index, title='Pie CHart')
fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-2.8.3.min.js"></script>                <div id="737da190-3a4d-4fec-a212-a9b419ff032e" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("737da190-3a4d-4fec-a212-a9b419ff032e")) {                    Plotly.newPlot(                        "737da190-3a4d-4fec-a212-a9b419ff032e",                        [{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"hovertemplate":"label=%{label}<br>value=%{value}<extra></extra>","labels":["Does not Holds Securites or Deposit"," Holds only Securites "," Holds only Deposit","Holds Securites & Deposit"],"legendgroup":"","name":"","showlegend":true,"values":[4323,375,155,147],"type":"pie"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"legend":{"tracegroupgap":0},"title":{"text":"Pie CHart"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('737da190-3a4d-4fec-a212-a9b419ff032e');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>


#### We could see that alomst 87% of customers do not hold any securities or deposit, and 3 % hold both securities as well as deposit. It will be good if we encourage those 87% to open any of these account as it will improve the assests of the bank


```python
data.columns
```




    Index(['Age', 'Income', 'Family', 'CCAvg', 'Education', 'Mortgage',
           'Personal Loan', 'Securities Account', 'CD Account', 'Online',
           'CreditCard', 'Edu_mark', 'Account_holder_category'],
          dtype='object')




```python
px.box(data, x='Education', y='Income', facet_col='Personal Loan')
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-2.8.3.min.js"></script>                <div id="caf4f5c2-4d25-4d22-8874-e17e21d359f5" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("caf4f5c2-4d25-4d22-8874-e17e21d359f5")) {                    Plotly.newPlot(                        "caf4f5c2-4d25-4d22-8874-e17e21d359f5",                        [{"alignmentgroup":"True","hovertemplate":"Personal Loan=0<br>Education=%{x}<br>Income=%{y}<extra></extra>","legendgroup":"","marker":{"color":"#636efa"},"name":"","notched":false,"offsetgroup":"","orientation":"v","showlegend":false,"x":[1,1,1,2,2,2,2,3,2,3,2,3,2,1,3,1,2,2,3,1,1,1,1,3,1,3,3,2,3,3,3,1,1,3,3,3,1,1,1,1,2,3,1,2,1,1,3,1,2,1,1,2,1,1,2,1,1,1,3,1,1,1,2,1,3,1,3,2,1,1,2,3,2,3,3,2,1,1,3,3,2,3,1,3,1,1,1,3,1,2,3,2,1,2,1,1,3,1,2,2,1,1,3,2,2,3,1,1,3,3,2,2,1,1,1,2,1,3,1,3,2,2,1,2,1,1,3,3,1,3,1,2,1,2,3,1,2,1,1,1,3,1,1,1,3,2,1,1,1,1,1,2,3,1,3,1,1,1,1,2,2,2,1,1,3,2,1,2,3,1,1,1,1,3,2,1,1,1,1,1,1,3,3,2,1,2,2,1,2,1,1,1,1,1,2,1,1,1,3,3,1,2,1,3,3,3,2,1,3,1,2,3,2,1,2,3,3,3,1,2,1,3,3,1,2,3,1,3,2,1,1,3,3,1,1,1,1,3,2,1,3,1,1,1,3,1,2,1,3,1,3,3,3,1,1,3,3,3,2,3,2,2,2,2,1,2,1,3,1,3,1,3,1,1,1,1,1,1,1,3,1,1,1,1,3,1,3,3,3,2,2,2,3,1,2,1,2,1,2,1,2,1,2,3,1,2,1,2,1,3,3,2,2,3,3,2,2,1,3,3,2,1,3,1,1,1,2,3,3,2,3,1,1,1,2,1,1,1,1,2,3,2,3,3,1,3,1,2,2,1,1,3,3,3,1,1,1,2,1,1,2,1,2,2,1,1,3,1,2,1,3,3,1,2,2,1,2,3,1,3,1,1,1,1,1,1,1,2,3,1,1,1,2,1,3,2,1,1,2,1,3,3,1,3,2,3,1,3,2,2,3,2,2,3,1,1,1,1,2,3,1,3,3,2,1,1,1,2,2,1,1,3,1,2,1,2,1,2,1,3,1,1,1,1,1,2,3,3,3,2,1,2,2,3,3,3,3,3,1,2,2,1,3,3,3,2,2,2,3,2,2,1,1,1,1,1,1,2,1,3,1,1,3,1,2,2,3,3,1,1,1,1,1,3,3,3,2,2,1,1,1,1,1,3,1,2,1,3,3,2,1,3,1,1,1,1,2,1,2,2,2,3,1,2,2,3,1,3,2,1,3,1,1,1,1,3,1,3,2,1,1,1,3,1,1,1,1,2,1,1,2,2,3,2,1,1,1,1,1,3,3,3,3,2,1,3,2,3,3,1,2,1,1,2,3,1,1,1,3,1,3,3,1,1,1,2,2,1,1,2,3,1,1,2,3,3,3,3,2,1,1,2,3,3,1,3,1,2,1,1,3,1,2,3,1,3,2,1,2,2,1,2,2,2,1,1,3,2,3,3,1,1,3,3,3,3,1,3,1,1,2,1,2,1,1,3,2,2,2,2,1,2,1,3,1,3,3,1,1,1,1,2,3,1,1,1,1,1,2,2,2,1,3,1,1,1,2,2,2,1,2,1,3,2,2,1,2,3,2,2,2,2,1,2,1,2,1,3,1,2,1,1,1,2,1,2,1,1,1,1,1,1,2,1,2,3,1,3,3,1,3,2,3,1,1,1,3,2,3,1,1,3,3,3,2,1,2,2,1,2,3,1,3,3,1,1,2,2,3,1,2,2,3,2,3,1,1,2,3,1,1,1,1,1,3,2,1,3,2,1,1,2,3,1,2,1,3,1,2,2,3,2,3,2,3,1,1,1,1,1,2,1,1,3,2,1,2,3,2,1,1,1,1,2,2,2,1,3,3,2,1,1,3,2,1,1,1,1,1,2,1,1,3,3,1,3,3,2,2,1,1,1,3,2,2,1,1,2,3,3,2,1,2,2,3,2,2,3,1,3,2,2,1,2,2,3,2,3,3,3,2,2,1,2,1,1,1,3,2,1,1,1,2,3,3,2,2,3,1,1,3,1,1,2,3,3,3,2,1,3,1,1,1,2,3,1,3,1,2,3,2,1,2,2,1,1,3,1,3,1,1,1,3,1,2,2,1,2,2,3,3,2,2,2,1,1,2,1,3,3,2,2,1,1,3,1,2,3,1,3,1,2,3,2,2,1,3,1,2,2,2,1,3,3,3,2,1,3,1,1,1,3,2,3,3,1,1,3,2,1,1,2,2,2,3,1,3,2,1,2,2,3,1,2,1,3,3,1,2,1,3,3,1,1,1,1,2,1,1,1,3,3,1,2,1,1,1,3,3,2,1,3,1,1,1,1,3,1,2,3,2,2,1,2,1,1,2,3,1,2,1,1,1,3,1,1,3,1,1,1,3,1,2,1,1,1,1,3,3,2,2,2,2,2,2,2,3,2,1,1,2,1,1,2,2,3,1,1,1,1,2,3,1,3,1,1,1,2,2,1,1,2,2,2,1,1,1,1,2,3,3,1,1,2,1,1,2,1,3,1,2,1,1,2,1,3,1,1,3,2,3,3,1,3,2,1,3,3,1,2,2,2,3,3,1,3,3,3,3,1,2,3,3,3,3,1,3,3,2,1,3,3,1,2,1,1,3,1,1,3,2,1,2,1,1,2,2,1,1,2,1,1,1,3,1,2,2,3,2,1,3,2,1,2,3,1,3,3,3,3,1,3,2,3,3,3,1,1,1,2,2,1,1,2,3,2,1,1,2,2,1,1,1,3,1,1,1,3,1,3,1,1,2,2,3,2,3,1,1,1,1,2,1,1,2,1,1,1,3,1,3,3,3,3,1,3,2,1,3,1,1,2,1,2,3,1,1,1,1,3,2,1,1,3,2,3,3,1,1,2,3,1,3,3,1,1,1,1,3,1,1,1,1,1,3,2,3,1,3,3,1,1,3,1,1,1,2,1,1,1,1,2,3,2,3,1,1,1,3,1,3,1,1,2,1,1,1,1,1,3,3,2,3,1,3,1,1,3,1,3,2,2,3,1,2,3,3,3,2,3,3,3,1,1,1,3,2,1,1,2,2,1,1,1,2,3,3,1,1,2,1,2,3,1,3,3,3,2,3,1,1,1,3,2,1,3,1,1,3,3,1,2,3,1,3,1,3,1,1,1,1,3,1,3,3,1,1,3,2,1,1,1,2,2,1,1,1,1,1,2,2,3,1,1,1,2,1,3,2,3,2,3,2,2,1,2,3,1,1,1,2,3,1,3,1,3,1,2,1,1,2,1,3,3,3,1,1,1,1,2,1,1,3,1,3,2,3,1,2,2,3,1,1,3,2,1,2,1,2,2,1,2,1,2,2,2,1,3,1,3,1,1,3,1,2,1,3,2,2,2,3,1,1,1,1,1,1,3,2,2,1,3,1,2,2,2,3,1,1,3,2,1,3,1,1,3,1,1,1,3,1,2,1,3,1,1,2,1,1,2,1,3,2,2,1,3,1,2,3,2,3,1,2,1,3,1,1,2,1,2,2,1,1,3,1,3,2,1,3,2,3,3,3,2,1,3,2,1,2,3,1,1,1,3,1,1,2,3,2,2,3,2,1,1,1,2,3,1,1,1,2,2,3,1,2,1,3,2,1,1,3,2,1,3,3,2,2,2,1,1,2,1,1,1,2,1,3,3,1,1,1,3,1,2,1,2,1,1,1,1,1,3,1,3,1,3,1,3,2,1,2,3,3,1,1,1,1,1,1,2,3,1,2,2,2,1,3,1,1,3,3,2,2,1,3,3,2,3,1,2,1,2,3,1,2,2,2,2,1,1,1,3,2,1,3,2,1,1,2,2,1,2,2,1,1,2,1,2,3,2,1,1,3,2,2,3,1,2,1,3,1,1,2,3,1,1,2,1,3,3,1,1,1,1,2,2,3,2,2,1,3,3,2,3,3,3,1,3,1,2,1,3,3,2,1,1,2,3,2,3,1,3,1,2,1,2,1,3,3,1,3,2,3,2,1,3,2,1,1,1,1,2,1,1,1,2,3,2,3,3,1,1,2,3,1,3,1,1,3,3,1,1,2,2,1,2,2,1,2,1,3,3,1,1,1,1,2,3,3,2,2,1,3,2,2,3,1,1,2,1,3,1,1,2,1,1,1,3,1,2,2,1,3,2,1,1,1,1,2,1,2,2,1,3,3,1,3,1,2,2,2,3,1,1,1,1,3,1,1,3,1,3,1,3,3,1,2,1,3,1,1,1,1,3,3,3,1,2,3,2,2,2,3,2,2,2,3,2,1,2,3,3,1,3,2,1,1,1,3,1,3,3,1,3,1,1,1,1,1,3,1,2,3,1,3,3,2,1,2,1,1,1,1,3,3,1,1,1,2,2,3,2,2,3,3,1,1,1,1,3,1,3,1,1,2,3,1,2,1,1,1,2,2,2,3,1,3,3,1,3,2,3,3,2,1,1,1,2,1,2,3,3,3,3,3,1,3,3,1,1,1,1,3,1,3,2,2,1,1,3,1,1,1,2,3,1,2,2,1,1,3,1,2,3,3,3,1,2,3,3,1,2,3,2,1,1,3,2,1,2,1,3,1,3,1,1,2,2,2,3,2,2,1,2,1,1,1,1,2,1,2,1,2,1,2,2,1,3,1,1,1,3,3,1,2,1,1,3,2,1,3,1,3,3,2,3,3,1,2,1,3,1,2,2,3,1,3,1,3,1,1,1,2,1,1,2,2,2,3,1,2,1,3,3,1,2,3,3,1,3,1,2,3,3,1,1,2,3,1,2,1,3,1,1,1,2,3,1,1,1,3,2,2,2,3,2,3,3,3,2,1,2,2,1,1,1,3,3,1,2,3,1,1,2,1,3,2,1,1,1,3,1,3,1,1,1,1,2,1,2,1,3,3,2,3,2,3,2,1,1,1,2,1,3,1,1,3,3,1,3,1,2,1,3,2,2,3,2,2,1,1,2,1,2,1,2,3,2,3,1,1,2,3,1,1,3,1,3,1,3,3,2,1,3,3,3,2,2,1,1,2,1,3,2,1,2,3,2,1,1,3,1,2,1,2,2,2,1,1,3,1,1,3,1,1,1,1,3,1,1,2,3,1,2,1,3,2,1,1,2,1,1,3,2,1,3,2,3,3,1,2,1,2,1,1,1,1,1,2,2,1,2,3,1,1,2,3,2,1,1,2,2,2,1,2,1,3,3,3,3,2,1,2,1,3,1,1,3,3,3,2,3,1,1,1,1,1,3,3,1,2,3,1,3,1,3,1,3,3,1,2,2,1,1,2,1,2,1,1,1,1,3,3,3,1,2,2,1,1,1,1,1,2,2,1,3,1,2,1,1,2,2,1,2,3,1,2,3,3,2,1,1,1,3,1,2,3,2,1,1,3,2,3,1,1,1,1,1,3,1,1,1,3,1,2,1,3,2,1,1,1,3,2,2,1,2,1,1,2,1,3,1,1,3,3,3,3,3,3,3,3,1,2,1,2,2,1,1,2,3,2,1,2,2,2,1,2,1,2,1,2,2,1,1,1,1,1,2,2,3,2,1,1,1,2,1,2,1,1,1,1,3,3,2,1,2,1,1,3,3,3,1,1,2,1,2,3,3,2,3,2,1,3,1,2,3,1,3,2,1,2,1,1,3,2,1,1,1,3,1,1,1,1,2,3,2,1,1,2,3,3,1,1,3,3,1,1,1,1,3,2,3,3,2,2,1,3,1,3,1,3,2,2,3,3,3,1,1,1,1,1,2,1,3,1,3,2,2,2,1,1,3,1,1,1,1,2,3,1,1,1,3,3,2,1,2,1,1,2,1,2,3,2,1,3,3,3,1,3,3,1,1,1,1,1,3,1,1,2,3,3,2,1,3,2,2,3,2,1,2,2,1,1,2,3,3,2,1,3,1,3,2,2,1,3,2,1,2,2,3,2,1,1,2,1,1,1,3,1,3,1,3,1,3,1,1,1,1,2,1,2,1,3,2,1,2,3,2,1,1,2,3,2,1,1,3,2,3,1,1,2,3,2,1,1,3,3,2,3,1,3,2,1,3,2,2,1,1,1,2,1,2,1,2,1,1,1,1,3,3,1,1,1,1,1,3,1,3,2,1,1,1,2,2,1,2,3,2,1,1,1,1,3,1,3,1,1,3,1,1,2,3,1,1,1,3,3,1,2,3,3,1,3,2,1,1,1,2,2,2,2,1,1,1,1,2,1,3,1,3,2,1,1,1,3,2,3,1,1,3,3,3,2,2,2,3,1,3,3,3,1,3,1,2,3,3,1,2,3,1,1,1,3,3,1,2,2,2,3,2,1,2,1,2,3,3,2,2,1,2,3,1,2,3,3,2,1,2,3,3,3,2,1,3,2,1,1,1,3,1,2,2,2,1,3,1,2,3,1,1,1,1,1,1,2,3,3,1,1,3,2,2,2,1,3,1,1,1,3,1,1,2,2,1,2,2,2,1,1,2,1,1,3,3,1,1,1,2,2,1,1,1,3,3,1,3,3,3,1,3,1,3,2,2,2,2,1,1,1,2,1,3,1,2,1,3,2,3,3,3,1,1,2,3,1,3,3,3,2,1,3,2,1,1,3,3,1,2,3,3,2,2,1,1,1,2,3,1,3,2,1,1,1,3,2,3,2,3,2,3,3,2,1,3,1,3,1,3,3,3,1,1,1,1,3,1,1,1,2,1,2,1,2,3,2,2,3,1,3,2,1,1,3,3,1,2,2,1,3,2,1,3,1,1,3,2,2,1,1,3,1,1,3,3,1,1,3,1,1,1,1,1,3,3,2,2,3,1,3,3,2,3,1,3,1,1,1,3,3,3,2,1,3,3,3,1,2,1,2,1,1,2,2,2,3,1,2,2,3,2,1,3,2,1,3,2,3,2,1,1,1,1,3,1,1,3,3,1,2,3,2,1,1,2,3,2,3,3,3,3,3,3,1,2,3,1,1,3,1,1,2,2,3,1,3,1,3,3,2,1,3,1,3,2,1,3,1,1,3,1,2,2,1,1,2,3,3,1,1,2,1,3,3,3,3,2,2,1,1,1,1,2,3,3,2,1,2,2,2,1,1,3,1,2,1,1,1,1,1,3,1,1,2,3,1,3,2,1,1,3,3,2,2,3,1,1,2,3,1,3,3,3,2,2,3,2,3,1,2,2,1,2,1,2,2,3,3,1,2,2,3,3,2,1,2,3,1,1,1,1,1,3,2,3,1,1,1,1,3,2,2,1,3,2,2,1,3,2,1,1,1,2,2,2,3,2,1,2,1,3,3,3,2,2,2,3,3,3,2,1,2,2,2,1,2,2,2,1,3,3,1,1,2,1,3,2,3,3,2,1,1,2,3,3,1,1,1,1,1,3,1,2,3,1,2,2,2,3,1,3,1,1,1,3,3,1,1,3,2,2,1,3,1,3,3,3,3,1,3,2,3,2,3,2,2,1,1,2,2,1,2,2,1,1,2,1,1,1,2,2,3,3,1,1,1,2,1,1,1,1,1,1,2,2,3,1,1,1,1,1,3,1,2,3,2,1,3,3,2,1,1,3,1,3,3,3,2,2,1,1,2,3,1,1,3,3,1,1,1,2,3,2,3,1,1,1,2,3,2,3,1,2,3,1,1,1,3,1,2,3,1,2,2,1,2,1,3,2,3,3,1,1,2,3,3,1,1,1,3,1,3,1,1,2,1,3,3,3,1,2,1,1,1,2,3,1,1,1,1,1,1,1,3,1,2,2,2,1,3,1,3,1,2,3,1,1,1,1,1,1,1,1,1,2,1,1,2,1,3,3,1,2,1,1,1,1,1,3,3,1,1,1,3,2,2,1,3,2,1,1,1,2,1,2,3,2,1,3,2,3,3,1,3,2,3,1,2,1,2,1,1,2,3,2,3,1,3,3,1,1,2,1,1,1,2,2,3,2,3,3,2,1,2,2,2,1,2,1,1,1,3,1,1,1,2,3,1,3,3,3,1,2,1,1,3,3,3,2,1,1,1,2,3,1,2,3,1,3,1,1,1,2,1,1,2,3,1,1,3,1,1,1,1,2,2,3,2,2,3,1,1,3,1,2,2,2,2,1,1,3,3,3,1,3,3,1,2,1,3,1,2,3,2,2,2,2,1,2,1,1,1,3,3,2,1,1,3,1,3,3,2,1,3,2,1,2,2,1,3,2,1,3,2,1,3,1,1,3,1,2,1,3,1,3,3,1,1,2,3,1,3,2,3,1,1,1,2,1,1,1,2,2,3,1,3,3,2,1,1,1,1,3,1,3,3,3,1,3,1,3,2,3,3,2,2,1,1,2,3,1,3,1,3,2,2,1,2,2,1,2,2,1,3,3,3,1,1,3,1,1,1,1,2,2,2,1,3,1,1,2,1,1,3,1,1,2,2,3,2,1,3,1,3,3,2,1,1,2,2,2,3,1,2,2,1,1,2,1,3,2,1,3,2,3,3,1,1,2,3,3,3,2,3,1,3,1,3,1,3,2,2,1,3,3,1,2,2,1,3,3,2,1,3,3,3,1,2,1,1,2,3,2,2,1,1,3,2,3,3,1,3,1,1,2,2,3,1,2,1,3,3,2,2,2,2,1,3,3,2,2,3,3,2,2,3,1,1,1,2,3,1,3,1,1,3,1,1,1,1,3,1,2,2,1,2,1,2,2,1,3,3,3,3,2,3,1,1,2,1,1,1,3,1,2,1,2,2,1,1,3,3,2,2,1,1,1,2,1,2,3,1,1,1,1,3,2,3,1,1,1,3,2,1,1,2,3,1,2,3,1,3,3,2,2,3,1,3,2,2,1,2,1,2,1,1,3,3,3,3,1,2,1,2,1,2,1,1,1,3,1,2,1,1,1,1,2,2,2,1,2,3,1,3,1,1,2,2,2,3,1,2,1,2,2,3,2,2,1,1,1,1,2,3,3,1,3,1,2,3,1,3,1,1,2,1,2,1,2,2,3,1,3,1,1,1,3,1,3,3,2,2,3,1,2,2,2,1,1,1,3,1,2,2,3,1,3,1,1,3,1,3,2,2,1,2,1,3,3,3,3,1,1,1,3,2,3,3,1,1,3,2,2,2,1,2,3,1,2,3,1,1,1,2,1,3,1,3,1,2,1,1,3,1,2,3,1,3,3,1,1,2,3,2,1,2,1,1,3,2,1,1,1,1,2,2,3,3,1,3,2,3,3,1,1,3,1,1,1,3,3,3,2,2,1,2,2,1,2,1,1,2,3,1,3,2,1,2,1,3,2,3,2,1,1,1,1,3,1,3,1,2,1,1,2,2,2,2,1,3,1,2,1,2,2,3,1,1,1,1,2,3,2,3,3,3,3,1,3,3,2,1,1,3,3,2,3,3,3,3,2,3,3,2,1,2,1,1,3,3,1,1,2,1,2,3,3,2,2,2,1,3,3,3,3,1,1,1,1,2,3,1,3,1,2,1,1,3,1,1,3,2,3,3,1,1,2,1,1,1,2,1,3,1,1,2,2,2,3,1,1,1,2,1,2,1,1,3,3,2,1,3,1,2,3,1,1,3,2,1,1,1,3,1,1,2,1,2,2,2,3,3,2,1,3,1,3,2,1,1,1,3,2,1,1,1,2,3,1,3,1,1,3,1,1,1,3,1,3,1,1,1,1,1,2,3,2,3,1,1,3,2,2,2,1,1,1,1,2,1,3,1,3,2,3,1,3,1,1,2,1,1,2,1,1,2,1,2,3,2,1,1,1,2,1,3,3,1,2,3,3,3,2,1,3,2,2,1,1,1,3,1,2,1,2,1,1,1,3,1,3,1,3,1,1,1,1,3,2,1,1,1,2,2,1,2,3,1,2,1,2,1,3,1,1,1,1,3,1,3,3,3,3,2,1,1,1,3,1,1,1,2,1,2,1,1,2,1,3,1,1,3,1,1,1,3,2,3,2,3,1,2,3,1,3,3,3,2,1,2,1,3,1,3,1,1,1,3,1,3,1,3,3,1,3,3,1,3,3,3,1,1,1,3,3,1,2,1,2,1,1,3,2,1,2,1,1,2,3,1,2,1,2,2,1,3,3,1,1,1,2,2,3,1,2,3,1,1,3,1,3,1,3,2,1,1,2,2,1,2,1,1,1,1,3,3,1,1,3,3,3,3,2,3,3,3,1,2,1,2,1,3,2,3,3,3,2,2,2,3,3,2,1,1,1,3,2,3,2,1,3,3,2,3,1,3,3,1,3,2,1],"x0":" ","xaxis":"x","y":[49,34,11,100,45,29,72,22,81,105,45,114,40,112,22,81,21,25,63,62,43,152,29,83,158,48,35,29,41,18,50,81,121,71,80,84,60,45,104,52,43,81,49,8,131,72,44,139,29,93,188,39,125,22,32,105,131,105,45,60,20,115,69,130,85,135,12,29,19,41,40,82,50,18,109,42,78,51,113,29,34,64,121,48,80,161,94,15,74,30,44,22,38,35,69,43,32,49,9,45,82,92,39,81,35,61,73,112,12,38,73,84,78,80,115,82,74,70,81,51,39,98,45,49,128,42,18,31,58,48,54,70,124,84,83,163,42,118,24,22,58,60,15,83,79,41,80,52,70,92,43,21,23,13,112,138,11,171,42,85,44,65,24,11,38,10,135,63,115,45,169,112,93,29,85,21,144,13,165,9,59,29,20,68,65,61,91,31,71,73,20,55,69,155,94,92,125,74,72,61,25,83,104,41,58,93,39,148,53,71,92,61,38,58,80,71,89,83,28,52,70,71,75,78,25,60,99,55,29,53,138,42,99,18,24,55,58,33,74,38,23,61,81,129,33,63,70,45,65,74,50,22,30,21,155,64,65,71,40,69,69,45,62,24,80,8,38,93,55,64,122,70,81,70,150,152,22,22,79,33,128,91,39,121,21,41,45,51,110,20,64,89,80,114,49,191,78,28,42,140,45,122,65,68,153,89,91,55,118,22,63,12,50,43,113,91,55,173,71,24,42,141,32,55,153,113,30,29,35,98,30,58,25,19,44,20,98,90,61,40,38,28,34,73,53,21,40,41,82,155,45,9,48,18,80,64,29,48,23,84,30,93,39,60,133,125,58,89,82,110,54,89,42,41,38,85,83,33,50,58,145,161,38,28,75,35,24,78,113,75,91,31,45,80,50,31,29,59,43,29,91,65,22,71,60,60,69,48,21,53,29,60,42,69,20,200,141,81,35,13,39,21,10,84,30,31,114,53,24,150,132,68,53,30,113,15,84,88,43,43,90,34,38,173,160,44,83,94,43,42,34,39,70,39,44,44,32,15,118,62,82,54,48,74,113,81,43,34,48,54,75,72,39,75,79,205,65,82,21,50,39,101,41,132,43,11,21,109,141,81,54,164,28,68,44,61,35,54,14,52,101,149,35,21,118,38,24,59,41,85,12,32,79,28,41,70,53,128,50,80,93,48,149,49,88,22,55,72,38,114,83,101,94,40,128,39,51,79,48,85,41,125,11,103,141,38,135,28,70,41,81,55,24,81,32,129,108,180,139,33,74,42,73,114,92,34,54,64,42,70,51,71,35,40,18,32,35,120,31,35,32,85,139,103,152,33,84,61,64,34,82,58,92,78,13,81,23,35,79,165,52,49,64,61,112,72,18,41,61,23,22,59,33,11,204,61,103,51,12,82,164,48,38,65,71,18,52,48,30,40,112,115,63,85,59,68,84,62,141,129,30,148,25,21,72,59,83,10,84,98,32,31,94,21,110,38,39,61,92,119,53,18,90,140,90,85,80,53,49,80,82,195,22,81,40,154,49,85,21,14,60,138,98,22,49,102,45,82,81,35,59,52,84,85,65,89,81,38,72,78,71,120,39,23,12,92,42,160,98,164,42,10,31,58,65,98,150,44,15,82,61,38,39,173,8,51,62,34,44,162,64,111,41,52,65,45,35,65,38,42,45,145,33,60,45,72,34,23,45,28,81,72,49,31,63,142,74,24,29,79,94,121,54,73,71,29,93,73,19,58,39,23,29,155,90,113,42,30,19,30,28,15,70,10,22,70,61,161,29,35,48,44,88,40,21,58,59,74,84,58,154,185,78,55,30,74,118,82,29,53,43,59,84,21,63,51,24,95,59,51,28,154,15,55,149,79,68,78,34,184,85,200,64,88,42,91,38,28,32,31,22,74,22,43,43,112,9,81,131,19,62,19,8,30,80,22,33,64,60,81,45,112,180,28,14,43,88,78,68,144,30,120,24,78,44,73,90,58,75,50,85,54,24,45,68,48,114,52,142,15,118,101,84,32,64,81,28,93,42,45,49,69,94,18,8,55,85,62,65,138,103,152,25,98,39,40,58,62,69,40,61,30,29,41,109,50,43,108,20,112,25,42,29,84,81,55,40,78,51,50,21,79,84,34,73,90,23,54,49,51,8,62,25,63,24,11,23,33,39,113,195,105,40,61,75,122,75,59,24,39,145,24,20,85,65,11,40,54,95,94,79,48,70,40,44,43,29,114,52,13,95,84,49,181,70,79,61,121,34,183,51,70,39,84,122,145,98,79,38,49,38,91,29,172,58,191,90,55,28,42,9,65,83,38,13,75,91,99,71,111,108,158,81,12,21,55,42,81,13,132,99,23,40,112,138,113,190,38,32,104,43,45,35,42,58,14,98,90,65,19,35,71,31,43,24,58,115,168,128,138,81,41,71,8,42,62,22,38,24,50,94,43,48,52,84,131,44,78,15,38,84,122,98,80,165,121,20,11,45,14,30,45,80,25,144,84,22,33,60,81,63,12,15,39,44,110,41,59,39,35,80,81,31,93,45,40,80,20,41,31,35,40,84,53,85,138,71,113,23,62,14,60,94,83,61,92,20,194,74,84,98,52,55,64,128,50,54,129,82,100,44,81,71,28,80,90,74,14,23,153,38,112,145,28,31,195,50,71,21,35,42,78,73,53,51,40,178,35,123,48,45,15,79,63,30,32,64,84,21,99,35,180,42,42,82,55,79,48,93,23,50,85,35,29,84,20,14,179,48,74,53,83,171,85,38,85,79,69,43,33,42,40,43,20,65,84,45,109,34,162,105,22,82,105,62,82,72,38,25,20,80,84,33,55,33,43,45,141,69,78,84,51,75,20,63,44,100,12,68,48,82,52,39,40,54,32,64,92,40,60,85,21,35,52,128,195,68,55,55,80,123,90,61,41,43,71,25,141,22,145,49,109,80,82,52,85,148,160,39,25,33,38,85,48,15,94,120,60,145,29,78,59,80,180,8,23,79,78,64,64,43,32,179,8,63,40,99,159,39,99,18,38,133,84,60,18,73,125,74,35,113,52,18,25,50,75,92,33,65,44,183,175,28,49,45,22,120,175,101,104,58,80,45,134,58,20,25,55,155,30,39,73,41,34,21,11,154,19,101,24,131,105,52,21,45,39,192,195,83,34,49,60,41,72,31,73,89,19,63,104,40,92,72,44,73,63,69,42,88,98,12,12,24,25,131,50,21,55,58,72,83,93,38,55,11,85,124,129,180,74,83,33,21,35,105,103,75,33,52,89,63,44,29,29,22,53,24,153,31,21,180,82,90,99,53,18,70,100,125,68,55,84,13,33,89,93,68,85,73,83,24,102,53,175,28,14,34,64,84,42,63,100,22,20,21,25,20,81,119,74,42,73,44,141,14,33,83,89,39,83,108,21,102,48,32,43,35,184,140,32,149,15,71,108,122,40,198,79,84,31,12,88,29,201,20,15,155,25,200,101,19,188,8,59,72,55,24,19,71,54,28,25,51,88,125,79,31,131,70,38,83,59,121,42,32,29,40,25,21,79,52,61,25,25,80,55,23,60,54,44,33,45,55,134,44,149,22,74,62,9,9,15,28,83,73,42,34,32,68,82,19,60,192,190,34,44,23,43,139,118,70,42,143,28,45,78,121,59,64,15,10,61,50,79,35,11,128,55,79,18,91,9,62,59,25,54,125,35,113,25,45,119,30,79,23,171,74,103,99,43,23,91,93,154,21,65,15,52,30,42,20,60,54,60,41,30,21,105,65,20,161,82,43,61,90,90,21,118,44,99,98,69,82,112,123,78,59,78,49,82,185,163,19,115,81,93,184,50,55,13,29,72,83,98,43,34,10,201,85,50,91,112,98,115,22,101,83,52,54,69,64,53,19,31,63,25,22,78,81,41,62,35,30,51,140,73,40,69,82,38,23,55,58,29,39,39,30,58,62,62,19,38,84,78,22,81,32,89,121,42,130,30,114,155,50,34,94,114,89,49,115,148,72,114,11,63,113,80,54,32,11,141,84,18,20,55,31,114,52,18,55,29,30,38,122,183,41,38,61,80,22,124,44,75,78,31,99,41,39,51,52,141,93,14,160,44,33,71,55,79,80,179,9,61,111,80,32,83,91,29,19,52,150,32,91,121,25,102,44,43,94,29,38,120,85,89,25,20,125,18,173,34,55,32,65,83,30,180,13,35,95,83,39,54,81,53,119,21,40,69,45,55,38,44,45,84,29,95,79,41,19,75,64,54,14,94,10,145,81,153,58,49,132,85,85,104,40,103,25,69,30,70,15,125,50,44,38,64,9,44,93,83,179,40,58,74,15,11,41,68,28,83,40,113,89,38,62,65,40,30,20,58,150,54,19,30,123,45,154,93,71,55,38,74,69,104,38,25,162,64,30,52,42,79,34,123,160,51,108,68,42,32,22,29,69,92,40,9,128,110,171,13,25,33,34,58,51,29,111,43,130,20,101,48,92,33,51,31,83,45,89,89,9,58,80,73,41,74,29,25,98,59,43,72,134,140,39,35,141,23,134,35,42,81,14,45,79,53,33,60,8,38,154,54,41,25,53,79,13,130,93,82,15,28,9,88,143,62,51,101,90,83,21,115,32,204,34,30,31,90,79,22,42,29,72,68,78,90,14,129,39,82,31,9,15,70,84,155,89,185,82,60,58,29,32,115,83,54,112,112,23,198,84,39,129,113,35,85,73,70,39,72,68,35,61,69,59,99,130,72,44,91,92,55,20,59,8,85,94,22,74,59,12,8,74,34,63,85,31,61,88,22,63,80,78,45,51,111,45,41,69,23,160,61,110,50,49,52,53,72,51,23,41,41,138,14,145,25,43,93,90,41,63,23,75,13,10,100,85,73,130,65,59,31,134,11,32,53,28,44,40,40,163,82,89,78,55,34,58,73,54,45,93,102,39,114,29,30,22,38,41,38,25,70,201,42,64,28,119,28,60,23,39,19,72,41,32,69,23,44,32,25,80,83,133,89,81,44,119,39,40,79,30,178,82,50,64,129,68,30,48,130,40,105,41,80,134,49,63,70,32,14,111,38,121,90,20,61,43,39,60,63,8,52,111,71,25,34,74,120,50,10,45,29,145,100,24,60,50,43,79,44,31,14,19,90,21,75,175,41,79,24,39,9,15,80,63,20,43,65,91,53,110,82,23,172,88,71,39,94,25,10,42,142,34,139,154,40,33,80,10,41,30,78,164,23,149,25,42,71,70,55,29,123,8,69,79,105,39,163,9,71,51,32,60,35,41,73,49,79,120,60,78,40,68,49,65,55,62,152,42,39,82,108,59,61,78,18,113,62,81,13,75,33,179,69,45,59,81,133,81,14,28,93,155,155,68,52,51,44,121,49,42,144,72,44,43,74,145,158,78,105,100,58,113,10,142,128,88,101,70,122,70,43,31,35,81,19,98,101,81,135,34,104,29,82,62,85,8,78,44,122,22,39,55,144,71,191,22,158,75,69,39,34,32,129,54,34,45,68,53,32,31,72,30,138,18,69,58,63,43,28,38,72,64,70,12,159,103,42,48,49,89,22,10,38,49,30,82,82,24,32,34,61,22,40,50,35,64,49,44,65,13,84,28,108,59,163,69,115,118,163,140,91,61,69,22,53,25,31,83,152,30,34,44,182,44,122,35,91,30,53,28,85,28,41,41,33,59,129,35,45,62,14,53,48,128,105,40,10,102,31,30,21,44,48,21,84,64,82,39,45,152,12,32,74,54,75,95,90,20,64,85,105,21,78,158,125,81,183,54,79,38,158,20,60,113,70,84,23,141,110,31,22,91,82,23,35,18,80,74,73,40,55,69,92,48,58,109,39,33,18,63,43,70,19,39,81,34,114,55,21,50,41,8,55,40,22,44,54,75,42,133,40,34,142,64,44,95,49,12,98,42,44,124,175,22,78,54,154,42,195,80,81,89,61,25,43,122,160,83,79,30,145,151,14,19,55,182,78,42,44,53,65,43,38,50,81,41,73,31,84,95,60,43,38,82,52,43,82,161,24,102,53,85,111,90,94,63,39,205,142,163,113,64,20,179,91,80,38,44,81,95,44,88,11,92,74,72,63,172,183,40,68,78,68,49,39,151,159,89,105,78,128,81,50,38,119,110,29,19,38,31,14,39,33,92,78,42,15,48,63,149,49,125,58,48,75,114,111,62,45,113,128,81,75,21,83,121,40,101,90,68,74,74,51,149,31,85,62,44,63,23,40,78,40,78,71,23,15,28,23,19,42,113,34,19,43,44,21,115,81,91,30,55,48,23,21,55,21,40,30,78,65,50,55,111,21,65,114,54,75,13,54,22,198,18,74,61,64,10,82,61,83,73,22,91,65,13,103,95,31,38,104,38,30,19,25,20,83,18,62,24,13,64,62,88,28,52,82,140,80,58,180,55,65,62,23,35,35,74,14,83,71,53,103,109,60,12,141,128,98,41,90,31,74,83,35,140,65,35,73,29,55,20,70,81,25,119,49,38,80,84,61,60,83,59,32,33,19,14,154,33,28,44,81,29,58,38,82,18,45,64,48,24,65,111,114,39,19,22,49,40,63,55,33,31,24,81,81,113,92,81,101,78,78,35,9,82,41,84,42,54,84,131,85,84,114,61,39,21,68,85,180,110,65,81,132,44,98,62,91,29,101,65,58,32,44,113,38,13,62,29,23,11,90,43,59,28,28,108,78,65,25,108,43,95,24,190,24,41,80,63,79,114,120,104,20,41,35,22,132,32,21,49,79,30,118,11,61,43,54,33,84,110,14,65,78,18,95,95,191,19,19,39,93,112,40,45,133,85,18,54,41,21,152,53,84,59,33,81,61,88,170,83,44,83,21,143,103,162,73,25,115,88,73,94,58,81,113,35,65,170,62,54,39,28,95,21,50,34,19,71,113,9,118,79,72,18,93,62,23,40,34,114,125,41,38,45,12,115,175,24,28,32,63,32,60,53,58,42,39,72,80,39,18,30,54,128,38,120,25,85,14,70,23,31,132,28,91,191,88,84,64,28,149,42,53,113,79,175,38,52,42,13,141,131,64,49,34,91,195,53,39,20,104,51,38,12,28,33,31,44,49,134,114,90,105,58,53,15,103,58,83,61,69,89,70,39,41,148,38,129,84,29,30,90,150,15,13,9,24,41,25,49,58,141,50,61,65,112,19,23,60,30,194,109,174,34,84,41,92,19,55,22,155,72,81,30,54,60,68,65,31,29,43,80,43,194,24,32,30,64,15,70,32,84,62,33,65,33,15,45,132,29,65,52,64,73,20,44,79,104,88,51,61,23,38,75,42,45,59,9,62,202,162,93,145,149,69,30,48,42,132,81,60,45,70,71,28,83,131,82,79,51,111,93,73,24,64,104,38,191,53,74,21,80,41,79,32,38,85,93,69,32,61,125,30,104,24,22,59,164,60,43,70,51,59,22,29,18,65,65,38,110,22,43,30,134,32,139,45,62,51,93,122,45,179,64,41,58,64,28,21,60,78,59,59,83,128,95,78,30,51,35,74,43,23,20,112,74,65,124,65,73,8,53,13,60,45,39,78,43,31,82,112,10,18,72,72,78,78,30,45,80,174,51,181,78,79,119,49,100,70,52,12,82,34,112,42,83,199,29,70,25,84,81,15,54,83,62,134,75,109,152,83,52,31,98,14,64,109,118,80,51,83,90,28,82,24,71,80,109,140,39,39,50,31,35,44,35,20,10,83,203,84,41,164,152,62,12,28,52,19,35,28,70,140,102,50,61,44,12,69,132,128,44,44,34,132,158,83,28,31,40,71,42,35,81,91,75,54,43,94,84,34,15,99,15,83,83,143,81,39,62,64,29,64,21,64,40,61,44,78,25,125,63,54,92,119,40,41,81,101,25,55,43,98,112,69,79,118,81,19,63,102,40,44,59,224,64,129,43,12,53,39,65,18,55,60,42,44,111,62,44,69,62,35,91,12,89,65,82,48,20,91,122,22,43,38,61,33,145,170,55,40,68,53,63,123,13,12,53,79,89,62,40,23,62,69,124,52,134,58,45,75,23,48,145,31,38,78,94,84,22,78,75,71,24,112,53,41,25,42,51,19,90,89,22,119,18,32,19,90,59,84,13,88,34,11,80,41,90,81,60,39,22,32,28,35,189,40,88,124,23,23,139,42,161,28,42,81,82,53,42,64,90,44,29,93,69,82,13,72,54,104,44,40,190,64,9,38,72,82,48,82,14,162,54,90,64,65,113,38,65,39,174,59,43,24,50,68,61,21,21,31,70,25,78,19,23,40,22,71,58,75,40,83,99,59,53,11,71,85,49,122,171,8,59,49,92,75,60,41,40,81,52,25,139,54,81,73,30,59,21,65,41,52,84,135,18,34,85,23,65,195,40,95,23,82,53,10,75,23,133,129,85,168,83,52,114,81,32,63,70,20,138,63,85,80,53,72,91,148,28,193,22,74,70,11,61,41,23,30,44,49,75,43,59,75,75,88,11,22,49,140,99,82,10,109,45,93,39,21,31,31,52,13,21,50,81,74,89,82,88,61,139,29,51,58,21,43,104,9,39,89,21,29,43,30,39,83,58,8,39,204,45,111,83,83,115,60,53,134,91,91,128,78,161,60,161,40,68,74,55,145,24,10,138,63,54,62,81,61,68,78,165,48,52,18,42,18,40,44,11,108,12,19,150,89,83,79,102,155,138,51,80,135,34,62,173,149,20,42,28,95,95,34,41,150,91,84,73,122,73,20,134,98,54,41,14,93,79,79,62,95,60,75,85,143,34,44,31,13,52,42,102,10,44,158,12,59,92,82,44,121,38,92,53,45,63,181,33,22,99,44,101,32,59,25,29,35,75,58,41,42,55,18,75,85,43,149,25,19,18,13,41,139,62,51,71,45,53,64,39,38,85,61,23,122,72,123,40,62,113,81,59,25,25,14,19,61,21,44,42,52,112,34,83,24,71,75,153,75,19,31,65,8,92,85,149,9,40,54,140,18,12,140,24,60,50,44,51,34,33,63,22,75,75,62,48,73,135,61,78,40,44,45,18,53,11,38,28,35,81,90,115,78,30,39,69,29,54,60,99,51,53,79,90,18,73,59,40,41,53,128,35,28,155,195,50,48,81,50,21,142,64,91,74,82,45,85,133,22,24,38,80,33,41,90,89,8,85,53,20,9,25,114,41,49,12,50,30,32,41,25,29,50,79,110,40,18,48,40,19,35,19,83,39,38,63,85,14,64,124,20,33,80,35,74,32,73,41,18,28,64,44,109,71,79,30,82,101,13,59,64,188,28,131,19,44,49,99,95,81,73,193,41,80,90,41,102,59,69,49,61,59,85,13,44,53,31,101,61,68,21,149,75,55,81,179,42,199,18,28,80,79,52,18,52,15,84,114,41,29,23,84,60,13,45,83,58,148,92,62,50,29,85,78,85,89,33,42,31,35,22,45,119,43,75,121,63,58,155,69,38,9,69,199,38,129,59,115,83,65,72,72,63,64,194,104,180,18,93,32,68,38,115,161,154,68,44,149,103,41,113,48,69,43,19,65,32,20,25,70,70,103,61,170,49,62,102,8,91,10,35,71,59,14,122,81,35,83,121,35,88,70,63,125,21,75,38,58,31,18,29,29,69,121,64,84,65,85,44,174,178,55,25,21,72,129,18,91,38,21,38,154,178,85,93,151,78,135,40,80,18,74,94,88,82,54,45,39,20,93,85,95,48,22,33,40,30,22,20,49,25,150,32,9,23,18,42,39,84,43,28,59,29,39,42,24,62,33,73,88,58,48,40,160,39,53,32,32,11,60,41,99,83,134,85,41,44,62,149,25,84,81,181,62,11,60,42,83,109,70,25,72,183,44,52,18,40,41,81,128,65,145,65,119,124,33,41,105,130,80,115,62,34,18,44,25,52,71,61,62,33,28,53,69,165,75,54,53,41,43,79,24,73,50,28,64,49,121,58,51,61,35,39,74,201,81,44,19,85,74,54,58,18,88,19,49,101,138,25,120,153,28,110,94,48,123,33,42,68,28,42,28,73,89,82,83,28,39,110,55,111,94,85,81,94,162,80,70,19,109,12,33,49,42,41,43,64,19,65,53,32,22,39,59,50,19,55,81,108,98,110,33,34,69,41,60,95,73,41,68,64,29,31,54,63,92,195,45,72,98,30,78,43,85,38,58,92,13,218,75,40,15,24,49,83],"y0":" ","yaxis":"y","type":"box"},{"alignmentgroup":"True","hovertemplate":"Personal Loan=1<br>Education=%{x}<br>Income=%{y}<extra></extra>","legendgroup":"","marker":{"color":"#636efa"},"name":"","notched":false,"offsetgroup":"","orientation":"v","showlegend":false,"x":[3,3,3,2,3,2,3,3,3,2,3,2,2,3,3,2,1,3,3,1,3,2,2,3,3,3,3,1,2,2,2,1,1,1,1,1,2,2,1,3,1,2,2,3,3,2,3,2,2,2,2,1,3,3,2,3,3,2,3,3,3,2,2,3,1,2,2,3,2,1,2,3,3,3,3,3,3,2,2,1,2,3,1,1,2,1,2,3,1,3,3,2,3,1,2,2,3,3,2,2,3,2,3,3,1,2,3,1,3,3,1,2,1,2,1,3,3,3,1,2,2,1,2,2,2,3,2,3,2,3,2,2,2,1,2,2,2,1,3,3,3,2,2,3,3,3,2,2,1,2,3,2,2,3,3,2,1,1,1,1,2,2,3,3,3,2,3,2,2,3,3,3,3,3,2,1,2,1,1,3,1,1,1,3,3,1,2,2,3,3,1,3,1,1,2,1,3,3,3,3,2,1,2,3,3,1,2,2,2,2,3,2,3,3,3,3,2,1,3,2,3,2,2,3,2,2,3,2,2,2,3,2,1,3,2,1,2,3,1,2,1,2,2,3,2,2,2,2,3,1,2,2,3,3,3,3,2,1,2,2,1,2,2,3,2,1,3,3,1,3,3,1,3,2,3,3,2,3,1,3,2,3,1,2,2,2,3,3,2,3,3,2,2,2,1,3,3,2,3,2,1,3,2,3,3,1,3,2,1,2,2,2,3,2,3,2,3,2,1,1,3,3,3,3,2,2,1,3,2,2,3,1,2,2,1,3,2,3,3,3,3,2,2,1,3,2,2,3,1,3,2,3,1,3,3,3,1,3,3,3,1,1,2,3,1,3,2,2,3,2,3,3,2,3,2,2,2,2,3,2,2,3,3,2,2,3,3,3,2,3,3,2,1,3,3,3,2,3,3,1,3,1,1,3,3,3,2,3,2,1,3,2,3,2,1,3,1,3,2,3,2,3,2,3,3,1,3,3,1,3,2,3,2,3,1,1,3,2,1,1,3,3,3,2,3,3,2,3,3,1,3,2,2,2,2,2,2,2,2,1,2,1,3,1,3,2,3,3,3,3,2,2,3,2,3,3,2,2,3,1],"x0":" ","xaxis":"x2","y":[180,130,193,119,141,132,194,190,131,135,133,118,149,132,134,168,148,159,158,172,170,120,170,134,111,172,159,195,165,119,101,101,99,158,173,60,155,174,133,100,179,115,113,122,183,149,83,182,152,173,69,122,131,175,161,152,170,115,165,122,129,185,105,164,109,132,114,152,138,109,172,165,135,192,158,194,202,132,130,150,124,182,120,161,172,195,95,193,129,83,169,135,174,83,142,185,99,192,118,122,110,145,175,145,83,138,125,75,175,191,188,182,104,171,73,134,143,120,181,94,112,103,71,123,190,165,118,169,130,130,113,164,192,145,179,180,171,139,135,179,190,172,183,101,129,184,154,154,160,141,178,91,191,64,155,102,114,65,122,170,184,124,192,138,111,84,154,153,93,193,152,158,182,139,190,173,139,114,125,155,128,119,171,113,98,185,112,194,161,178,179,110,125,111,191,134,118,191,181,103,128,142,170,169,103,113,180,161,134,159,160,188,174,203,178,115,168,83,99,158,183,123,189,180,193,162,183,190,150,131,168,105,172,114,170,170,182,129,130,201,89,184,123,109,184,185,125,138,169,140,108,123,161,164,179,168,178,92,111,104,98,171,90,180,114,149,152,161,161,110,160,111,191,154,131,148,119,131,158,173,195,183,130,185,168,115,154,182,185,183,133,181,190,172,184,188,114,122,113,140,190,102,130,132,195,162,121,153,103,164,148,169,130,158,141,134,168,129,179,131,175,114,124,99,158,173,94,111,179,93,159,140,141,132,105,153,174,164,141,125,115,163,182,135,142,82,135,125,172,149,120,133,154,182,125,95,144,118,169,139,153,165,128,130,99,148,154,141,140,120,129,160,184,131,118,83,142,108,135,115,188,178,131,129,158,142,139,179,188,123,119,123,123,93,182,85,95,93,154,173,191,175,142,180,155,179,138,141,180,123,163,108,135,144,139,143,154,91,89,173,155,158,194,195,191,130,85,128,132,188,170,143,182,184,140,141,145,181,178,145,164,164,104,111,145,182,133,120,115,83,180,151,133,81,102,134,130,142,143,184,188,162,108,141,140,173,101,123,184,115,174,135,165,133,142,129,121,112,122,135],"y0":" ","yaxis":"y2","type":"box"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"xaxis":{"anchor":"y","domain":[0.0,0.49],"title":{"text":"Education"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Income"}},"xaxis2":{"anchor":"y2","domain":[0.51,1.0],"matches":"x","title":{"text":"Education"}},"yaxis2":{"anchor":"x2","domain":[0.0,1.0],"matches":"y","showticklabels":false},"annotations":[{"font":{},"showarrow":false,"text":"Personal Loan=0","x":0.245,"xanchor":"center","xref":"paper","y":1.0,"yanchor":"bottom","yref":"paper"},{"font":{},"showarrow":false,"text":"Personal Loan=1","x":0.755,"xanchor":"center","xref":"paper","y":1.0,"yanchor":"bottom","yref":"paper"}],"legend":{"tracegroupgap":0},"margin":{"t":60},"boxmode":"group"},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('caf4f5c2-4d25-4d22-8874-e17e21d359f5');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>


#### Inference:From the above plot we could say that Income of customers who availed personal loan are alomst same irrescpective of their Education


```python
plt.figure(figsize=(12, 8))
sns.distplot(data[data['Personal Loan'] == 0]['Income'], hist=False, label='Income with no personal loan')
sns.distplot(data[data['Personal Loan'] == 1]['Income'], hist=False, label='Income with personal loan')
plt.legend()
```

    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2619: FutureWarning:
    
    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `kdeplot` (an axes-level function for kernel density plots).
    
    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2619: FutureWarning:
    
    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `kdeplot` (an axes-level function for kernel density plots).
    





    <matplotlib.legend.Legend at 0x7f392c9b6590>




    
![png](Bank_Loan_files/Bank_Loan_47_2.png)
    


#### Conclusion: Customers Who have availed personal loan seem to have higher income than those who do not have personal loan

### automate above stuffs


```python
def plot(col1, col2, label1, label2, title):
    plt.figure(figsize=(12, 8))
    sns.distplot(data[data[col2] == 0][col1], hist=False, label=label1)
    sns.distplot(data[data[col2] == 1][col1], hist=False, label=label2)
    plt.legend()
    plt.title(title)
```


```python
plot('Income', 'Personal Loan', 'Income with no personal loan', 'Income with personal loan', 'Income Distribution')
```

    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2619: FutureWarning:
    
    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `kdeplot` (an axes-level function for kernel density plots).
    
    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2619: FutureWarning:
    
    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `kdeplot` (an axes-level function for kernel density plots).
    



    
![png](Bank_Loan_files/Bank_Loan_51_1.png)
    



```python
plot('CCAvg', 'Personal Loan', 'Credit card avg with no personal loan', 'Credit card avg with personal loan', 'Credit card avg Distribution')
```

    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2619: FutureWarning:
    
    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `kdeplot` (an axes-level function for kernel density plots).
    
    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2619: FutureWarning:
    
    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `kdeplot` (an axes-level function for kernel density plots).
    



    
![png](Bank_Loan_files/Bank_Loan_52_1.png)
    



```python
plot('Mortgage', 'Personal Loan', 'Mortgage of customers with no personal loan', 'Mortgage of customers  with personal loan', 'Mortgage of customers  Distribution')
```

    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2619: FutureWarning:
    
    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `kdeplot` (an axes-level function for kernel density plots).
    
    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2619: FutureWarning:
    
    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `kdeplot` (an axes-level function for kernel density plots).
    



    
![png](Bank_Loan_files/Bank_Loan_53_1.png)
    


#### People with high mortgage value, i.e more than 400K have availed personal Loan


```python
data.columns
```




    Index(['Age', 'Income', 'Family', 'CCAvg', 'Education', 'Mortgage',
           'Personal Loan', 'Securities Account', 'CD Account', 'Online',
           'CreditCard', 'Edu_mark', 'Account_holder_category'],
          dtype='object')




```python
col_names = ['Securities Account', 'Online', 'Account_holder_category', 'CreditCard']
for i in col_names:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=i, hue='Personal Loan', data=data)
```


    
![png](Bank_Loan_files/Bank_Loan_56_0.png)
    



    
![png](Bank_Loan_files/Bank_Loan_56_1.png)
    



    
![png](Bank_Loan_files/Bank_Loan_56_2.png)
    



    
![png](Bank_Loan_files/Bank_Loan_56_3.png)
    



```python

```


    
![png](Bank_Loan_files/Bank_Loan_57_0.png)
    



    
![png](Bank_Loan_files/Bank_Loan_57_1.png)
    



    
![png](Bank_Loan_files/Bank_Loan_57_2.png)
    



    
![png](Bank_Loan_files/Bank_Loan_57_3.png)
    


#### From the above graph we could infer that , customers who hold deposit account & customers who do not hold either a securities account or deposit account have aviled personal loan

#### Perform Hypothesis Testing


### Q.. How Age of a person is going to be a factor in availing loan ??? Does Income of a person have an impact on availing loan ??? Does the family size makes them to avail loan ???¶


```python
sns.scatterplot(data['Age'], data['Personal Loan'], hue=data['Family'])
```

    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning:
    
    Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
    





    <matplotlib.axes._subplots.AxesSubplot at 0x7f392c92d410>




    
![png](Bank_Loan_files/Bank_Loan_61_2.png)
    



```python
import scipy.stats as stats
```


```python
Ho = 'Age does not have impact on availing personal loan'
Ha = 'Age does  have impact on availing personal loan'

Age_no = np.array(data[data['Personal Loan'] == 0]['Age'])
Age_yes = np.array(data[data['Personal Loan'] == 1]['Age'])

t, p_value = stats.ttest_ind(Age_no, Age_yes, axis=0)
if p_value < 0.05:
    print(Ha, ' as the p_value is less than 0.05 with a value of {}'.format(p_value))
else:
    print(Ho, ' as the p_value is greater than 0.05 with a value of {}'.format(p_value))
```

    Age does not have impact on availing personal loan  as the p_value is greater than 0.05 with a value of 0.584959263705325


#### automate above stuffs


```python
def Hypothesis(col1, col2, HO, Ha):
    arr1 = np.array(data[data[col1] == 0][col2])
    arr2 = np.array(data[data[col1] == 1][col2])
    t, p_value = stats.ttest_ind(arr1, arr2, axis=0)
    if p_value < 0.05:
        print('{}, as the p_value is less than 0.05 with a value of {}'.format(Ha,p_value))
    else:
        print('{} as the p_value is greater than 0.05 with a value of {}'.format(HO,p_value))
```


```python
Hypothesis('Personal Loan', 'Age', HO='Age does not have impact on availing personal loan', Ha='Age does  have impact on availing personal loan')
```

    Age does not have impact on availing personal loan as the p_value is greater than 0.05 with a value of 0.584959263705325


### Q..Income of a person has significant impact on availing Personal Loan or not?


```python
Hypothesis(col1='Personal Loan', col2='Income', HO='Income does not have impact on availing personal loan', Ha='Income does  have impact on availing personal loan')
```

    Income does  have impact on availing personal loan, as the p_value is less than 0.05 with a value of 0.0


#### Income have phenomenal significance on availing personal Loan , As the P_value is less than 0.05 with a value of :0.0

### Q..Number of persons in the family has significant impact on availing Personal Loan or not?


```python
Hypothesis('Personal Loan', 'Family', HO='AgFamily does not have impact on availing personal loan', Ha='Family does  have impact on availing personal loan')
```

    Family does  have impact on availing personal loan, as the p_value is less than 0.05 with a value of 1.4099040685673807e-05


#### Family have phenomenal significance on availing personal Loan , As the P_value is less than 0.05 with a value of :1.4099040685673807e-05
