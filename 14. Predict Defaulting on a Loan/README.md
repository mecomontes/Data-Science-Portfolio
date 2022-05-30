# How can I compare different models that predict the probability of defaulting on a loan?


```python
### Load relevant packages
import pandas                  as pd
import numpy                   as np
import matplotlib.pyplot       as plt
import seaborn                 as sns
import os

from scipy import stats

%matplotlib inline
plt.style.use('ggplot')

import warnings
warnings.filterwarnings('ignore')
```

## Introduction (5 mts)

**Business Context.** Traditional commercial banks typically did not rely on statistical modeling to decide whether personal loans should be issued, although this is changing rapidly nowadays. You are a data scientist working in a modern commercial bank. Your data science team has already built simple regression models for predicting the probability that those loans would be defaulted on. However, you have noticed that many of these models perform much worse in production than they do in testing.

**Business Problem.** Your task is to **build a default probability model that you feel comfortable putting into production.**

**Analytical Context.** The dataset contains the details of 5000 loans requests that have been previously issued by your bank. For each loan, the final status of the loan (i.e. whether the loan defaulted) is also available:

1. The file **"loan_light.csv"** contains the details of 5000 loans
2. The file **"loan_param.xlsx"** contains the description of each covariate

The case will proceed as follows: you will 1) perform some data exploration to determine the appropriate variable transformations to make; 2) fit some simple models; 3) learn about **cross-validation** and use this to select the best simple model; and finally 4) responsibly construct more complex models using cross-validation.

## Data exploration (40 mts)

Let's start by taking a look at the data:


```python
Data = pd.read_csv("/content/loan_light.csv")
Data = Data.sample(frac=1)  #shuffle the rows
```


```python
print(Data.shape)
Data.head()
```

    (5000, 21)





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
      <th>annual_inc</th>
      <th>application_type</th>
      <th>avg_cur_bal</th>
      <th>chargeoff_within_12_mths</th>
      <th>delinq_2yrs</th>
      <th>dti</th>
      <th>emp_length</th>
      <th>grade</th>
      <th>inq_last_12m</th>
      <th>installment</th>
      <th>loan_amnt</th>
      <th>num_actv_bc_tl</th>
      <th>pub_rec_bankruptcies</th>
      <th>home_ownership</th>
      <th>term</th>
      <th>mort_acc</th>
      <th>num_tl_90g_dpd_24m</th>
      <th>purpose</th>
      <th>year</th>
      <th>loan_default</th>
      <th>job</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2790</th>
      <td>38000.0</td>
      <td>Individual</td>
      <td>2375.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12.19</td>
      <td>1</td>
      <td>C</td>
      <td>1.0</td>
      <td>601.36</td>
      <td>17100</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>OWN</td>
      <td>36</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>debt_consolidation</td>
      <td>2017</td>
      <td>1</td>
      <td>other</td>
    </tr>
    <tr>
      <th>1822</th>
      <td>55000.0</td>
      <td>Joint App</td>
      <td>20371.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>21.19</td>
      <td>1</td>
      <td>C</td>
      <td>1.0</td>
      <td>586.45</td>
      <td>17500</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>MORTGAGE</td>
      <td>36</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>credit_card</td>
      <td>2017</td>
      <td>0</td>
      <td>professor</td>
    </tr>
    <tr>
      <th>2857</th>
      <td>45000.0</td>
      <td>Individual</td>
      <td>2437.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>17.09</td>
      <td>10</td>
      <td>A</td>
      <td>0.0</td>
      <td>308.64</td>
      <td>10000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>RENT</td>
      <td>36</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>credit_card</td>
      <td>2016</td>
      <td>0</td>
      <td>teacher</td>
    </tr>
    <tr>
      <th>4776</th>
      <td>125000.0</td>
      <td>Individual</td>
      <td>10340.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>14.10</td>
      <td>1</td>
      <td>D</td>
      <td>4.0</td>
      <td>874.93</td>
      <td>35000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>MORTGAGE</td>
      <td>60</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>debt_consolidation</td>
      <td>2015</td>
      <td>0</td>
      <td>manager</td>
    </tr>
    <tr>
      <th>4482</th>
      <td>70000.0</td>
      <td>Individual</td>
      <td>2089.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9.12</td>
      <td>3</td>
      <td>C</td>
      <td>0.0</td>
      <td>848.27</td>
      <td>25000</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>RENT</td>
      <td>36</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>debt_consolidation</td>
      <td>2017</td>
      <td>0</td>
      <td>other</td>
    </tr>
  </tbody>
</table>
</div>




```python
Data.keys()
```




    Index(['annual_inc', 'application_type', 'avg_cur_bal',
           'chargeoff_within_12_mths', 'delinq_2yrs', 'dti', 'emp_length', 'grade',
           'inq_last_12m', 'installment', 'loan_amnt', 'num_actv_bc_tl',
           'pub_rec_bankruptcies', 'home_ownership', 'term', 'mort_acc',
           'num_tl_90g_dpd_24m', 'purpose', 'year', 'loan_default', 'job'],
          dtype='object')




```python
df_description = pd.read_excel('loan_param.xlsx').dropna()
df_description.style.set_properties(subset=['Description'], **{'width': '1000px'})
```




<style  type="text/css" >
#T_af79a8c0_0278_11ec_8aca_0242ac1c0002row0_col1,#T_af79a8c0_0278_11ec_8aca_0242ac1c0002row1_col1,#T_af79a8c0_0278_11ec_8aca_0242ac1c0002row2_col1,#T_af79a8c0_0278_11ec_8aca_0242ac1c0002row3_col1,#T_af79a8c0_0278_11ec_8aca_0242ac1c0002row4_col1,#T_af79a8c0_0278_11ec_8aca_0242ac1c0002row5_col1,#T_af79a8c0_0278_11ec_8aca_0242ac1c0002row6_col1,#T_af79a8c0_0278_11ec_8aca_0242ac1c0002row7_col1,#T_af79a8c0_0278_11ec_8aca_0242ac1c0002row8_col1,#T_af79a8c0_0278_11ec_8aca_0242ac1c0002row9_col1,#T_af79a8c0_0278_11ec_8aca_0242ac1c0002row10_col1,#T_af79a8c0_0278_11ec_8aca_0242ac1c0002row11_col1,#T_af79a8c0_0278_11ec_8aca_0242ac1c0002row12_col1,#T_af79a8c0_0278_11ec_8aca_0242ac1c0002row13_col1,#T_af79a8c0_0278_11ec_8aca_0242ac1c0002row14_col1,#T_af79a8c0_0278_11ec_8aca_0242ac1c0002row15_col1,#T_af79a8c0_0278_11ec_8aca_0242ac1c0002row16_col1,#T_af79a8c0_0278_11ec_8aca_0242ac1c0002row17_col1,#T_af79a8c0_0278_11ec_8aca_0242ac1c0002row18_col1,#T_af79a8c0_0278_11ec_8aca_0242ac1c0002row19_col1,#T_af79a8c0_0278_11ec_8aca_0242ac1c0002row20_col1{
            width:  1000px;
        }</style><table id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >BrowseNotesFile</th>        <th class="col_heading level0 col1" >Description</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row0_col0" class="data row0 col0" >loanAmnt</td>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row0_col1" class="data row0 col1" >The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.</td>
            </tr>
            <tr>
                        <th id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row1_col0" class="data row1 col0" >annualInc</td>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row1_col1" class="data row1 col1" >The self-reported annual income provided by the borrower during registration.</td>
            </tr>
            <tr>
                        <th id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row2_col0" class="data row2 col0" >application_type</td>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row2_col1" class="data row2 col1" >Indicates whether the loan is an individual application or a joint application with two co-borrowers</td>
            </tr>
            <tr>
                        <th id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row3_col0" class="data row3 col0" >avg_cur_bal</td>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row3_col1" class="data row3 col1" >Average current balance of all accounts</td>
            </tr>
            <tr>
                        <th id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row4_col0" class="data row4 col0" >chargeoff_within_12_mths</td>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row4_col1" class="data row4 col1" >Number of charge-offs within 12 months</td>
            </tr>
            <tr>
                        <th id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002level0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row5_col0" class="data row5 col0" >delinq2Yrs</td>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row5_col1" class="data row5 col1" >The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years</td>
            </tr>
            <tr>
                        <th id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002level0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row6_col0" class="data row6 col0" >dti</td>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row6_col1" class="data row6 col1" >A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.</td>
            </tr>
            <tr>
                        <th id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002level0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row7_col0" class="data row7 col0" >emp_length</td>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row7_col1" class="data row7 col1" >Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.</td>
            </tr>
            <tr>
                        <th id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002level0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row8_col0" class="data row8 col0" >grade</td>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row8_col1" class="data row8 col1" >LC assigned loan grade</td>
            </tr>
            <tr>
                        <th id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002level0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row9_col0" class="data row9 col0" >homeOwnership</td>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row9_col1" class="data row9 col1" >The home ownership status provided by the borrower during registration. Our values are: RENT, OWN, MORTGAGE, OTHER.</td>
            </tr>
            <tr>
                        <th id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002level0_row10" class="row_heading level0 row10" >10</th>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row10_col0" class="data row10 col0" >inq_last_12m</td>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row10_col1" class="data row10 col1" >Number of credit inquiries in past 12 months</td>
            </tr>
            <tr>
                        <th id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002level0_row11" class="row_heading level0 row11" >11</th>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row11_col0" class="data row11 col0" >installment</td>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row11_col1" class="data row11 col1" >The monthly payment owed by the borrower if the loan originates.</td>
            </tr>
            <tr>
                        <th id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002level0_row12" class="row_heading level0 row12" >12</th>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row12_col0" class="data row12 col0" >job</td>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row12_col1" class="data row12 col1" >Job Description</td>
            </tr>
            <tr>
                        <th id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002level0_row13" class="row_heading level0 row13" >13</th>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row13_col0" class="data row13 col0" >loanAmnt</td>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row13_col1" class="data row13 col1" >The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.</td>
            </tr>
            <tr>
                        <th id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002level0_row14" class="row_heading level0 row14" >14</th>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row14_col0" class="data row14 col0" >loanDefault</td>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row14_col1" class="data row14 col1" >0: Loan was uptimated paid in full. 1: A default even occurred</td>
            </tr>
            <tr>
                        <th id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002level0_row15" class="row_heading level0 row15" >15</th>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row15_col0" class="data row15 col0" >mortAcc</td>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row15_col1" class="data row15 col1" >Number of mortgage accounts.</td>
            </tr>
            <tr>
                        <th id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002level0_row16" class="row_heading level0 row16" >16</th>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row16_col0" class="data row16 col0" >num_tl_90g_dpd_24m</td>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row16_col1" class="data row16 col1" >Number of accounts 90 or more days past due in last 24 months</td>
            </tr>
            <tr>
                        <th id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002level0_row17" class="row_heading level0 row17" >17</th>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row17_col0" class="data row17 col0" >pub_rec_bankruptcies</td>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row17_col1" class="data row17 col1" >Number of public record bankruptcies</td>
            </tr>
            <tr>
                        <th id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002level0_row18" class="row_heading level0 row18" >18</th>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row18_col0" class="data row18 col0" >purpose</td>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row18_col1" class="data row18 col1" >A category provided by the borrower for the loan request. </td>
            </tr>
            <tr>
                        <th id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002level0_row19" class="row_heading level0 row19" >19</th>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row19_col0" class="data row19 col0" >term</td>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row19_col1" class="data row19 col1" >The number of payments on the loan. Values are in months and can be either 36 or 60.</td>
            </tr>
            <tr>
                        <th id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002level0_row20" class="row_heading level0 row20" >20</th>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row20_col0" class="data row20 col0" >Year</td>
                        <td id="T_af79a8c0_0278_11ec_8aca_0242ac1c0002row20_col1" class="data row20 col1" >Year of Issue of the loan</td>
            </tr>
    </tbody></table>




```python
Data.dtypes
```




    annual_inc                  float64
    application_type             object
    avg_cur_bal                 float64
    chargeoff_within_12_mths    float64
    delinq_2yrs                 float64
    dti                         float64
    emp_length                    int64
    grade                        object
    inq_last_12m                float64
    installment                 float64
    loan_amnt                     int64
    num_actv_bc_tl              float64
    pub_rec_bankruptcies        float64
    home_ownership               object
    term                          int64
    mort_acc                    float64
    num_tl_90g_dpd_24m          float64
    purpose                      object
    year                          int64
    loan_default                  int64
    job                          object
    dtype: object



### Exercise 1: (20 mts)

For each of the following, perform the directed visualization and discuss your conclusions from it.

#### 1.1 

Create a bar chart showing the number of loans that did and did not default.

**Answer:**


```python
sns.countplot(Data.loan_default)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f88964c7c50>




    
![png](Predict_Defaulting_Loan_files/Predict_Defaulting_Loan_12_1.png)
    


Se observa que más del 20% de las deudas no se pagan.

#### 1.2 

Plot a histogram of the annual incomes.

**Answer:**


```python
Data.annual_inc.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f8895d1e850>




    
![png](Predict_Defaulting_Loan_files/Predict_Defaulting_Loan_16_1.png)
    


Los ingresos anuales de las mayoría de los prestamistas son inferiores a 150.000 dólares.

#### 1.3

Is the distribution of annual incomes different between applicants who defaulted vs. applicants who did not default on their loans?

**Answer:**


```python
np.log(Data.annual_inc[Data.loan_default == 0]).hist(bins=100, density=True)
np.log(Data.annual_inc[Data.loan_default == 1]).hist(bins=100, density=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f8895cc3050>




    
![png](Predict_Defaulting_Loan_files/Predict_Defaulting_Loan_20_1.png)
    


We can see that the distributions are not that different, indicating that income alone is not likely to explain a significant fraction of the difference in loan default status.

#### 1.4

Explore the association between annual income and the monthly installment.

**Answer:**


```python
plt.scatter(np.log(Data.annual_inc), np.log(Data.installment))
```




    <matplotlib.collections.PathCollection at 0x7f88959c2e10>




    
![png](Predict_Defaulting_Loan_files/Predict_Defaulting_Loan_24_1.png)
    


Se visualiza una pequeña relación lineal a escala logarítimica entre los ingresos y el pago mensual del préstamo.

__________

Here are a few more figures which look at the relationship between other numerical covariates and the probability of default, as well as annual income:

`emp_length`:


```python
fig, (ax1, ax2) = plt.subplots(figsize = (10,5), ncols=2, sharey= False)
sns.boxplot(x='emp_length', y = 'annual_inc', data = Data, showfliers=False, ax = ax1) #showfliers=False for nice display
ax1.set_title("Annual income vs Employment length")
Data[["emp_length",'loan_default']].groupby("emp_length").mean().plot.bar(rot=90,ax = ax2)
plt.title("Default probability vs. Employment length")
```




    Text(0.5, 1.0, 'Default probability vs. Employment length')




    
![png](Predict_Defaulting_Loan_files/Predict_Defaulting_Loan_29_1.png)
    


Con el diagrama de cajas y bigotes se observa que la experiencia laboral de los prestamistas no diferencia los ingresos obtenidos entre los mismos, es decir, mayor experiencia laboral no significa mayores ingresos. Además, no se observa una relación clara entre la probabilidad de no pagar la deuda y la experiencia laboral ya que las diferencias entre las probabilidades mínimas y máximas son de al menos el 5%.

`homeOwnership`:


```python
fig, (ax1, ax2) = plt.subplots(figsize = (10,5), ncols=2, sharey= False)
sns.boxplot(x="home_ownership",y="annual_inc", data = Data, showfliers=False, ax = ax1) #showfliers=False for nice display
ax1.set_title("Annual income vs Home ownership")
Data[["home_ownership",'loan_default']].groupby("home_ownership").mean().plot.bar(rot=90,ax = ax2)
plt.title("Default probability vs. home ownership");
```


    
![png](Predict_Defaulting_Loan_files/Predict_Defaulting_Loan_32_0.png)
    


De la primera gráfica inferimos que en promedio los prestamistas con hipotecas tienen mayores ingresos anuales que los propietarios de viviendas y aquellos que rentan. Además, de la segunda gráfica se puede concluir que las personas que pagan renta tienen mayor probabilidad de no pagar sus deudas en comparación a los otros prestamistas con propiedad o hipoteca.

Here are some figures that show the relationship between various categorical variables and the probability of default:

`purpose`:


```python
plt.figure(figsize= (10,5))
Data.emp_length.value_counts()
sns.countplot(x='purpose', order=Data['purpose'].value_counts().index, data = Data) 
plt.xticks(rotation=90)
plt.title("Distribution of Loan Purposes", fontsize=20);
```


    
![png](Predict_Defaulting_Loan_files/Predict_Defaulting_Loan_36_0.png)
    


Se observa que la mayor proporción de los créditos son destinados a compra y consolidación de cartera, tarjetas de crédito y mejoras de vivienda.


```python
plt.figure(figsize= (10,5))
purpose_default = Data[["loan_default", "purpose"]].groupby("purpose").mean()
purpose_default = purpose_default.sort_values(by="loan_default",axis=0, ascending=False)
sns.barplot(x=purpose_default.index[:30], 
            y=purpose_default["loan_default"][:30].values,
            orient="v")
plt.xticks(rotation=90)
plt.ylabel("Default Probability");
plt.title("Default Probability by Loan Purpose")

```




    Text(0.5, 1.0, 'Default Probability by Loan Purpose')




    
![png](Predict_Defaulting_Loan_files/Predict_Defaulting_Loan_38_1.png)
    


Se observa en la gráfica anterior que la probabilidad de no pago de deudas sin caracterizar sumadas contribuyen en mayor cantidad que las deudas no pagadas por otros conceptos, a estas le siguen los no pagos a deudas de pequeños negocios, gastos médicos y compra - consolidación de cartera.

`job`:


```python
plt.figure(figsize= (15,5))
sns.barplot(x=Data["job"].value_counts()[:30].index.values , 
            y=100 * Data.job.value_counts()[:30].values / len(Data),
            orient="v")
plt.xticks(rotation=90)
plt.ylabel("Percentage of Population")
plt.title("Distribution of Jobs", fontsize=20);

```


    
![png](Predict_Defaulting_Loan_files/Predict_Defaulting_Loan_41_0.png)
    


Comparando entre los diferentes cargos de los prestamistas aquellos que tienen oficios administrativos aquieren la mayor cantidad de deudas en este banco.


```python
plt.figure(figsize= (20,5))

df_job_default = Data[["loan_default", "job"]].groupby("job").mean()
df_job_default = df_job_default.sort_values(by="loan_default",axis=0, ascending=False)
sns.barplot(x=df_job_default.index[:50], 
            y=df_job_default["loan_default"][:50].values,
            orient="v")
plt.xticks(rotation=90)

plt.ylabel("Defaut Probability")
plt.title("Default Probability by Job Type", fontsize=20, verticalalignment='bottom');
```


    
![png](Predict_Defaulting_Loan_files/Predict_Defaulting_Loan_43_0.png)
    


La mayoría de las profesiones de los prestasmistas tienen probabilidad de impago entre el 25% y el 50%. Por otro lado, las profesiones con esta probabilidad mayor al 50% son educador, producción y científico con el 68%, 71% y 80% respectivamente.

### Adding a new variable

The yearly payment owed by the borrower, as a fraction of their annual income, is a standard metric used in evaluating whether a loan should be issued. Let's define a new variable **"install_income"** which codes the installment as a fraction of the annual income and study its association with the other features:


```python
Data['install_income'] = 12 * Data.installment / Data.annual_inc
H = plt.hist(Data['install_income'], bins=100, density=True)
plt.xlabel(r"Installment / Income")
```




    Text(0.5, 0, 'Installment / Income')




    
![png](Predict_Defaulting_Loan_files/Predict_Defaulting_Loan_46_1.png)
    


Se observa que los prestamistas destinan máximo el 20% de sus ingresos anuales en el pago mensual de la cuota del crédito.

In order to easily investigate this variable's association with the probability of default, define a new covariate named `install_income_disc` that is a discretized version of `install_income`:


```python
# let us discretize the "install_income" variable to study the probability of default 
# as a function of "install_income"
Data["install_income_disc"] = (Data.install_income*50).astype(int)/50.  #discretization
Data[["loan_default", "install_income_disc"]].groupby("install_income_disc").mean().plot.bar(rot=90)
Data = Data.drop(["install_income_disc"], axis=1)

# --> there is a clear positive association: as the fraction of the annual income devoted to the re-imbursement of 
# the loan increases, the probability of default sharply increases
```


    
![png](Predict_Defaulting_Loan_files/Predict_Defaulting_Loan_49_0.png)
    


La gráfica anterior muestra la relación de no pago y la proporción de la cuota respecto al salario del prestamista, a mayor proporción de pago mensual mayor probabilidad de no pago.

### Exercise 2: (10 mts)

Visualize the correlation matrix across all numerical features by using the `sns.heatmap()` command

**Answer:**


```python
#compute correlation matrix
labels = ['application_type', 'grade', 'purpose', 'job', 'home_ownership']# drop rows
# get correlations
df_corr = Data.corr()# irrelevant fields
df_corr

#mask the upper half for visualization purposes
np.ones_like(df_corr, dtype=np.bool)

fig, ax = plt.subplots(figsize=(17, 15))# mask
mask = np.triu(np.ones_like(df_corr, dtype=np.bool))# adjust mask and df
mask = mask[1:, :-1]
corr = df_corr.iloc[1:,:-1].copy()# plot heatmap
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='Blues',
           vmin=-1, vmax=1, cbar_kws={"shrink": .8})# yticks
plt.yticks(rotation=0)

# Draw the heatmap with the mask and correct aspect ratio
plt.show()
```


    
![png](Predict_Defaulting_Loan_files/Predict_Defaulting_Loan_52_0.png)
    


## Building a predictive model (20 mts)

Let's first start by building a standard logistic regression model. In general, it is important and extremely useful to first create baseline/simple models which can be compared to more complex models later.

### Exercise 3: (15 mts)

#### 3.1

Using the `LogisticRegression()` function from `scikit-learn`, write a function named `fit_logistic_regression(X,y)` that fits a logistic regression on the array of covariates `X` and associated response variable `y`.

**Answer:**


```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# import the metrics class
from sklearn.metrics import accuracy_score, recall_score, f1_score


def fit_logistic_regression(X, y, split):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42)
    
    # instantiate the model (using the default parameters)
    logreg = LogisticRegression()

    # fit the model with data
    logreg.fit(X_train, y_train)
    
    #
    y_pred = logreg.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    return np.array(y_test), y_pred
```

#### 3.2

Create a basic logistic regression model for predicting the loan default with only one feature: `install_income`.  Call this model `model1`. Use a 70/30 train-test split of the data.

**Answer:**


```python
X = Data[['install_income']] # Features
y = Data.loan_default

y_test, y_pred = fit_logistic_regression(X, y, split=0.3)
```

    Accuracy: 0.776
    Recall: 0.005934718100890208
    F1 Score: 0.011764705882352941


#### 3.3

Plot the ROC curve of `model1` and find the area under the curve.

**Answer:**


```python
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

def plot_roc(y, y_pred):
    # función para pintar la curva ROC y mostrar el AUC
    roc_novs = roc_curve(y, y_pred) # .cat.codes
    auc_novs = auc(roc_novs[0], roc_novs[1])
    print('AUC ROC')
    print(auc_novs)
    print('------------------------------------------------------')

    plt.figure()
    lw = 2

    plt.plot(roc_novs[0], roc_novs[1], color='darkgreen',
    lw=lw, label='Without verification_status (AUC = %0.2f)' % auc_novs)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='Random guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Classification of Defaulters')
    plt.legend(loc="lower right")
    plt.show()

plot_roc(pd.DataFrame(y_test).iloc[:,0].astype(int).tolist(), pd.DataFrame(y_pred).iloc[:,0].astype(int).tolist())

```

    AUC ROC
    0.5025374364365156
    ------------------------------------------------------



    
![png](Predict_Defaulting_Loan_files/Predict_Defaulting_Loan_62_1.png)
    


### Exercise 4: (5 mts)

#### 4.1

Consider `model1` from above. Would you want this to be your final model? Why or why not?

**Answer:**

Sí, ya que la exactitud del modelo es de 75% aproximadamente y por el principio de la parsimonia sería un modelo adecuado.

#### 4.2

Let's instead put all the variables available in the model, so that we are maximally leveraging our available info. Would you be in favor of this or not?


```python
X = Data.drop(columns=['loan_default','application_type', 'grade', 'purpose', 'job', 'home_ownership']) # Features
y = Data.loan_default

y_test, y_pred = fit_logistic_regression(X,y,split = 0.3)
```

    Accuracy: 0.7766666666666666


**Answer:**

Se concluye que añadir todas las variables al modelo no tiene mucho sentido, ya que la diferencia en la exactitud es de aproximadamente el 1% y agregar todas las variables podría sobreajustar el modelo.

## Cross-validation (30 mts)

**Cross-validation** is a set of techniques for assessing how well the results of a model will generalize to an out-of-sample dataset; i.e. in practice or production. It is chiefly used to flag overfitting.

Cross-validation works as follows: one splits the available data into $k$ sets, or **folds**. $k - 1$ of these folds will be used to train the model, while the held-out fold will be used as the test set on which the model is evaluated. For computational stability, this procedure is generally split many times, such that each fold has an opportunity to serve as the test set. For each repetition, a metric of prediction performance (e.g. AUC) is calculated on the test set. The average of these metrics, as well as their standard deviation, is then reported. An example is shown here for 5-fold cross-validation:

![](cv_fig.png)

Let's do this with code. The following code displays the 5 different folds used in a standard 5-fold cross-validation approach. To do so, use the `StratifiedKFold()` function from `scikit-learn`:


```python
skf = StratifiedKFold(n_splits=5)
for k, (train_index, test_index) in enumerate( skf.split(X, y) ):
    plt.plot(train_index, [k+1 for _ in train_index], ".")
plt.ylim(0,6)
plt.ylabel("FOLD")
plt.title("CROSS VALIDATION FOLDS")
```




    Text(0.5, 1.0, 'CROSS VALIDATION FOLDS')




    
![png](Predict_Defaulting_Loan_files/Predict_Defaulting_Loan_74_1.png)
    


The following code defines a function `compute_AUC(X, y, train_index, test_index)` that computes the AUC of a model trained on "train_index" and tested in "test_index".


```python
def compute_AUC(X, y, train_index, test_index):
    """
    feature/output: X, y
    dataset split: train_index, test_index
    """
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_test, y_test = X.iloc[test_index], y.iloc[test_index]

    clf = fit_logistic_regression(X_train, y_train, split=0.3)
    default_proba_test = clf.predict_proba(X_test)[:,1]  
    fpr, tpr, _ = roc_curve(y_test, default_proba_test)
    auc_score = auc(fpr, tpr)
    return auc_score, fpr, tpr
```

### Exercise 5: (5 mts)

With the help of the `compute_AUC` function defined above, write a function `cross_validation_AUC(X,y,nfold)` that carries out a 10-fold cross-validation and returns a list which contains the area under the curve for each fold of the cross-validation.

**Answer:**


```python
from scipy import mean
def cross_validation_AUC(X,y,nfold):
    AUC = []
    skf = StratifiedKFold(n_splits=nfold)
    for train_index, test_index in skf.split(X, y) :
        auc_score, fpr, tpr = compute_AUC(X, y, train_index, test_index)
        AUC.append(auc_score)
    auc_listas = AUC   
    return auc_listas
```


```python
X = Data[['install_income']] # Features
y = Data.loan_default
lista_1 = cross_validation_AUC(X, y, nfold = 10)
```


```python
X = Data.drop(columns=['loan_default','application_type', 'grade', 'purpose', 'job', 'home_ownership']) # Features
y = Data.loan_default
lista_2 =cross_validation_AUC(X, y, nfold = 10)
```

__________

We will now estimate and compare, through cross-validation analysis, the performance of all the "simple models" that only use one numerical feature as input. As discussed in the EDA section, we will use the logarithmic transform for the `anual_income`, `loan_amount`, and `avg_cur_bal` variables:


```python
# let us extract only the numerical (i.e non-categorical) features
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
Data_numerics = Data.select_dtypes(include=numerics)
Data_numerics = Data_numerics.drop(["installment", "year"], axis=1)

# Using a log scale when appropriate
Data_numerics["annual_inc"] = np.log10(Data_numerics["annual_inc"])
Data_numerics["loan_amnt"] = np.log10(Data_numerics["loan_amnt"])
Data_numerics["avg_cur_bal"] = np.log10(1.+Data_numerics["avg_cur_bal"])
```

Let's compute cross-validation estimates of the AUC for each single-feature model:


```python
model_perf = pd.DataFrame({}) #this data-frame will contain the AUC estimates
for key in Data_numerics.keys():
    if key == "loan_default": continue
    X_full, y_full = Data_numerics[[key]], Data_numerics.loan_default
    auc_list = cross_validation_AUC(X_full, y_full, nfold=10)
    model_perf["SIMPLE:" + key] = auc_list
```


```python
model_perf
```




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
      <th>SIMPLE:annual_inc</th>
      <th>SIMPLE:avg_cur_bal</th>
      <th>SIMPLE:chargeoff_within_12_mths</th>
      <th>SIMPLE:delinq_2yrs</th>
      <th>SIMPLE:dti</th>
      <th>SIMPLE:emp_length</th>
      <th>SIMPLE:inq_last_12m</th>
      <th>SIMPLE:loan_amnt</th>
      <th>SIMPLE:num_actv_bc_tl</th>
      <th>SIMPLE:pub_rec_bankruptcies</th>
      <th>SIMPLE:term</th>
      <th>SIMPLE:mort_acc</th>
      <th>SIMPLE:num_tl_90g_dpd_24m</th>
      <th>SIMPLE:install_income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.554410</td>
      <td>0.542259</td>
      <td>0.494783</td>
      <td>0.501942</td>
      <td>0.573698</td>
      <td>0.537041</td>
      <td>0.590830</td>
      <td>0.567871</td>
      <td>0.509689</td>
      <td>0.467544</td>
      <td>0.560926</td>
      <td>0.548797</td>
      <td>0.512456</td>
      <td>0.621604</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.573744</td>
      <td>0.568978</td>
      <td>0.499142</td>
      <td>0.478848</td>
      <td>0.578566</td>
      <td>0.530627</td>
      <td>0.535325</td>
      <td>0.607566</td>
      <td>0.589080</td>
      <td>0.488379</td>
      <td>0.580407</td>
      <td>0.599481</td>
      <td>0.484404</td>
      <td>0.661683</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.564472</td>
      <td>0.575065</td>
      <td>0.501750</td>
      <td>0.508187</td>
      <td>0.580305</td>
      <td>0.546990</td>
      <td>0.499435</td>
      <td>0.499718</td>
      <td>0.542823</td>
      <td>0.515460</td>
      <td>0.546527</td>
      <td>0.579944</td>
      <td>0.517527</td>
      <td>0.564438</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.577538</td>
      <td>0.526606</td>
      <td>0.498701</td>
      <td>0.507149</td>
      <td>0.598125</td>
      <td>0.518758</td>
      <td>0.514850</td>
      <td>0.554274</td>
      <td>0.506200</td>
      <td>0.497470</td>
      <td>0.569565</td>
      <td>0.536036</td>
      <td>0.495099</td>
      <td>0.639097</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.584190</td>
      <td>0.584822</td>
      <td>0.503049</td>
      <td>0.486358</td>
      <td>0.570695</td>
      <td>0.496454</td>
      <td>0.585251</td>
      <td>0.567781</td>
      <td>0.525409</td>
      <td>0.487081</td>
      <td>0.563523</td>
      <td>0.578701</td>
      <td>0.487250</td>
      <td>0.664009</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.568786</td>
      <td>0.532038</td>
      <td>0.500452</td>
      <td>0.458893</td>
      <td>0.568504</td>
      <td>0.555855</td>
      <td>0.516194</td>
      <td>0.529091</td>
      <td>0.532829</td>
      <td>0.505759</td>
      <td>0.545737</td>
      <td>0.548063</td>
      <td>0.494241</td>
      <td>0.582101</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.523343</td>
      <td>0.544370</td>
      <td>0.495257</td>
      <td>0.503185</td>
      <td>0.563467</td>
      <td>0.490785</td>
      <td>0.593586</td>
      <td>0.563840</td>
      <td>0.512851</td>
      <td>0.507679</td>
      <td>0.612196</td>
      <td>0.558521</td>
      <td>0.513676</td>
      <td>0.586155</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.564710</td>
      <td>0.573005</td>
      <td>0.494819</td>
      <td>0.475559</td>
      <td>0.536815</td>
      <td>0.492739</td>
      <td>0.531963</td>
      <td>0.553586</td>
      <td>0.485320</td>
      <td>0.502432</td>
      <td>0.604400</td>
      <td>0.538917</td>
      <td>0.474889</td>
      <td>0.596696</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.549257</td>
      <td>0.518555</td>
      <td>0.496114</td>
      <td>0.549484</td>
      <td>0.596809</td>
      <td>0.494512</td>
      <td>0.559108</td>
      <td>0.584504</td>
      <td>0.521453</td>
      <td>0.516635</td>
      <td>0.583174</td>
      <td>0.545144</td>
      <td>0.517328</td>
      <td>0.637806</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.525248</td>
      <td>0.560131</td>
      <td>0.504863</td>
      <td>0.529747</td>
      <td>0.572391</td>
      <td>0.480649</td>
      <td>0.539156</td>
      <td>0.574720</td>
      <td>0.530656</td>
      <td>0.486149</td>
      <td>0.548087</td>
      <td>0.577470</td>
      <td>0.518873</td>
      <td>0.602070</td>
    </tr>
  </tbody>
</table>
</div>



### Exercise 6: (5 mts)

Construct a boxplot which shows the distribution of cross-validation scores of each variable (remember, each variable has 10 total scores). Which feature has the highest/lowest predictive power?

**Answer:**


```python
sns.boxplot(data = model_perf,  orient="h" ,showfliers=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f8895126c90>




    
![png](Predict_Defaulting_Loan_files/Predict_Defaulting_Loan_90_1.png)
    


La mejor variable predictora sería la proporción del ingreso neto anual que es destinado al pago de la cuota del crédito (install_income) y la peor variable para predecir el impago es el número de veces que una persona se atraza en mas de 30 días para el pago de sus obligaciones de los últimos 2 años (delinq_2yrs).

### Exercise 7: (5 mts)

Consider the model that consists of using *all* the numerical features (and none of the categorical features). Carry out a 10-fold cross-validation analysis to determine whether this model has better predictive performance than the best single-feature model. Use the boxplot method again as we did in Exercise 7.

**Answer:**


```python
X = Data.drop(columns=['loan_default','application_type', 'grade', 'purpose', 'job', 'home_ownership']) # Features
y = Data.loan_default
lista_2 = cross_validation_AUC(X, y, nfold = 10)
print(lista_2)
sns.boxplot(data = lista_2,  orient="h" ,showfliers=False)
```

    [0.6716431394692264, 0.7152343308865049, 0.6423941276115188, 0.6683229813664596, 0.7050028232636929, 0.5804404291360813, 0.6765894974590627, 0.6467593855104081, 0.6659167348422871, 0.674484137805654]





    <matplotlib.axes._subplots.AxesSubplot at 0x7f88954a4650>




    
![png](Predict_Defaulting_Loan_files/Predict_Defaulting_Loan_94_2.png)
    


De acuerdo al gráfico anterior, podemos concluir que hay una leve mejora en la efectividad del modelo, solo tomando las variables numéricas.

## Incorporating categorical variables (25 mts)

The grade of a loan (i.e. the LC-assigned loan grade feature) has not been used so far. The following is the distribution of the categorical grade feature:


```python
Data.emp_length.value_counts()
sns.countplot(x='grade', data = Data) 
plt.xticks(rotation=90)
```




    (array([0, 1, 2, 3, 4, 5, 6]), <a list of 7 Text major ticklabel objects>)




    
![png](Predict_Defaulting_Loan_files/Predict_Defaulting_Loan_98_1.png)
    


### Exercise 8: (5 mts)

#### 8.1

Use `pandas.get_dummies()` to transform this into its one-hot encoded version.

**Answer:**


```python
cols = ['application_type', 'grade', 'purpose', 'job', 'home_ownership'] 
Data_dum = pd.get_dummies(data = Data,columns=cols)
Data_dum
```




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
      <th>annual_inc</th>
      <th>avg_cur_bal</th>
      <th>chargeoff_within_12_mths</th>
      <th>delinq_2yrs</th>
      <th>dti</th>
      <th>emp_length</th>
      <th>inq_last_12m</th>
      <th>installment</th>
      <th>loan_amnt</th>
      <th>num_actv_bc_tl</th>
      <th>pub_rec_bankruptcies</th>
      <th>term</th>
      <th>mort_acc</th>
      <th>num_tl_90g_dpd_24m</th>
      <th>year</th>
      <th>loan_default</th>
      <th>install_income</th>
      <th>application_type_Individual</th>
      <th>application_type_Joint App</th>
      <th>grade_A</th>
      <th>grade_B</th>
      <th>grade_C</th>
      <th>grade_D</th>
      <th>grade_E</th>
      <th>grade_F</th>
      <th>grade_G</th>
      <th>purpose_car</th>
      <th>purpose_credit_card</th>
      <th>purpose_debt_consolidation</th>
      <th>purpose_home_improvement</th>
      <th>purpose_house</th>
      <th>purpose_major_purchase</th>
      <th>purpose_medical</th>
      <th>purpose_moving</th>
      <th>purpose_other</th>
      <th>purpose_renewable_energy</th>
      <th>purpose_small_business</th>
      <th>purpose_vacation</th>
      <th>job_accountant</th>
      <th>job_accounting</th>
      <th>...</th>
      <th>job_planner</th>
      <th>job_practitioner</th>
      <th>job_president</th>
      <th>job_principal</th>
      <th>job_processor</th>
      <th>job_production</th>
      <th>job_professor</th>
      <th>job_programmer</th>
      <th>job_realtor</th>
      <th>job_receptionist</th>
      <th>job_recruiter</th>
      <th>job_representative</th>
      <th>job_resources</th>
      <th>job_sales</th>
      <th>job_scientist</th>
      <th>job_secretary</th>
      <th>job_security</th>
      <th>job_sergeant</th>
      <th>job_server</th>
      <th>job_service</th>
      <th>job_services</th>
      <th>job_specialist</th>
      <th>job_superintendent</th>
      <th>job_supervisor</th>
      <th>job_support</th>
      <th>job_teacher</th>
      <th>job_tech</th>
      <th>job_technician</th>
      <th>job_technologist</th>
      <th>job_teller</th>
      <th>job_therapist</th>
      <th>job_trainer</th>
      <th>job_underwriter</th>
      <th>job_vp</th>
      <th>job_warehouse</th>
      <th>job_welder</th>
      <th>job_worker</th>
      <th>home_ownership_MORTGAGE</th>
      <th>home_ownership_OWN</th>
      <th>home_ownership_RENT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2790</th>
      <td>38000.0</td>
      <td>2375.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12.19</td>
      <td>1</td>
      <td>1.0</td>
      <td>601.36</td>
      <td>17100</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>36</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2017</td>
      <td>1</td>
      <td>0.189903</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1822</th>
      <td>55000.0</td>
      <td>20371.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>21.19</td>
      <td>1</td>
      <td>1.0</td>
      <td>586.45</td>
      <td>17500</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>36</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2017</td>
      <td>0</td>
      <td>0.127953</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2857</th>
      <td>45000.0</td>
      <td>2437.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>17.09</td>
      <td>10</td>
      <td>0.0</td>
      <td>308.64</td>
      <td>10000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>36</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2016</td>
      <td>0</td>
      <td>0.082304</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4776</th>
      <td>125000.0</td>
      <td>10340.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>14.10</td>
      <td>1</td>
      <td>4.0</td>
      <td>874.93</td>
      <td>35000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>60</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2015</td>
      <td>0</td>
      <td>0.083993</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4482</th>
      <td>70000.0</td>
      <td>2089.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9.12</td>
      <td>3</td>
      <td>0.0</td>
      <td>848.27</td>
      <td>25000</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>36</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2017</td>
      <td>0</td>
      <td>0.145418</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3528</th>
      <td>260000.0</td>
      <td>38359.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.54</td>
      <td>10</td>
      <td>4.0</td>
      <td>1153.67</td>
      <td>35000</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>36</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2016</td>
      <td>0</td>
      <td>0.053246</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3722</th>
      <td>90000.0</td>
      <td>31537.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>18.27</td>
      <td>6</td>
      <td>1.0</td>
      <td>743.90</td>
      <td>22400</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>36</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2017</td>
      <td>0</td>
      <td>0.099187</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4405</th>
      <td>68000.0</td>
      <td>1849.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>17.00</td>
      <td>9</td>
      <td>3.0</td>
      <td>712.96</td>
      <td>20000</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>36</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2016</td>
      <td>0</td>
      <td>0.125816</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3968</th>
      <td>68500.0</td>
      <td>17577.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>27.45</td>
      <td>3</td>
      <td>6.0</td>
      <td>978.27</td>
      <td>32875</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>60</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2016</td>
      <td>1</td>
      <td>0.171376</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4577</th>
      <td>121000.0</td>
      <td>62367.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>11.14</td>
      <td>10</td>
      <td>1.0</td>
      <td>777.55</td>
      <td>25000</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>36</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>2017</td>
      <td>0</td>
      <td>0.077112</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 160 columns</p>
</div>



#### 8.2

Add this feature to the all-numerical model from earlier and investigate whether this leads to a significant increase in predictive accuracy.


```python
X = Data_dum.drop(columns=['loan_default']) # Features
y = Data_dum.loan_default
y_test, y_pred = fit_logistic_regression(X, y, split=0.3)
```

    Accuracy: 0.7766666666666666
    Recall: 0.06824925816023739
    F1 Score: 0.12073490813648292


No se presenta una mejora significativa en el Accuracy agregando  las variables dummies al modelo.

**Answer:**

### Exercise 9: (15 mts)

Investigate whether the categorical variable `job` brings any predictive value when added to the current best model. Again, you may want to use a one-hot encoding scheme.

**Answer:**


```python
cols = ['loan_default','application_type', 'grade', 'purpose', 'home_ownership']
Data_dum_2 = pd.get_dummies(data = Data,columns=['job']) 
X = Data_dum_2.drop(columns=cols)
y =  Data_dum_2.loan_default
y_test, y_pred = fit_logistic_regression(X,y,split = 0.3)
```

    Accuracy: 0.7766666666666666
    Recall: 0.06824925816023739
    F1 Score: 0.12073490813648292


No se presenta ninguna mejora estadisticamente significativa en modelo.

## Conclusions (5 mts)

In this case, we first explored the loan dataset and found the single-variable associations between the available features and the default rate. We also discovered which features required transformations (e.g. log transform).

Once we started building models, we started with very simple logistic regressions approaches – these baseline models were useful for quickly evaluating the predictive power of each individual variable. Next, we employed cross-validation approaches for building more complex models, often exploiting the interactions between the different features. Since the loan dataset contains a large number of covariates, using cross-validation was revealed to be crucial for avoiding overfitting, choosing the correct number of features and ultimately choosing an appropriate model that balanced complexity with accuracy.

## Takeaways (5 mts)

Cross-validation is a robust and flexible technique for evaluating the predictive performance of statistical models. It is especially useful in big data settings where the number of features is large compared to the number of observations. When used appropriately, cross-validation is a powerful method for choosing a model with the correct complexity and best predictive performance.
