```python
### import necessary stuffs
```


```python
import pandas as pd
```


```python
df=pd.read_csv(r'F:\NLP\Projects\Fake_news/train.csv')
```


```python
df.head()
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
      <th>id</th>
      <th>title</th>
      <th>author</th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>House Dem Aide: We Didn’t Even See Comey’s Let...</td>
      <td>Darrell Lucus</td>
      <td>House Dem Aide: We Didn’t Even See Comey’s Let...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>FLYNN: Hillary Clinton, Big Woman on Campus - ...</td>
      <td>Daniel J. Flynn</td>
      <td>Ever get the feeling your life circles the rou...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Why the Truth Might Get You Fired</td>
      <td>Consortiumnews.com</td>
      <td>Why the Truth Might Get You Fired October 29, ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>15 Civilians Killed In Single US Airstrike Hav...</td>
      <td>Jessica Purkiss</td>
      <td>Videos 15 Civilians Killed In Single US Airstr...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Iranian woman jailed for fictional unpublished...</td>
      <td>Howard Portnoy</td>
      <td>Print \nAn Iranian woman has been sentenced to...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (20800, 5)




```python
df.isnull().sum()
```




    id           0
    title      558
    author    1957
    text        39
    label        0
    dtype: int64




```python
## drop your missing values
df.dropna(inplace=True)
```


```python
df.shape
```




    (18285, 5)




```python

```


```python
## checking distribution of data
import seaborn as sns
def create_distribution(feature):
    return sns.countplot(df[feature])
```


```python
df.dtypes
```




    id         int64
    title     object
    author    object
    text      object
    label      int64
    dtype: object




```python
df['label']=df['label'].astype(str)
```


```python
df.dtypes
```




    id         int64
    title     object
    author    object
    text      object
    label     object
    dtype: object




```python
create_distribution('label')
```




    <matplotlib.axes._subplots.AxesSubplot at 0xf362a0e908>




    
![png](NLP_fake_news_deploy_files/NLP_fake_news_deploy_13_1.png)
    



```python

```


```python
df.head(20)
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
      <th>id</th>
      <th>title</th>
      <th>author</th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>House Dem Aide: We Didn’t Even See Comey’s Let...</td>
      <td>Darrell Lucus</td>
      <td>House Dem Aide: We Didn’t Even See Comey’s Let...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>FLYNN: Hillary Clinton, Big Woman on Campus - ...</td>
      <td>Daniel J. Flynn</td>
      <td>Ever get the feeling your life circles the rou...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Why the Truth Might Get You Fired</td>
      <td>Consortiumnews.com</td>
      <td>Why the Truth Might Get You Fired October 29, ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>15 Civilians Killed In Single US Airstrike Hav...</td>
      <td>Jessica Purkiss</td>
      <td>Videos 15 Civilians Killed In Single US Airstr...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Iranian woman jailed for fictional unpublished...</td>
      <td>Howard Portnoy</td>
      <td>Print \nAn Iranian woman has been sentenced to...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>Jackie Mason: Hollywood Would Love Trump if He...</td>
      <td>Daniel Nussbaum</td>
      <td>In these trying times, Jackie Mason is the Voi...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>Benoît Hamon Wins French Socialist Party’s Pre...</td>
      <td>Alissa J. Rubin</td>
      <td>PARIS  —   France chose an idealistic, traditi...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>A Back-Channel Plan for Ukraine and Russia, Co...</td>
      <td>Megan Twohey and Scott Shane</td>
      <td>A week before Michael T. Flynn resigned as nat...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>Obama’s Organizing for Action Partners with So...</td>
      <td>Aaron Klein</td>
      <td>Organizing for Action, the activist group that...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>BBC Comedy Sketch "Real Housewives of ISIS" Ca...</td>
      <td>Chris Tomlinson</td>
      <td>The BBC produced spoof on the “Real Housewives...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>Russian Researchers Discover Secret Nazi Milit...</td>
      <td>Amando Flavio</td>
      <td>The mystery surrounding The Third Reich and Na...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>US Officials See No Link Between Trump and Russia</td>
      <td>Jason Ditz</td>
      <td>Clinton Campaign Demands FBI Affirm Trump's Ru...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>Re: Yes, There Are Paid Government Trolls On S...</td>
      <td>AnotherAnnie</td>
      <td>Yes, There Are Paid Government Trolls On Socia...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>In Major League Soccer, Argentines Find a Home...</td>
      <td>Jack Williams</td>
      <td>Guillermo Barros Schelotto was not the first A...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>Wells Fargo Chief Abruptly Steps Down - The Ne...</td>
      <td>Michael Corkery and Stacy Cowley</td>
      <td>The scandal engulfing Wells Fargo toppled its ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>Anonymous Donor Pays $2.5 Million To Release E...</td>
      <td>Starkman</td>
      <td>A Caddo Nation tribal leader has just been fre...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>FBI Closes In On Hillary!</td>
      <td>The Doc</td>
      <td>FBI Closes In On Hillary! Posted on Home » Hea...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>Chuck Todd: ’BuzzFeed Did Donald Trump a Polit...</td>
      <td>Jeff Poor</td>
      <td>Wednesday after   Donald Trump’s press confere...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>Monica Lewinsky, Clinton Sex Scandal Set for ’...</td>
      <td>Jerome Hudson</td>
      <td>Screenwriter Ryan Murphy, who has produced the...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>Rob Reiner: Trump Is ’Mentally Unstable’ - Bre...</td>
      <td>Pam Key</td>
      <td>Sunday on MSNBC’s “AM Joy,” actor and director...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
messages=df.copy()
```


```python
#why to rset_index,bcz in above we can check,when we drop our rows get deleted as 6 and 8th so to make it in a order , we have to use reset_index

messages.reset_index(inplace=True)
```


```python
messages.head(10)
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
      <th>index</th>
      <th>id</th>
      <th>title</th>
      <th>author</th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>House Dem Aide: We Didn’t Even See Comey’s Let...</td>
      <td>Darrell Lucus</td>
      <td>House Dem Aide: We Didn’t Even See Comey’s Let...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>FLYNN: Hillary Clinton, Big Woman on Campus - ...</td>
      <td>Daniel J. Flynn</td>
      <td>Ever get the feeling your life circles the rou...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
      <td>Why the Truth Might Get You Fired</td>
      <td>Consortiumnews.com</td>
      <td>Why the Truth Might Get You Fired October 29, ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
      <td>15 Civilians Killed In Single US Airstrike Hav...</td>
      <td>Jessica Purkiss</td>
      <td>Videos 15 Civilians Killed In Single US Airstr...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
      <td>Iranian woman jailed for fictional unpublished...</td>
      <td>Howard Portnoy</td>
      <td>Print \nAn Iranian woman has been sentenced to...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>5</td>
      <td>Jackie Mason: Hollywood Would Love Trump if He...</td>
      <td>Daniel Nussbaum</td>
      <td>In these trying times, Jackie Mason is the Voi...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>7</td>
      <td>Benoît Hamon Wins French Socialist Party’s Pre...</td>
      <td>Alissa J. Rubin</td>
      <td>PARIS  —   France chose an idealistic, traditi...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9</td>
      <td>9</td>
      <td>A Back-Channel Plan for Ukraine and Russia, Co...</td>
      <td>Megan Twohey and Scott Shane</td>
      <td>A week before Michael T. Flynn resigned as nat...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10</td>
      <td>10</td>
      <td>Obama’s Organizing for Action Partners with So...</td>
      <td>Aaron Klein</td>
      <td>Organizing for Action, the activist group that...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>11</td>
      <td>11</td>
      <td>BBC Comedy Sketch "Real Housewives of ISIS" Ca...</td>
      <td>Chris Tomlinson</td>
      <td>The BBC produced spoof on the “Real Housewives...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
messages.drop(['index','id'],axis=1,inplace=True)
```


```python
messages.head()
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
      <th>title</th>
      <th>author</th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>House Dem Aide: We Didn’t Even See Comey’s Let...</td>
      <td>Darrell Lucus</td>
      <td>House Dem Aide: We Didn’t Even See Comey’s Let...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FLYNN: Hillary Clinton, Big Woman on Campus - ...</td>
      <td>Daniel J. Flynn</td>
      <td>Ever get the feeling your life circles the rou...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Why the Truth Might Get You Fired</td>
      <td>Consortiumnews.com</td>
      <td>Why the Truth Might Get You Fired October 29, ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15 Civilians Killed In Single US Airstrike Hav...</td>
      <td>Jessica Purkiss</td>
      <td>Videos 15 Civilians Killed In Single US Airstr...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Iranian woman jailed for fictional unpublished...</td>
      <td>Howard Portnoy</td>
      <td>Print \nAn Iranian woman has been sentenced to...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#note we will consider only title for pre-processing
```


```python
data=messages['title'][0]
data
```




    'House Dem Aide: We Didn’t Even See Comey’s Letter Until Jason Chaffetz Tweeted It'




```python

```


```python
import re
```


```python
re.sub('[^a-zA-Z]',' ', data)
```




    'House Dem Aide  We Didn t Even See Comey s Letter Until Jason Chaffetz Tweeted It'




```python
data=data.lower()
data
```




    'house dem aide: we didn’t even see comey’s letter until jason chaffetz tweeted it'




```python
list=data.split()
list
```




    ['house',
     'dem',
     'aide:',
     'we',
     'didn’t',
     'even',
     'see',
     'comey’s',
     'letter',
     'until',
     'jason',
     'chaffetz',
     'tweeted',
     'it']




```python
!pip install nltk

```


```python
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
```


```python
ps=PorterStemmer()
```


```python
review=[ps.stem(word) for word in list if word not in set(stopwords.words('english'))]
review
```




    ['hous',
     'dem',
     'aide:',
     'didn’t',
     'even',
     'see',
     'comey’',
     'letter',
     'jason',
     'chaffetz',
     'tweet']




```python
review=[]
for word in list:
    if word not in set(stopwords.words('english')):
        review.append(ps.stem(word))
review
```




    ['hous',
     'dem',
     'aide:',
     'didn’t',
     'even',
     'see',
     'comey’',
     'letter',
     'jason',
     'chaffetz',
     'tweet']




```python
' '.join(review)
```




    'hous dem aide: didn’t even see comey’ letter jason chaffetz tweet'




```python
### lets do same task for each & every row
```


```python
corpus=[]
sentences=[]
for i in range(0,len(messages)):
    review=re.sub('[^a-zA-Z]',' ', messages['title'][i])
    review=review.lower()
    list=review.split()
    review=[ps.stem(word) for word in list if word not in set(stopwords.words('english'))]
    sentences=' '.join(review)
    corpus.append(sentences)
```


```python

```


```python
corpus[0]
```




    'hous dem aid even see comey letter jason chaffetz tweet'




```python
corpus
```




    ['hous dem aid even see comey letter jason chaffetz tweet',
     'flynn hillari clinton big woman campu breitbart',
     'truth might get fire',
     'civilian kill singl us airstrik identifi',
     'iranian woman jail fiction unpublish stori woman stone death adulteri',
     'jacki mason hollywood would love trump bomb north korea lack tran bathroom exclus video breitbart',
     'beno hamon win french socialist parti presidenti nomin new york time',
     'back channel plan ukrain russia courtesi trump associ new york time',
     'obama organ action partner soro link indivis disrupt trump agenda',
     'bbc comedi sketch real housew isi caus outrag',
     'russian research discov secret nazi militari base treasur hunter arctic photo',
     'us offici see link trump russia',
     'ye paid govern troll social media blog forum websit',
     'major leagu soccer argentin find home success new york time',
     'well fargo chief abruptli step new york time',
     'anonym donor pay million releas everyon arrest dakota access pipelin',
     'fbi close hillari',
     'chuck todd buzzfe donald trump polit favor breitbart',
     'monica lewinski clinton sex scandal set american crime stori',
     'rob reiner trump mental unstabl breitbart',
     'abort pill order rise latin american nation zika alert new york time',
     'nuke un histor treati ban nuclear weapon',
     'exclus islam state support vow shake west follow manchest terrorist massacr breitbart',
     'humili hillari tri hide camera caught min ralli',
     'andrea tantaro fox news claim retali sex harass complaint new york time',
     'hillari clinton becam hawk new york time',
     'chuck todd buzzfe eic publish fake news breitbart',
     'bori johnson brexit leader fumbl new york time',
     'texa oil field rebound price lull job left behind new york time',
     'bayer deal monsanto follow agribusi trend rais worri farmer new york time',
     'russia move ban jehovah wit extremist new york time',
     'still danger zone januari th',
     'open thread u elect',
     'democrat gutierrez blame chicago gun violenc nra breitbart',
     'avoid peanut avoid allergi bad strategi new york time',
     'mri show detail imag week unborn babi breitbart',
     'best kind milk dairi',
     'ryan locht drop speedo usa retail new york time',
     'conserv urg session clean obama civil right divis breitbart',
     'intern inquiri seal fate roger ail fox new york time',
     'press tv debat duff lebanon hezbollah aoun presid video',
     'samsung combust galaxi note unveil new smartphon new york time',
     'poland vow referendum migrant quota amidst eu pressur public voic heard breitbart',
     'spark inner revolut',
     'studi half car crash involv driver distract cell phone breitbart',
     'trump elect spark individu collect heal',
     'ep fade black jimmi church w laura eisenhow restor balanc video',
     'cognit true islam book review',
     'donald trump win elect biggest miracl us polit histori',
     'mind eat way fight bing new york time',
     'major potenti impact corpor tax overhaul new york time',
     'wonder glp like day elect',
     'maker world smallest machin award nobel prize chemistri new york time',
     'massiv anti trump protest union squar nyc live stream',
     'review lion bring tear lost boy wipe dri googl new york time',
     'u gener islam state chemic attack impact u forc',
     'juri find oregon standoff defend guilti feder conspiraci gun charg',
     'clinton campaign stun fbi reportedli reopen probe hillari clinton email',
     'penc speak anti abort ralli new york time',
     'berni sander say media trump gutless polit coward',
     'make briquett daili wast',
     'treason nyt vow reded report',
     'dress like woman mean new york time',
     'ella brennan still feed lead new orlean new york time',
     'press asia agenda obama tread lightli human right new york time',
     'democrat percent chanc retak senat new york time',
     'judg spank transgend obsess obama lie redflag news',
     'u diplomat urg strike assad syria new york time',
     'franken call independ investig trump putin crush breitbart',
     'louisiana simon bile u presidenti race tuesday even brief new york time',
     'turkey threaten open migrant land passag europ row dutch',
     'huma weiner dog hillari',
     'colin kaepernick start black panther inspir youth camp wow',
     'trump immigr polici explain new york time',
     'mari tyler moor mourn dick van dyke star new york time',
     'poison',
     'trump fan ralli across nation support presid new york time',
     'fox biz report help bash clinton ralli cover trump pack event day',
     'fiction podcast worth listen new york time',
     'mike birbiglia tip make small hollywood anywher new york time',
     'invest strategist forecast collaps timelin last gasp econom cycl come',
     'venezuela muzzl legislatur move closer one man rule new york time',
     'whether john mccain mitt romney donald trump democrat alway run war women tactic destroy republican candid',
     'breitbart news daili trump boom breitbart',
     'white hous confirm gitmo transfer obama leav offic',
     'poll voter heard democrat elect candid breitbart',
     'migrant confront judgment day old deport order new york time',
     'n u yale su retir plan fee new york time',
     'technocraci real reason un want control internet',
     'american driver regain appetit ga guzzler new york time',
     'hillari clinton build million war chest doubl donald trump new york time',
     'trump catch sick report snuck interview priceless respons',
     'senat contact russian govern week',
     'imag perfectli illustr struggl dakota access pipelin',
     'washington state take refuge muslim rest countri take refuge muslim breitbart',
     'ncaa big keep watch eye texa bathroom bill breitbart',
     'massiv espn financi subscrib loss drag disney first quarter sale breitbart',
     'megyn kelli contract set expir next year prime big show new york time',
     'teacher suspend allow student hit trump pinata cinco de mayo',
     'break trump express concern anthoni weiner illeg access classifi info month ago truthfe',
     'snap share leap debut investor doubt valu vanish new york time',
     'clinton campaign chair dinner top doj offici one day hillari benghazi hear',
     'tv seri first femal mlb pitcher can one low rate season breitbart',
     'seek best fit women final four return friday sunday new york time',
     'propos canadian nation bird ruffl feather new york time',
     'review beyonc make lemonad marit strife new york time',
     'trump ask sharp increas militari spend offici say new york time',
     'waterg smoke gun email discuss clean obama hillari email',
     'chapo trap hous new left wing podcast flagrant rip right stuff',
     'taiwan respond china send carrier taiwan strait new york time',
     'mother octob surpris hous card come tumbl',
     'explos assang pilger interview us elect expect riot hillari win',
     'telescop ate astronomi track surpass hubbl new york time',
     'close afghan pakistani border becom humanitarian crisi new york time',
     'tv anchor arriv white hous lunch donald trump breitbart',
     'pelosi republican tell trump bring dishonor presid breitbart',
     'beauti prehistor world earth wasteland',
     'ignor trump news week learn new york time',
     'donald trump unveil plan famili bid women vote new york time',
     'montana democrat vote bill ban sharia law call repugn breitbart',
     'monsanto tribun go happen',
     'offici simon bile world best gymnast new york time',
     'liter hurt brain read econom idioci emit trumpkin libertarian',
     'u n secretari gener complain mass reject global favor nation',
     'trump bollywood ad meant sway indian american voter hilari fail video',
     'fbi find previous unseen hillari clinton email weiner laptop',
     'year american journalist kill conspiraci theori syria proven fact',
     'report illeg alien forego food stamp stay trump radar',
     'make netherland great hahaha spread worldwid',
     'four kill injur jerusalem truck ram terror attack',
     'leader salut comrad newt brutal megyn sic kelli beatdown play game',
     'student black colleg got beaten mace protest kkk david duke',
     'despit strict gun control one child youth shot everi day ontario',
     'rise internet fan bulli new york time',
     'newli vibrant washington fear trump drain cultur new york time',
     'fed hold interest rate steadi plan slower increas new york time',
     'battl unesco',
     'latest test white hous pull easter egg roll new york time',
     'burlesqu dancer fire investig secret servic trump assassin tweet breitbart',
     'clinton haiti',
     'cuomo christi parallel path top troubl got new york time',
     'top place world allow visit',
     'new studi link fluorid consumpt hypothyroid weight gain wors',
     'jame matti secretari offens',
     'black church burn spray paint vote trump',
     'sear agre sell craftsman stanley black amp decker rais cash new york time',
     'takata chief execut resign financi pressur mount new york time',
     'goodby good black sabbath new york time',
     'teen geisha doll gang bust arm robberi breitbart',
     'mohamad khwei anoth virginia man palestinian american muslim charg terror',
     'price obamacar replac nobodi wors financi breitbart',
     'va fail properli examin thousand veteran',
     'trump famili alreadi sworn secreci fake moon land soon',
     'sport writer nfl great jim brown decad civil right work eras say nice thing donald trump breitbart',
     'watch tv excus republican skip donald trump convent new york time',
     'open letter trump voter told like',
     'comment power corpor lobbi quietli back hillari nobodi talk runsinquicksand',
     'hijack end peac libyan airlin land malta new york time',
     'like girl girl geniu new york time',
     'scientist say canadian bacteria fossil may earth oldest new york time',
     'pro govern forc advanc syria amid talk u russia cooper new york time',
     'cancer agenc fire withhold carcinogen glyphos document',
     'work walk minut work new york time',
     'steve harvey talk hous presid elect trump new york time',
     'coalit u troop fight mosul offens come fire',
     'uk citizen war hero get cheap pre fab hous muslim colon get taxpay fund luxuri council home',
     'vet fight war fed demand money back illeg refuge keep money',
     'mr trump wild ride new york time',
     'fbi director comey bamboozl doj congress clinton',
     'food natur unclog arteri prevent heart attack',
     'death two state solut',
     'comment parent date asleep car cop arriv kill facespac',
     'donald trump team show sign post elect moder new york time',
     'miami beach tri tame raucou street fishbowl drink stay new york time',
     'doctor mysteri found dead summit breakthrough cure cancer',
     'donald trump unsink candid new york time',
     'shock new mock hillari ad campaign warn take us war enlistforh fightforh dieforh',
     'exclus famili slain border patrol agent brian terri say eric holder among real crimin respons',
     'trump tell report wall work ask israel breitbart',
     'america surviv next year',
     'commission start press cleveland indian logo new york time',
     'un plan implant everyon biometr id drill',
     'trump attack senat credibl gorsuch comment new york time',
     'clinton advisor lose leak email hillari illeg activ',
     'art laffer paul ryan perfect right breitbart',
     'donald trump blame econom crash',
     'pokemon go player inadvert stop peopl commit suicid japan',
     'california senat race tale divers flail g p new york time',
     'exclus sourc say megyn kelli would welcom back fox news',
     'break preced obama envoy deni extens past inaugur day new york time',
     'brexit vote go monti python may offer clue new york time',
     'blind mystic predict bad news trump',
     'total vet fail left wing snowden fan girl realiti winner get access nsa secret',
     'somalia u escal shadow war new york time',
     'free care bless victim orlando nightclub attack new york time',
     'durabl democrat counti countri could go trump',
     'fed challeng rais rate may existenti new york time',
     'russia intent attack anyon absurd say vladimir putin',
     'f investig errant flight involv harrison ford new york time',
     'fed rais key interest rate cite strengthen economi new york time',
     'la expresi n lo siguient ya es la utilizada lo siguient en el castellano',
     'trump berat news media new strategi need cover new york time',
     'u drone strike target taliban leader new york time',
     'u intellig expect al qaeda attack monday new york virginia texa',
     'told cannabi great revers alzheim',
     'report megyn kelli kick nbc show kardashian famili interview',
     'local percent may rich think new york time',
     'dr david duke mark collett uk collett explain duke trump victori would chang polit forev',
     'statement senat well fargo chief deepli sorri new york time',
     'cnn statement distanc network buzzfe fake news dossier breitbart',
     'c e ponder new game trump rule new york time',
     'spicer bradi stolen jersey anoth bad press breitbart',
     'scaredi cat investig peopl enjoy fear new york time',
     'left vision',
     'showdown loom u question chines deal german chip design new york time',
     'trump administr take harder tack trade china new york time',
     'pew american trust level feder govern plummet histor low breitbart',
     'islam state support former nation guardsman plead guilti terror charg virginia',
     'spicer report go rais hand like big boy girl breitbart',
     'leader applaud gorsuch confirm win pro life movement',
     'newstick',
     'french vogu march cover featur transgend model new york time',
     'trump veer parti line gun control new york time',
     'oligarchi prepar groundwork steal elect',
     'ya hay reencuentro de operaci n triunfo que edicion de operaci n triunfo',
     'chatsworth hous tale five centuri new york time',
     'uncomfort love affair donald trump new england patriot new york time',
     'john mccain withdraw support donald trump disclosur record new york time',
     'strang unend limbo egypt hosni mubarak new york time',
     'poverti rose u hous district obama presid',
     'huma abedin seek fbi immun deal',
     'singl mom escap friend zone one non date time new york time',
     'boe suit futur spaceflight new spacesuit design breitbart',
     'trump float oliv branch might keep part health law new york time',
     'wapo tri compar elizabeth warren break senat rule milo shut violent riot breitbart',
     'crumpl school bu leav chattanooga daze new york time',
     'die came back life incred messag human',
     'fight nation african american museum new york time',
     'father manchest suicid bomber arrest libya breitbart',
     'secret true leader',
     'muslim demand local walk dog public violat sharia disrespect',
     'hillari campaign bed pac staff donat k fbi agent wife investig',
     'still tri flip elector colleg block trump win',
     'al sharpton dem point appeal archi bunker trump voter breitbart',
     'think mani doom sayer trump get offic',
     'democrat jump session resign band wagon breitbart',
     'alt right architect glenn beck open fire alt right grave threat republ audio tweet',
     'politic justic protect hillari',
     'north carolina satur surpris reel hurrican matthew new york time',
     'live love submit memori new york time',
     'achiev mind work medit cushion requir new york time',
     'world first zero emiss hydrogen power passeng train unveil germani',
     'confus chip credit card get line new york time',
     'brook trump side foreign leader us presid israel russia breitbart',
     'toxic air home get rid natur',
     'connecticut reader report record voter registr inspir trump',
     'germani react merkel trump visit could lot wors new york time',
     'justin rose outduel henrik stenson golf gold medal new york time',
     'iceland water cure new york time',
     'shorten l train shutdown month new york time',
     'time presid decid new york time',
     'u n relief offici call crisi aleppo apex horror new york time',
     'berkeley treat violent anti speech left like kkk',
     'statist propaganda mani syrian us regim chang kill',
     'self help guru jame altuch own thing new york time',
     'trump religi liberti order give session major leeway breitbart',
     'demoledor amparo contra salgado keiko congresista',
     'era trump china presid champion econom global new york time',
     'sad saga john walker lindh rebel without clue',
     'court disagre michigan vote recount new york time',
     'berni sander feud democrat leadership heat new york time',
     'shortest power explan trump victori ever seen',
     'russia look popul far east wimp need appli new york time',
     'johnson amend trump vow destroy explain new york time',
     'donald trump march life full support',
     'polic offici found dead long island suicid suspect new york time',
     'radic chang store world global market readi',
     'like make showbiz best friend new york time',
     'trump nomin neil gorsuch suprem court new york time',
     'bidder cast doubt serious mexican border wall project',
     'stake us elect',
     'israel approv addit fund settlement west bank new york time',
     'tx gov abbott sign legisl could put sheriff sanctuari citi jail breitbart',
     'donald trump hold thank ralli cincinnati announc pick defens secretari new york time',
     'lawmak look bipartisanship health care new york time',
     'insid conserv push state amend constitut new york time',
     'donald trump tell n r hillari clinton want let violent crimin go free new york time',
     'pope franci trump japan tuesday brief new york time',
     'mayorsstand day tout support illeg immigr',
     'trump campaign celebr',
     'democrat drag jeff session confirm fight breitbart',
     'break news podesta brother pedo ring mr trump drain swamp v guerrilla economist',
     'la frase destacada del debat de investidura',
     'confus jihad hirabah build peac world',
     'lazi liber journalist smear bannon',
     'australia close detent center manu island still accept asylum seeker new york time',
     'politico hillari clinton run breitbart',
     'giant lynx make ador sound whenev human rub face',
     'minnesota cop found guilti philando castil shoot trial',
     'million american kill minut',
     'wayn madsen cia alway serv interest wall street',
     'novemb daili contrarian read',
     'flashback report obama campaign rep talk iran hama',
     'lesseroftwoevil',
     'aya cash first time ate veget new york time',
     'trump advis say isra settlement illeg',
     'serena william prevail open problem new york time',
     'soul man sam moor honor perform trump inaugur',
     'ferrel came back bush destroy trump video',
     'chines govern concern tough talk trump cabinet breitbart',
     'billionair report seiz hong kong hotel taken china new york time',
     'easi know link',
     'homebodi find ultim home offic new york time',
     'ann coulter unload paul ryan deepli unpopular obamacar bill breitbart',
     'transgend bathroom debat turn person vermont high school new york time',
     'obama hillari want libya gaddafi toppl kill',
     'alert former soro associ warn pro wrong gold silver skyrocket like',
     'hillari panick fbi weiner email',
     'chaotic minut trump defend fine tune machin new york time',
     'one polic shift patrol anxiou america new york time',
     'fight ghost fascist aid real one',
     'monday even brief brexit abort game throne new york time',
     'eu increas brexit bill demand billion billion',
     'arianna huffington sleep revolut start home new york time',
     'iranian missil accident destroy iranian ship aim syria',
     'sonoma counti california vote creat largest gmo free zone america',
     'fbi comey wikileak intellig porn journal breitbart',
     'clare waight keller name first femal design givenchi new york time',
     'realiti face black canadian nation shame',
     'top nfl draft prospect caleb brantley charg punch woman face breitbart',
     'video pacif crest trail associ lavoy finicim murder',
     'life lesson man seen death',
     'trump g p work win repeal obama health act new york time',
     'atlant goldberg confid trump handl matter life death breitbart',
     'orthodox rabbi support trump',
     'nuclear tension us russia reach danger point',
     'gambia join south africa burundi exodu intern crimin court',
     'peyton man golf presid trump sunday',
     'obama urg donald trump send signal uniti minor group women new york time',
     'like miracl woman give birth use ovari frozen sinc childhood new york time',
     'uconn recip success run run run new york time',
     'assang final afford opportun give statement rape accus',
     'hillari puppet show much hillari care god omiss word',
     'stock market gone high problem new york time',
     'john kerri reject suggest u involv turkey coup new york time',
     'trump aid stephen miller u absolut sovereign right determin cannot enter countri breitbart',
     'year old russian girl speak languag',
     'power corpor lobbi quietli back hillari nobodi talk',
     'maintain sunni spirit face hardship new york time',
     'rush limbaugh comey fire epic troll trump dem breitbart',
     'twitter sue govern block unmask account critic trump new york time',
     'warrior resili home cruis cavali new york time',
     'comey letter clinton email subject justic dept inquiri new york time',
     'spain malta u uk pressur refus allow russian carrier group refuel port',
     'review radiohead moon shape pool patient perfection new york time',
     'man militar polic stand rock work',
     'woman arrest properti land stolen dapl',
     'pulitz prize new york time win daili news propublica share public servic award new york time',
     'vanquish wit takeov bush clinton attend donald trump inaugur breitbart',
     'specul possibl obama pardon edward snowden bow bergdahl chelsea man breitbart',
     'cheesi mash potato soul new york time',
     'unprincipl wapo editor damn comey critic join',
     'nation wreck immigr civil war brew good swede turn muslim migrant violenc rape murder',
     'man shot dead offic crucifix gun polic say new york time',
     'exclus rep jim jordan trump first day think great start breitbart',
     'haley attack syria one presid finest hour breitbart',
     'justin bieber defec ador irish fan hotel window',
     'cancer agenc fire withhold carcinogen glyphos document',
     'gianno caldwel claim hillari care black vote black live',
     'isi use ramadan call new terrorist attack new york time',
     'daili show mock mahatma blondi megyn kelli nbc debut',
     'health care bill failur part art deal breitbart',
     'lewandowski comey liar look sign major book deal breitbart',
     'kimberli guilfoyl discuss potenti white hous press secretari job interview',
     'clinton campaign chairman john podesta invit occult spirit cook dinner marina abramovi',
     'obama coalit crumbl leav open trump new york time',
     'pregnanc chang brain way may help mother new york time',
     'fema open loan window red cross tri shut shelter',
     'gaiaport interweb gaia energet strengthen',
     'scarborough trump poop pant call modern art breitbart',
     'cook invest time work new york time',
     'bad news jackson famili woman leak star sick k sex secret',
     'gorsuch london republican parti thursday even brief new york time',
     'emma morano last person born die breitbart',
     'loserpalooza craziest scene anti trump protest breitbart',
     'trump victori mean africa',
     'vision life mar earth depth new york time',
     'obama cancel talk rodrigo dutert philippin say regret slur new york time',
     'new jersey increas ga tax end long polit stalem new york time',
     'abc manchest attack like inflam anti islam sentiment breitbart',
     'hidden plain sight global depopul agenda',
     'job american new york time',
     'sander ask obama interven dakota access pipelin disput',
     'googl add job section search engin includ employ rate breitbart',
     'cri jordan meme die new york time',
     'dem rep nchez trump use fear muslim immigr promot polici undermin valu breitbart',
     'announc saker commun german saker blog vineyard saker',
     'shoot victim famili watch gun measur stall new york time',
     'imag reveal crash schiaparelli mar lander',
     'grassroot coalit share mani question concern betsi devo senat',
     'octob daili contrarian read',
     'trump inaugur ball work begin play game breitbart',
     'trump choic stephen bannon nod anti washington base new york time',
     'media outrag white hous exclus fake news breitbart',
     'benni morri unten denial ethnic cleans palestin',
     'obama hillari clinton pardon could heal divid nation',
     'except handl',
     'nota conceptu para la presidencia de rusia',
     'respons philippin presid fatal blast rais fear new york time',
     'david adjay design museum speak differ languag new york time',
     'bulletin righteou jew trump rule ruin gop establish etc item',
     'britain reduc terror level one notch sever terror cell arrest',
     'mute alon never short kind word friend new york time',
     'iranian militari command claim rogu nation send elit fighter infiltr us europ',
     'account',
     'watch muslim student claim non believ kill islam countri breitbart',
     'yemen yet anoth fals flag protect saudi us interest middl east new eastern outlook',
     'watch muslim palestinian declar follow prophet muhammad kill christian jew',
     'street dog kerala call upon superdog krypto rescu human',
     'israel track anti govern journalist facebook',
     'new alaska law take first step common core',
     'key baylor footbal execut demarko butler fire text scandal breitbart',
     'calgari airport arriv yyc',
     'georg michael wrestl fame frank sinatra advic new york time',
     'sharon old laureat sexual scrutin bodi ode new york time',
     'blue collar elect shock liber media',
     'die new york time',
     'polic fire rubber bullet pipelin protest',
     'ann coulter hit suppos gay icon kathi griffin isi crib antic',
     'contamin food china enter u organ label',
     'ten famou peopl read summer new york time',
     'hillari clinton knew year ago anthoni weiner pedophil wikileak',
     'million stairway nowher far west side new york time',
     'cyber war trifl catastroph inform',
     'presid putin ask us stop provok russia',
     'inquiri cloud de blasio bid come strong manag race new york time',
     'us drone strike afghanistan kill wound sever civilian',
     'bobbi hutcherson vibraphonist colorist rang sound die new york time',
     'amnesti advoc boycott agenc meet pro american advoc invit breitbart',
     'project verita robert creamer illeg foreign wire transfer caught tape',
     'ex flotu michel obama trump want feed crap kid',
     'south sudan slide closer war gunfir rumbl capit new york time',
     'nation mood focu group reflect angri divid america',
     'bankrupt puerto rico vote u statehood breitbart',
     'overwhelm brexit basic new york time',
     'seattl gay mayor accus sexual molest teen breitbart',
     'gender fluiditi runway new york time',
     'us nato attack putin militari drill russia world war red alert kopya',
     'new clinton email came underag sex pest anthoni weiner',
     'montana bear attack lesson hope surviv first aid new york time',
     'best america new york time',
     'pecan step pie plate new york time',
     'barack obama plagiar tell word matter breitbart',
     'lower back ach activ wait new guidelin say new york time',
     'sleep hour peopl die new york time',
     'veep season episod littl danc new york time',
     'dr oz trump offer placebo transpar new york time',
     'comment gold medalist wrestler get violent polic cop choos engag deadli forc buck roger',
     'one year water orang counti four day breitbart',
     'doctor enemi afghan forc target f hospit new york time',
     'year old hebrew mention jerusalem found',
     'spirit cook disturb podesta email yet warn graphic content',
     'donald trump misstep risk put ceil support swing state new york time',
     'green parti margaret flower challeng us senat debat maryland undemocrat',
     'offic rescu drown deer pool quick think',
     'polic suspect punch armi veteran steal servic dog outsid home bronx breitbart',
     'yike megyn kelli receiv rude awaken remind replac',
     'iran send elit irgc warfight europ unit state prepar battl',
     'hillari endors donald trump presid accord wikileak',
     'woodward trump dossier garbag document intellig chief apolog trump breitbart',
     'emmi nomin traci morgan emot return saturday night live new york time',
     'trump pick mick mulvaney south carolina congressman budget director new york time',
     'donald trump el primer president naranja de lo estado unido',
     'attorney gener loretta lynch plead fifth',
     'russia u missil defens pose deep risk secur asia',
     'washington plan b syria realli mean',
     'upset brexit british jew look germani new york time',
     'girl ask boyfriend give iphon use money buy hous',
     'amnesti intern slam obama gov kill civilian syria',
     'find flock rural writer book club new york time',
     'venezuela crisi enter danger phase maduro foe go milit',
     'emigr super bloc part viii quasi legal coup hillari clinton inform oper elect',
     'could question ask love one vote way new york time',
     'bbc ask realli happen clinton haiti',
     'trump pick thoma bossert top counterterror advis new york time',
     'come unglu',
     'trump jr suspici help arizona woman push stall car use photo op video',
     'suicid chicago polic offic skyrocket',
     'think suv safe shock video',
     'peopl tortur kill sufi muslim shrine pakistan',
     'comment architectur clever arrang brick man transform heal process architectur clever arrang brick man transform heal process new earth media',
     'hybrid war strategi africa introduct',
     'julian assang speak prerecord rt interview',
     'watch israel love hollywood actor issu major plea america minut rock elect',
     'n l donald trump botch independ day moment new york time',
     'exclus congresswoman marsha blackburn say elimin net neutral preserv open internet breitbart',
     'attach year round greenhous home',
     'sheriff clark obama final day obama like tenant evict properti gonna trash place way door video',
     'weird ban women iran',
     'end game close clinton deep state turn',
     'trump fight chanc establish new eastern outlook',
     'trade stanc toward china clinton trump signal chill new york time',
     'review secret life pet amus miss opportun new york time',
     'first case demonetis relat hiv man unprotect sex atm machin',
     'thing need know dakota access pipelin protest',
     'koch brother battl prevent dark money disclosur south dakota',
     'milo berkeley event evacu mask protest light fire storm venu breitbart',
     'authoritarian west demon strong popular leader',
     'swedish journo sweden collaps without illeg migrant',
     'promis internet career vine dead write newsbiscuit cheer',
     'iowa trump voter unfaz controversi new york time',
     'radic american grown marxist terror group made announc januari th conserv daili post',
     'guardian opinion writer trump bannon count terrorist massacr',
     'putin pro trump onlin troll spill bean samantha bee',
     'along mosul front line desper civilian dug troop fighter new york time',
     'second avenu subway open train delay end happi tear new york time',
     'year old cher don see top nippl pasti billboard award',
     'report stop nine percent illeg alien border crosser would pay trump border wall breitbart',
     'ice union issu final warn voter',
     'elit new york polic unit rehears terrorist attack new york time',
     'trump hollywood star vandal out shameless new video taunt polic seek',
     'california today view san francisco lean tower space new york time',
     'matthew trump speech putin say america first hitlerian background breitbart',
     'vladimir putin valdai intern discuss club shape world tomorrow vladimir putin',
     'chines social media rage unit airlin controversi breitbart',
     'america better without border',
     'chao terror tie make venezuela direct threat usa former un secur council presid say breitbart',
     'rick ross documentari video doubl ad checker new york time',
     'oil spill pacif ocean sinc last month total ignor',
     'podesta mill go dump email',
     'thoughtlessli disbeliev conspiraci theori need read',
     'dr david duke dr slatteri expos hillari treason trump duke win',
     'russian frigat syrian cost blast terrorist hq cruis missil video',
     'hillari clinton like covert action stay covert transcript show new york time',
     'us kurdish troop involv invad isi capit raqqa',
     'awaken human await fulli script end controversi elect day near',
     'georg soro back climat march bring celeb nation mall swelter saturday breitbart',
     'edward snowden long strang journey hollywood new york time',
     'black trump man belong murder cult',
     'bill clinton lover call ruthless hillari warden',
     'russia cina e arabia saudita domano l egemonia del dollaro di ariel noyola rodr guez',
     'third month india cash shortag begin bite new york time',
     'one trump administr posit gain popular go shock breitbart',
     'bill clinton said white middl class life expect declin obama year',
     'furiou eric holder issu dire warn comey partisan smear',
     'memo trump action day',
     'chao desper thousand flee aleppo amid govern advanc new york time',
     'wildfir empti fort mcmurray alberta oil sand region new york time',
     'schumer session investig seem violat recus breitbart',
     'amid divis march washington seek bring women togeth new york time',
     'zoe saldana trump hollywood got cocki becam arrog bulli',
     'anti establish trump plan appoint goldman sach georg soro insid',
     'cholesterol drug men gonad',
     'trump flynn treat unfair fake media illeg leak breitbart',
     'california today virtual realiti investig trayvon martin case new york time',
     'nearli decad later apolog lynch georgia new york time',
     'marcia clark final moment savor emmi new york time',
     'brexit speech theresa may outlin clean break u k new york time',
     'man hate black men found victim care other new york time',
     'michael flynn fail disclos incom russia link entiti new york time',
     'feder judg throw convict c sniper four life sentenc',
     'donald trump obama thanksgiv weekend brief new york time',
     '',
     'vine celebr life death app clip',
     'squatti potti ceo griffin imag divis disturb decis realli',
     'thing let go new year',
     'comment polic union post pic hillari arrest facebook dan',
     'koch brother secretli alli w georg soro hillari clinton',
     'trump expand search secretari state new york time',
     'teacher yr old wait trump elect go deport muslim',
     'cost choic top concern health insur custom new york time',
     'shaq announc plan run sheriff breitbart',
     'illeg immigr cross border vote',
     'pirat fail take helm iceland pirat parti gain mileag enough steadi ship alon',
     'billionair build davo new york time',
     'run danger alaskan trail new york time',
     'la pel cula de su vida descubr que ha llevado siempr un trozo de lechuga entr lo dient',
     'dem win congression basebal game give trophi republican steve scalis breitbart',
     'franc present strict gun control paper tiger breitbart',
     'china russia silk road commod nixon massiv bull market gold silver',
     'troubl quarterback johnni manziel appear shop mall sign autograph next super bowl breitbart',
     'bob dylan accus lift part nobel prize speech sparknot',
     'blimp crash e coli contamin snakebitten u open wit spectat death breitbart',
     'public employe shadow world american carnag',
     'espn lz granderson justifi think kaepernick blackbal nobodi sign breitbart',
     'prescript painkil death drop state legal marijuana',
     'low growth world get new york time',
     'istanbul donald trump benjamin netanyahu morn brief new york time',
     'hitler hillari',
     'dalian wanda hollywood event product new york time',
     'comment pm water cooler timmi',
     'breakdown clinton money machin',
     'california forefront climat fight back trump new york time',
     'g e year old softwar start new york time',
     'hillari clinton hamilton would enough new york time',
     'brain concuss children adult know vaccin damag',
     'trump great paul craig robert',
     'hillari frantic dirti secret implod get wors prison bombshel',
     'rent car know rule road new york time',
     'impeach polit brazil someth sinist new york time',
     'trump interview moder view defi convent new york time',
     'review sweetbitt bright light big citi restaur set new york time',
     'gonzaga beat south carolina final four shot anoth first new york time',
     'gorka trump interventionist command chief noth chang breitbart',
     'thank fbi clinton email investig shift poll number significantli trump favor',
     'race class dictat republican futur',
     'gorsuch scalia lion law judg look law demand prefer breitbart',
     'nc state provid student post elect comfort food therapi',
     'iraqi troop push mosul kill across iraq',
     'pregnant women turn marijuana perhap harm infant new york time',
     'biggest winner loser u presidenti elect',
     'trump psychic listen word one year ago look happen',
     'mccain trump attack press dictat get start breitbart',
     'judgment day one reason everi christian jew america vote donald trump',
     'coup stolen elect',
     '',
     'thing need know black dakota access pipelin protest',
     'germani iraqi asylum seeker convict rape chines student',
     'teenag boy knock classmat assault femal teacher face epic',
     'take bring hillari clinton justic',
     'ice round crimin alien texa capit',
     'elect hate grief new stori',
     'hillari question michel obama help',
     'keith vaz british lawmak quit senior post amid sex drug scandal new york time',
     'comment sunday devot whole univers grain sunday devot whole univers grain fellowship mind kommonsentsjan',
     'donald trump syria emperor akihito morn brief new york time',
     'hell frozen michel obama made hillari destroy move twitter',
     'flynn critic call nuclear scientist miss usa dumb diss femin call health care privileg breitbart',
     'hampshir colleg student accus assault basketbal player wear hair braid claim cultur appropri breitbart',
     'obamamomet toxic legaci rule lawless',
     'simon manuel gold rippl beyond pool new york time',
     'vote machin program order steal elect',
     'redraw tree life scientist discov new bacteria group stun microbi divers underground',
     'trump putin destroy isi',
     'photo game camera catch glimps possibl antler buck',
     'turkey relat europ sink amid quarrel netherland new york time',
     'wikileak document reveal unit nation interest ufo video',
     'blue state blue deliber politic intimaci',
     'clare hollingworth report broke news world war ii die new york time',
     'north miami polic offic shoot man aid patient autism new york time',
     'path total dictatorship america shadow govern silent coup',
     'effort curb polic abus mix record uncertain futur new york time',
     'signal major bottom gold silver',
     'hey ho old england embrac punk rock year later new york time',
     'cano reek genocid theft white privileg say canadian professor',
     'illeg immigr advoc pledg resist deport trump',
     'cdc scientist confirm donald trump right vaccin autism',
     'breitbart news daili drain swamp breitbart',
     'trump media frankenstein monster',
     'watch toni romo say goodby dalla cowboy instagram video breitbart',
     'global far right conspiraci theori buoy trump new york time',
     'georgia father convict murder toddler death hot car new york time',
     'richard boll die wrote color parachut new york time',
     'stranahan steve bannon nail media fight trump breitbart',
     'texa elector expect massiv corrupt relat elector colleg vote',
     'guilti power nullif counteract govern tyranni',
     'trillion new debt day',
     'tori councillor say homeless peopl elimin',
     'review hillbilli elegi tough love analysi poor back trump new york time',
     'crimin chief',
     'govern lie movi',
     'trump brexit defeat global anyway',
     'face congress sport offici begin confront sexual abus new york time',
     'airbnb end fight new york citi fine new york time',
     'big pharma martin shkreli suspend twitter breitbart',
     'next big tech corridor seattl vancouv planner hope new york time',
     'evid robot win race american job new york time',
     'see africa road new york time',
     'happen hip hop hillari goe dead broke brace',
     'bill maher high trump state free speech new era new york time',
     'l p g tour donald trump complic new york time',
     'legend art cashin trump presid new world order gold brexit great depress see panic',
     'fl sheriff day goe arrest lot illeg alien prey peopl breitbart',
     'donald trump threaten cancel berkeley feder fund riot shut milo event',
     '',
     'bill herz last war world broadcast crew die new york time',
     'gari johnson equat syria death caus assad west new york time',
     'republican senat bill defund un anti israel resolut',
     'rose evanski pioneer women hairstyl die new york time',
     'blackston saudi arabia announc billion invest u infrastructur breitbart',
     'satur fat heart diseas greatest scam histori medicin',
     'dem sen merkley gorsuch nomin court pack scheme turn nomin breitbart',
     'four common mistak burn wood',
     'push internet privaci rule move statehous new york time',
     'madonna gave surpris pop concert support clinton new york time',
     'trump support plan anti trump ralli lack tax reform',
     'british healthcar offer glimps futur obamacar',
     'trump iranian presid rouhani better care breitbart',
     'break sec defens carter attempt fool american public veteran caught red hand',
     'putin advis take credit trump victori mayb help bit wikileak',
     'warren buffett stake suggest appl grown new york time',
     'fbi clinton email investig shift poll number significantli trump favor',
     'hillari horrifi pic surfac overnight want',
     'paul lepag governor main say quit new york time',
     'colorado radio station paul martin interview dave hodg elect fraud stand rock',
     'bill clinton want call someth complet ridicul hillari elect',
     'son death salli mann stage haunt show new york time',
     'trump team link russia crisscross washington new york time',
     'north korean arrest kill kim jong un half brother new york time',
     'surgeon admit mammographi outdat harm women',
     'war street pari arm migrant fight run battl french capit',
     'oscar voter meryl streep nomin anti trump speech',
     'texa enact anti sharia law',
     'chicago polic board chair windi citi need feder help turn tide crime breitbart',
     'look beyond novemb th song oligarchi doom',
     'neil young celebr st birthday perform stand rock',
     'exclus amid paul ryan obamacar push mississippi chri mcdaniel prep potenti senat run breitbart',
     'boom short list peopl inspir michel obama',
     'artist go boycott grammi face fallout fraught award new york time',
     'video idiot destroy trump hollywood star get bad news second later',
     'elit want global economi collaps',
     'best health benefit sweat',
     'senat confirm scott pruitt e p head new york time',
     'cricket snake crab mix fact fraud new york subway new york time',
     'session potenti deputi face stern test russia inquiri new york time',
     'suicid squad top box offic second weekend new york time',
     'health insur plan rate hike obamacar exchang breitbart',
     'look like someon think democrat ohio full manur',
     'syrian war report novemb govt forc relaunch offens oper insid outsid aleppo',
     'mysteri solv get hillari clinton get movin twitchi com',
     'senat narrowli pass rollback obama era auto r rule new york time',
     'cramp costli bay area cri build babi build new york time',
     'champion optim obama hail clinton polit heir new york time',
     'fiona appl releas trump protest chant new york time',
     'susan rice u must integr lgbt right gov foreign polici',
     'trump organ move avoid possibl conflict interest new york time',
     'sweden brink polic forc push break point violenc amid migrant influx',
     'comment hemp vs cotton ultim showdown hemp readdress cannabi kuebiko co',
     'jare kushner trump son law clear serv advis new york time',
     'fisherman face life prison catch worth cocain sell breitbart',
     'step ring roll punch new york time',
     'adnan sy serial podcast get retrial murder case new york time',
     'scientist say weird signal space probabl alien',
     'u swimmer disput robberi claim fuel tension brazil new york time',
     'way take self care vacat new york time',
     'economist sign letter urg america vote donald trump',
     'campaign long expens chaotic mayb good thing new york time',
     'alien megastructur star target million seti search',
     'brazen kill myanmar lawyer came spar militari new york time',
     'jane pauley back new york time',
     'deutsch bank consid altern pay cash bonu',
     'danni dyer footbal foul up dvd second',
     'lack oxford comma could cost main compani million overtim disput new york time',
     'european parliament committe consid legal right robot breitbart',
     'rex tillerson aggress dealmak whose tie russia may prompt scrutini new york time',
     'fbi conduct new investig email clinton privat illeg server',
     'heart mine us empir cultur industri',
     'report megyn trash trump newt murdoch announc replac avail',
     'russian scientist track sea lion space',
     'seiz definit popul reuter warn us chao come',
     'seaworthi readi earli unveil new york time',
     'donald trump add k mcfarland nation secur team new york time',
     'snowstorm bring wintri mix slush gripe new york time',
     'republican wilder tillerson thursday even brief new york time',
     'review bryan cranston shine lyndon johnson way new york time',
     'samsung urg consum stop use galaxi note batteri fire new york time',
     'elect result discuss presidenti elect open thread',
     'ag lynch told fbi director comey go public new clinton email investig',
     'donald trump rise white ident polit',
     'red blue divid six view america new york time',
     'franc identifi nd man attack church kill priest new york time',
     'dreamer arrest nationwid gang crackdown',
     'maxin water american public get weari trump impeach yet breitbart',
     'wwn horoscop',
     'deport italian mobster caught sneak across u mexico border',
     'leak audio hillari clinton push rig palestin elect',
     'iranian saudi proxi struggl tore apart middl east new york time',
     'lavrov kerri discuss syrian settlement',
     'l influenc de usa et de l otan dan le rapport de l ue avec la chine manlio dinucci',
     'open border group gird h b fight',
     'weinerg expos darker dirtier secret imagin',
     'trump elect break chain polit correct',
     'whitehous gov take climat page put america first energi plan breitbart',
     'iran warn presid elect trump mess sweetheart nuclear deal obama',
     'toni perkin trump eo affirm jefferson doctrin separ church state',
     'moon fell heaven',
     'mom star launch campaign plan parenthood',
     'assad lesson aleppo forc work consequ new york time',
     'penc bossert bannon demot continu play import polici role breitbart',
     'bundi ranch occupi acquit count challeng corrupt bureau land manag',
     'report googl face fine billion eu antitrust case breitbart',
     'ask thom york write cover quot book',
     'internet flasher',
     'gretchen carlson suit aim retali discrimin new york time',
     'googl launch ai program detect hate speech breitbart',
     'poll show hillari lead useless mislead cartoon',
     'last second lane merger good traffic new york time',
     'macau skip casino embrac past new york time',
     'montreal ungainli unlov christma tree new york time',
     'photo jupit nasa spacecraft near far new york time',
     'trump labor pick andrew puzder critic minimum wage increas new york time',
     'nico rosberg take formula one driver titl despit lewi hamilton win abu dhabi new york time',
     'sensori isol tank taught brain new york time',
     'chelsea handler botch tweet attack trump grandchild',
     'report trump move tax reform plan without speaker paul ryan breitbart',
     'ag jeff session unveil program acceler deport imprison illeg breitbart',
     'former us attorney dc new hillari email probe result revolt insid fbi',
     'keep appear ruin former dalla banker new york time',
     'sesam seed knee osteoarthr',
     'white cop interact black real life',
     'pennsylvania republican push ban privat gun sale breitbart',
     'review warcraft orc differ domain fight heart new york time',
     'good peopl share bad info need fact check click share',
     'charit wed registri new york time',
     'nypd raid hillari properti found ruin life usa newsflash',
     'feinstein gorsuch originalist doctrin realli troubl origin would allow segreg breitbart',
     'confront flare obama travel parti reach china new york time',
     'reason appli job trump administr',
     'thought silver market rig',
     'hear agn martin seren john zorn frenzi music new york time',
     'must see welcom famili mani hispan american vote donald trump',
     'iraqi forc enter western mosul fierc battl isi new york time',
     'european futur putin migrant crisi video',
     'roll stone defam case magazin report order pay million new york time',
     'hillari sick tire suffer weiner backup',
     'taiwan itali joe mcknight friday even brief new york time',
     'shiit militia say close tal afar turkey warn limit',
     'brother clinton campaign chair activ foreign agent saudi arabian payrol',
     'must fight trump goe conserv freedom caucu new york time',
     'isi kidnap kill least civilian afghanistan',
     'hold hillari account',
     'break ted cruz call special prosecutor investig hillari truthfe',
     'spare gunman charleston churchgoer describ night terror new york time',
     'war satur fat harm peopl poor countri shun tradit fat like coconut oil',
     'democrat garland mind mobil suprem court fight new york time',
     'comment best kind milk dairi best kind milk dairi collect evolut apg editori',
     'harri reid blast comey misconduct drop bombshel fbi sit russian trump info',
     'nation review conservat inc plan cave even immigr',
     'watch thug call us marin pussi bare live tell tale',
     'next us presid psycho lesbian plu break news video',
     'critic see effort counti town purg minor voter roll new york time',
     'even brief hillari clinton donald trump cultur revolut new york time',
     'indiana parent lose babi year live jail abus say never happen',
     'roll stone paint blue new album new york time',
     'uber extend oliv branch local govern data new york time',
     'wilder put dutch first brussel africa asylum seeker',
     'donald trump good educ enrich mind soul',
     'watch brad pitt play afghanistan war gener war machin teaser breitbart',
     'two power earthquak strike central itali',
     'indoor garden made easi nutritow',
     'elizabeth warren defin sleazi hypocrisi',
     'rush limbaugh reilli departur natur campaign breitbart',
     'museum truste trump donor support group deni climat chang new york time',
     'cecil richard credit plan parenthood support stop ahca breitbart',
     'switch chip know anymor',
     'u conced million payment iran delay prison leverag new york time',
     'australia say foil terrorist plot new york time',
     'orovil dam state feder govern share blame',
     'daili traditionalist jeff schoep nsm',
     'michael moor joe blow vote trump ultim f elit human molotov cocktail',
     'jaguar owner shahid khan oppos trump immigr ban new york time',
     'anti trump advert side bu realli visual clever see motion',
     'piano man mani face stranger stori new york time',
     'ticket releas harri potter curs child new york time',
     'fbi visit man home film us postal distribut center',
     'hispan crowd boo marco rubio stage',
     'fire tv report receiv thousand sexual violent threat',
     'detain illeg alien end day hunger strike',
     'break silenc offic testifi kill walter scott new york time',
     'illeg immigr allegedli kill park spot',
     'presid trump honor littl sister poor first white hous nat l day prayer year breitbart',
     'hillari clinton support call recount vote battleground state',
     'hbo scrap jon stewart anim comedi seri',
     'sentenc murder rare book dealer',
     'latest stock market invest book financi market',
     'fashion industri ceo support plan parenthood civic respons',
     'anti trump protest paid stage craigslist reveal',
     'men exercis put damper sex life new york time',
     'bake soda coconut oil kill cancer eye open evid',
     'trump cite evid suggest susan rice commit crime new york time',
     'breitbart news daili gorsuch scotu breitbart',
     'hillari arrest',
     'siri open smart lock let neighbor walk hous',
     'donald trump michael phelp zika tuesday even brief new york time',
     'obama furiou fed deplor drop piec gift',
     'l mark year sinc rodney king riot breitbart',
     'de facto us al qaeda allianc inform',
     'cori booker paul rever moment underway russian come breitbart',
     'rapper troy ave shot brooklyn new york time',
     'seattl judg ignor jihad convict prior impos refuge reform ban breitbart',
     'texa student skip school protest arrest violent crimin',
     'trump aid tri reassur europ mani wari new york time',
     'china trigger next global recess',
     'bomb kill baghdad new york time',
     'comment black racism martin wright',
     'thank fbi clinton email investig shift poll number significantli trump favor',
     'u rescu attempt afghanistan miss western hostag hour new york time',
     'rick rule look sprott asset manag client money right',
     'farm owner arrest protest dakota access pipelin theft land',
     'trump pois lift ban c black site prison new york time',
     'tesla musk investig solarc congress',
     'ridicul stupid thing men keep women',
     'press tv duff un condemn moder terrorist',
     'nemesi scourg western world',
     'trump budget new foundat american great breitbart',
     'exclus islam state support vow terror group retak mosul liber',
     'roll thunder motorcyclist return c honor pow mia breitbart',
     'interest fun fact stethoscop',
     'china seek bigger role world stage xi jinp go davo world econom forum new york time',
     'veteran prepar join stand rock protest stop dakota access pipelin',
     'photo latin america condemn venezuela excess use forc protest breitbart',
     'sugar feed cancer cell may even creat',
     'john kerri urg ground militari aircraft key area syria new york time',
     'outsid money favor hillari clinton rate donald trump new york time',
     'fake news new york time target breitbart report truth breitbart',
     'blast new yorker examin psycholog shrapnel new york time',
     'resist schwarzenegg call grassroot revolut u exit pari agreement',
     'virginia offici request u inquiri inmat death jail new york time',
     'report voter fraud crash cours',
     'associ press report admit fake news stori hillari clinton',
     'review garth brook bring rous anthem ballad yanke stadium new york time',
     'sicher trainieren beim spin gilt ab sofort helmpflicht',
     'everi asset class collaps need look wealth term mani chicken much ga buy',
     'review halt catch fire time travel silicon valley dawn new york time',
     'obama use religi test favor muslim christian',
     'wingsuit flyer vs tree',
     'turkey say airport bomber kyrgyzstan russia uzbekistan new york time',
     'turmer power artifici drug',
     'facebook caught sell target advertis exclud differ race',
     'gatlinburg resid return home wildfir destruct new york time',
     'year old girl use human bomb nigeria attack',
     'shock berkeley poll california voter want democrat work trump breitbart',
     'famili terror attack victim sue twitter provid resourc servic isi breitbart',
     'philippin leader vow pardon polic accus mayor death new york time',
     'fda found manipul media favor big pharma',
     'jay h lehr begin end epa',
     'imahdi arriv satan practic leader',
     'ag session dem sen harri abl rush fast make nervou breitbart',
     'u n envoy say u still back palestinian state new york time',
     'trump happen',
     'japan vote strengthen shinzo abe goal chang constitut new york time',
     'trump camp caught cam brag voter suppress women black video',
     'chelsea man ask obama cut sentenc time serv new york time',
     'red sox broadcast jerri remi think foreign player use translat',
     'shi ite militia join iraq mosul attack',
     'escap reign super bowl commerci polit prove inescap new york time',
     'naval brief novemb th ledahu',
     'scandal video footag anonym expos huma hillari',
     'jar new level confront conflict hit washington new york time',
     'clinton transmit classifi info lawyer',
     'hillari russian hack guid american might trump breitbart',
     'gener elect campaign suspend wake manchest suicid bomb',
     'grand slam father son film smash hit famili',
     'venezuela econom crisi mean left fail',
     'herd stamped wild boar kill three islam state jihadist breitbart',
     'venu mar believ gender often wrong new york time',
     'obama pardon list hotel magnat own studio new york time',
     'suspect captur ambush style kill two iowa cop',
     'megyn sic kelli gowdi triumphantli comment hillari case reopen',
     'famili friendli polici friendliest male professor new york time',
     'ringo starr anuncia que deja lo beatl',
     'war less immin clinton defeat',
     'mi chief present russia grow threat british interest',
     'washington state upend trump travel ban new york time',
     'protect swerv polic chief caught speed get laugh ticket video',
     'fake news trump hit mexico explod',
     'go wikileak',
     'former cia director blame millenni wikileak document breitbart',
     'tell stori slaveri new york time',
     'live wire dutch elect high turnout expect controversi turkish mosqu poll centr',
     'shave mean freedom omar',
     'bombshel leak email expos muslim got obama administr',
     'nativ american part ten lost tribe jewish peopl',
     'blm rapper bill alleg son nasti surpris hillari',
     'kevin durant join golden state warrior new york time',
     'indian call center becom major center defraud american breitbart',
     'congress attorney gener lynch plead fifth secret iran ransom payment',
     'lo angel time editori lose trump narcissist demagogu breitbart',
     'swirl untruth falsehood call lie lie new york time',
     'hillari clinton lead donald trump new hampshir florida poll show new york time',
     'gene wilder huma abedin donald trump monday even brief new york time',
     'watch blimp crash catch fire us open breitbart',
     'toddler loos gun car mother die new york time',
     'asian american actor fight visibl ignor new york time',
     'mexican feel environ',
     'insid donald trump last stand anxiou nomine seek assur new york time',
     'park servic name divers nation landmark new york time',
     'weiner rise white hous alli disgrac ex congressman hire breitbart',
     'trump lose grab musket former congressman readi go full revolut',
     'decod north korea claim success nuclear test new york time',
     'bill belichick wit list testifi aaron hernandez doubl murder case breitbart',
     'l mayor silent citi rise violent crime breitbart',
     'theresa may new british prime minist give bori johnson key post new york time',
     'donald trump hillari clinton iphon wednesday even brief new york time',
     'usa kill million peopl victim nation sinc world war ii',
     'polic offic face backlash hillari photo',
     'shakespear remain authent bore',
     'georgia candid jon ossoff film firm financ facebook fact check funder',
     'effort defeat isi u iran imped one anoth new york time',
     'rick put bullet trump wilson potu support see scalis shoot bless breitbart',
     'western intellig agenc run al qaeda camp north africa',
     'trump tell plan parenthood fund stay abort goe new york time',
     'comment hillari clinton gun control agenda expos wikileak email braindiseasecalledliber',
     'barack obama delay suspend elect hillari forc new fbi email investig',
     'donald trump incit feud g p candid flee shadow new york time',
     'germani student forc chant allahu akbar punish refus trip mosqu',
     'meteor space junk rocket mysteri flash hit siberia',
     'trump nation secur advis call russian envoy day sanction impos new york time',
     'project verita implic democrat oper claim credit romney video',
     'laid american requir zip lip way grow bolder new york time',
     ...]




```python
len(corpus)
```




    18285




```python

```


```python
## Applying Countvectorizer
# Creating the Bag of Words model

from sklearn.feature_extraction.text import CountVectorizer
```


```python
## max_features=5000, it means I just need top 5000 features 
#example ABC News is basically 2 words,so in ngram,i have Given (1,3),so it will take the combination of 1 word,then 2 words 
#then 3 words

cv=CountVectorizer(max_features=5000,ngram_range=(1,3))
```


```python
X=cv.fit_transform(corpus).toarray()
```


```python
X.shape
#ie we get 5000 features now
```




    (18285, 5000)




```python
X
```




    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=int64)




```python

```


```python
cv.get_feature_names()[0:20]
```




    ['abandon',
     'abc',
     'abc news',
     'abduct',
     'abe',
     'abedin',
     'abl',
     'abort',
     'abroad',
     'absolut',
     'abstain',
     'absurd',
     'abus',
     'abus new',
     'abus new york',
     'academi',
     'accept',
     'access',
     'access pipelin',
     'access pipelin protest']




```python
messages.columns
```




    Index(['title', 'author', 'text', 'label'], dtype='object')




```python
y=messages['label']
```


```python

```


```python
## Divide the dataset into Train and Test
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25, random_state=42)
```


```python
X_test
```




    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=int64)




```python
X_test.shape
```




    (4572, 5000)




```python

```

###  MultinomialNB Algo


```python
#this algo works well with text data

from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()
```


```python
classifier.fit(X_train,y_train)
```




    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)




```python
pred=classifier.predict(X_test)
pred
```




    array(['1', '0', '0', ..., '0', '0', '0'], dtype='<U1')




```python

```


```python
from sklearn import metrics
```


```python
metrics.accuracy_score(y_test,pred)
```




    0.8904199475065617




```python
cm=metrics.confusion_matrix(y_test,pred)
cm
```




    array([[2301,  299],
           [ 202, 1770]], dtype=int64)




```python
import matplotlib.pyplot as plt
import numpy as np
```


```python
### make your confusion amtrix more user-friendly

plt.imshow(cm,interpolation='nearest',cmap=plt.cm.Blues)
plt.colorbar()
plt.title('Confusion Matrix')
labels=['positive','negative']
tick_marks=np.arange(len(labels))
plt.xticks(tick_marks,labels)
plt.yticks(tick_marks,labels)
```




    ([<matplotlib.axis.YTick at 0xf370f1be48>,
      <matplotlib.axis.YTick at 0xf3704681c8>],
     [Text(0, 0, 'positive'), Text(0, 0, 'negative')])




    
![png](NLP_fake_news_deploy_files/NLP_fake_news_deploy_65_1.png)
    



```python
labels=['positive','negative']
np.arange(len(labels))
```




    array([0, 1])




```python
def plot_confusion_matrix(cm):
    plt.imshow(cm,interpolation='nearest',cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title('Confusion Matrix')
    labels=['positive','negative']
    tick_marks=np.arange(len(labels))
    plt.xticks(tick_marks,labels)
    plt.yticks(tick_marks,labels)
```


```python
plot_confusion_matrix(cm)
```


    
![png](NLP_fake_news_deploy_files/NLP_fake_news_deploy_68_0.png)
    



```python

```

### Passive Aggressive Classifier Algorithm


```python
#this algo works well with text data and is basica0lly used for text data
```


```python
from sklearn.linear_model import PassiveAggressiveClassifier
```


```python
linear_clf=PassiveAggressiveClassifier()
```


```python
linear_clf.fit(X_train,y_train)
```




    PassiveAggressiveClassifier(C=1.0, average=False, class_weight=None,
                                early_stopping=False, fit_intercept=True,
                                loss='hinge', max_iter=1000, n_iter_no_change=5,
                                n_jobs=None, random_state=None, shuffle=True,
                                tol=0.001, validation_fraction=0.1, verbose=0,
                                warm_start=False)




```python
predictions=linear_clf.predict(X_test)
```


```python
metrics.accuracy_score(y_test,predictions)
```




    0.9103237095363079




```python
cm2=metrics.confusion_matrix(y_test,predictions)
cm2
```




    array([[2363,  237],
           [ 173, 1799]], dtype=int64)




```python
plot_confusion_matrix(cm2)
```


    
![png](NLP_fake_news_deploy_files/NLP_fake_news_deploy_78_0.png)
    



```python

```


```python
## Get Features names
#to detect which fake and which is most real word

feature_names=cv.get_feature_names()
```


```python

```


```python
#most negative value is most fake word,if we go towards lower value in -ve,ie we have most fake value
classifier.coef_[0]
```




    array([ -8.86060051,  -8.60928608,  -9.19707274, ..., -10.80651066,
            -8.72706912,  -9.4202163 ])




```python

```


```python
### Most 20 real values
sorted(zip(classifier.coef_[0],feature_names),reverse=True)[0:20]
```




    [(-3.9648951809317863, 'trump'),
     (-4.272721819476034, 'hillari'),
     (-4.368759007672977, 'clinton'),
     (-4.861090048802803, 'elect'),
     (-5.219261999009128, 'new'),
     (-5.230561554263062, 'comment'),
     (-5.269176390390841, 'video'),
     (-5.355472203843678, 'war'),
     (-5.372788653855138, 'hillari clinton'),
     (-5.394864605554338, 'us'),
     (-5.412883111057016, 'fbi'),
     (-5.483500678270969, 'vote'),
     (-5.483500678270969, 'email'),
     (-5.559486585248892, 'obama'),
     (-5.570068694579429, 'world'),
     (-5.718914322176994, 'donald'),
     (-5.743915624382411, 'donald trump'),
     (-5.8229040357010415, 'russia'),
     (-5.864868234800074, 'presid'),
     (-5.872036724278686, 'america')]




```python

```


```python

```


```python

```


```python

```
