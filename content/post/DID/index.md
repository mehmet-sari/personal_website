```python
from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')
```




<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>



## Empirical Practice of Difference-in-Differences 

### Callaway and Sant'Anna(2020)

To get more familiar with the recent methodological studies in dif-in-diff(DID) methods, I attempt to replicate [Callaway and Sant'Anna(2020)](https://pedrohcgs.github.io/files/Callaway_SantAnna_2020.pdf)(Hereafter CS). CS proposes a DID estimator for research designs with multiple time periods and units particularly when units are being treated at staggered time periods. As it is well-known causal inference method, a standard did design includes only two time periods and two units. For example please see the famous example of Card and Kruger Wage Study(1994) in 2x2 table below.

   |      | Pre | Post | Difference |
   | --- | --- | --- | --- |
   |New Jersey     |  20.44 | 21.03 | 0.59 |
   |Pennsylvania     |  23.33 | 21.17 | - 2.16 |
   |**Difference**      | - 2.89 |  -0.14 | 2.75 |

A standard 2x2 DID method is not applicable for cases such as Medicaid expansion. The Affordable Care Act (ACA) mandated to expand the elibility of Medicaid so the number of low-income Americans with health insurance increases. When Medicaid expansion took effect in 2014, 27 seven states adopted the expansion that year. As of 2021 the total number of states with Medicaid expansion is 39. Only 12 states have not adopted Medicaid expansion yet so far. Different expansion year makes the data perfect for a DID analysis with multiple time periods and groups. 

I apply CS method in this post using Medicaid expansion data which I extracted from American Community Survey (ACS) and [Kaiser Family Foundation website](https://www.kff.org/medicaid/issue-brief/status-of-state-medicaid-expansion-decisions-interactive-map/). The data includes median income level, uninsured and unemployment rate at the county level from 2011 to 2019. For the sake of transparency, I am sharing the data at [the this link](https://raw.githubusercontent.com/mehmet-sari/Exercises_Applications/main/medicaidexpansion.csv).


```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

#pd.set_option('display.max_rows', 500)
%load_ext rpy2.ipython
```

#### Summary of the Data
The next two outputs show the data and summary statistics of uninsured rate, median income level, and unemployment rate.


```python
df = pd.read_csv("output.csv", encoding='latin-1')
df['AdoptedYear'] = df['AdoptedYear'].replace([2020, 2021],0)
df.head(10)
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
      <th>State</th>
      <th>id</th>
      <th>unemployment</th>
      <th>median_income</th>
      <th>uninsured</th>
      <th>year</th>
      <th>County</th>
      <th>MedicaidStatus</th>
      <th>AdoptedYear</th>
      <th>SID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>0500000US01003</td>
      <td>5.6</td>
      <td>50900</td>
      <td>12.0</td>
      <td>2011</td>
      <td>Baldwin County</td>
      <td>Not Adopted</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alabama</td>
      <td>0500000US01015</td>
      <td>7.5</td>
      <td>39037</td>
      <td>15.6</td>
      <td>2011</td>
      <td>Calhoun County</td>
      <td>Not Adopted</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Alabama</td>
      <td>0500000US01043</td>
      <td>5.5</td>
      <td>40054</td>
      <td>12.6</td>
      <td>2011</td>
      <td>Cullman County</td>
      <td>Not Adopted</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Alabama</td>
      <td>0500000US01049</td>
      <td>7.1</td>
      <td>36541</td>
      <td>19.6</td>
      <td>2011</td>
      <td>DeKalb County</td>
      <td>Not Adopted</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alabama</td>
      <td>0500000US01051</td>
      <td>6.5</td>
      <td>57405</td>
      <td>10.6</td>
      <td>2011</td>
      <td>Elmore County</td>
      <td>Not Adopted</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Alabama</td>
      <td>0500000US01055</td>
      <td>4.8</td>
      <td>33313</td>
      <td>14.4</td>
      <td>2011</td>
      <td>Etowah County</td>
      <td>Not Adopted</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Alabama</td>
      <td>0500000US01069</td>
      <td>5.2</td>
      <td>40336</td>
      <td>13.9</td>
      <td>2011</td>
      <td>Houston County</td>
      <td>Not Adopted</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Alabama</td>
      <td>0500000US01073</td>
      <td>7.2</td>
      <td>41976</td>
      <td>13.3</td>
      <td>2011</td>
      <td>Jefferson County</td>
      <td>Not Adopted</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Alabama</td>
      <td>0500000US01077</td>
      <td>3.1</td>
      <td>40121</td>
      <td>11.0</td>
      <td>2011</td>
      <td>Lauderdale County</td>
      <td>Not Adopted</td>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Alabama</td>
      <td>0500000US01081</td>
      <td>6.4</td>
      <td>42965</td>
      <td>12.6</td>
      <td>2011</td>
      <td>Lee County</td>
      <td>Not Adopted</td>
      <td>0</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('Summary Statistics')

df[["uninsured", "unemployment", "median_income"]].describe()
```

    Summary Statistics
    




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
      <th>uninsured</th>
      <th>unemployment</th>
      <th>median_income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7380.000000</td>
      <td>7341.000000</td>
      <td>7380.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>10.381260</td>
      <td>6.009767</td>
      <td>56981.417344</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.094168</td>
      <td>2.350090</td>
      <td>15740.901841</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.400000</td>
      <td>1.000000</td>
      <td>24945.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.400000</td>
      <td>4.400000</td>
      <td>45894.750000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9.600000</td>
      <td>5.600000</td>
      <td>53360.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>13.500000</td>
      <td>7.200000</td>
      <td>64381.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>37.900000</td>
      <td>21.700000</td>
      <td>151800.000000</td>
    </tr>
  </tbody>
</table>
</div>



For the first step, I will look at the impact of Medicaid expansion on uninsured rate at the county level without pre-treatment covariates.

Briefly, the identification CS method relies on parallel trends assumption after conditioning on pre-treatment covariates. Once there is no covariates in the analysis and the comparison group is 'never treated' group, CS proposes the following causal parameter of interest (equation 2.8 in their paper).

\begin{gather*}
ATT^{nev}_{unc} (g, t) = E[Y_{t} - Y_{g  - 1}|G_{g} = 1] - E[Y_{t} - Y_{g - 1}|C = 1]
\end{gather*}  

$ATT^{nev}_{unc} (g, t)$ compares the outcome evolution of group $g$, which are the units first treated at the time $g$, between the period of time $t$ and $g$-1 with the outcome evolution of comparison group, which are never treated group, between the same time period. The outcome evolution of comparison group provides the parallel trends in case the treated group would not experienced the effect of treatment. The parameter calculates the treatment effect for group $g$ at time $t$ and it is called group-time average treatment effect.

To give an example, let's calculate $ATT_{g, t}$ for group 2014 at time 2015 using the Medicaid expansion. Group of 2014 presents the states that adopted Medicaid expansion in 2014.

\begin{gather*}
ATT(2014,2015) = (Y_{2014,2015} - Y_{2014,2013}) - (Y_{C,2015} - Y_{C, 2013})
\end{gather*} 

Let's show this in 2x2 table:

   |      | Pre(2013) | Post(2015) | Difference |
   | --- | --- | --- | --- |
   |Adopted     |  Y(2014,2013) | Y(2014,2015) |  |
   |Non-Adopted     |  Y(C, 2013) | Y(C,2015)  | |
   |**Difference**      |  |   |    

The ATT parameter calculates the treatment effect by turning the data to a standard 2x2 DID setup with 2 groups (group of 2014 and never adopted group) and 2 time periods (2013 and 2015). Once ATT for group of 2014 is calculated for every year, it is possible to aggregate treatment effects in several ways to understand the treatment effect heterogeneity between groups.

Now I need to calculate ATT for each year for each group. For that, I need to create dummy variables for each group and treatment years for counties with Medicaid expansion.


```python
## Create dummy variables for each cohort/group of treated counties.
dummies = pd.get_dummies(df, prefix='g', columns=['AdoptedYear'], drop_first= True)
df1 = pd.concat([dummies, df['AdoptedYear']],axis=1)

#Crate a dummy variable for treatment years for counties with medicaid expansion.
df1['treat'] = np.where((df1['AdoptedYear'] <= df1['year']), 1, 0)
df1.loc[df1['AdoptedYear'] == 0, 'treat'] = 0
```


```python
## ATT estimation attempt to replicate what did package in R produces.

# function for dif-in-diff calculation.

def att(df,g,t):
    
    y11 = df[(df['MedicaidStatus'] == "Adopted") 
             & (df['AdoptedYear'] == g)
             & (df['year'] == t)]['uninsured'].mean()
             
    y10 = df[(df['MedicaidStatus'] == "Adopted") 
             & (df['AdoptedYear'] == g)
             & (df['year'] == g - 1)]['uninsured'].mean()
             
    y01 = df[(df['MedicaidStatus'] == "Not Adopted") 
             & (df['year'] == t)]['uninsured'].mean()    

    y00 = df[(df['MedicaidStatus'] == "Not Adopted") 
             & (df['year'] == g - 1)]['uninsured'].mean()
    
    did = (y11 - y10) - (y01 - y00)
    
    return did
## 
result = []
groups = df1['AdoptedYear'].unique()
times = df1['year'].unique()

for group in groups:
    if group > 0:
        for time in times:
            x = att(df, group, time)
            result.append((group,time,x))
            
att_estimate = pd.DataFrame(result)
att_estimate.columns = ['group','time', 'att']
att_estimate.sort_values(['group', 'time'])
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
      <th>group</th>
      <th>time</th>
      <th>att</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>2014</td>
      <td>2011</td>
      <td>-0.120971</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2014</td>
      <td>2012</td>
      <td>-0.037960</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2014</td>
      <td>2013</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2014</td>
      <td>2014</td>
      <td>-0.726021</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2014</td>
      <td>2015</td>
      <td>-0.902026</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2014</td>
      <td>2016</td>
      <td>-0.951459</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2014</td>
      <td>2017</td>
      <td>-1.284129</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2014</td>
      <td>2018</td>
      <td>-1.246134</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2014</td>
      <td>2019</td>
      <td>-1.275390</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2015</td>
      <td>2011</td>
      <td>-1.058772</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>2012</td>
      <td>-0.912148</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015</td>
      <td>2013</td>
      <td>-0.969544</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015</td>
      <td>2014</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015</td>
      <td>2015</td>
      <td>-0.315004</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015</td>
      <td>2016</td>
      <td>-0.407644</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2015</td>
      <td>2017</td>
      <td>-1.212228</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2015</td>
      <td>2018</td>
      <td>-1.049331</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2015</td>
      <td>2019</td>
      <td>-1.254695</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2016</td>
      <td>2011</td>
      <td>1.380633</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2016</td>
      <td>2012</td>
      <td>1.178327</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2016</td>
      <td>2013</td>
      <td>-0.226087</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2016</td>
      <td>2014</td>
      <td>0.675362</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2016</td>
      <td>2015</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2016</td>
      <td>2016</td>
      <td>-1.024033</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2016</td>
      <td>2017</td>
      <td>-3.051785</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2016</td>
      <td>2018</td>
      <td>-3.697584</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2016</td>
      <td>2019</td>
      <td>-3.071063</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2018</td>
      <td>2011</td>
      <td>-2.328676</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2018</td>
      <td>2012</td>
      <td>-1.707346</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2018</td>
      <td>2013</td>
      <td>-1.567781</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2018</td>
      <td>2014</td>
      <td>-0.083433</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2018</td>
      <td>2015</td>
      <td>0.206857</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2018</td>
      <td>2016</td>
      <td>0.111374</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2018</td>
      <td>2017</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>43</th>
      <td>2018</td>
      <td>2018</td>
      <td>-0.261886</td>
    </tr>
    <tr>
      <th>44</th>
      <td>2018</td>
      <td>2019</td>
      <td>-1.662467</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2019</td>
      <td>2011</td>
      <td>-1.551790</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2019</td>
      <td>2012</td>
      <td>-1.065460</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2019</td>
      <td>2013</td>
      <td>-0.645895</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2019</td>
      <td>2014</td>
      <td>-0.449880</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2019</td>
      <td>2015</td>
      <td>-0.012924</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2019</td>
      <td>2016</td>
      <td>-0.671740</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2019</td>
      <td>2017</td>
      <td>-0.853114</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2019</td>
      <td>2018</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2019</td>
      <td>2019</td>
      <td>-0.728915</td>
    </tr>
  </tbody>
</table>
</div>



Now I calculated group-time treatment affects for each group and time pair and I can calculate the aggregate ATT for each group. Before that I need to point that ATT(g,t) is zero once $t$<$g$ is. It means that ATT for time 2011, 2012, and 2013 are not considered in the aggregate ATT(g) for group of 2014. 


```python
att_estimate[att_estimate['time'] >= att_estimate['group']].groupby(['group'])['att'].mean()
```




    group
    2014   -1.064193
    2015   -0.847780
    2016   -2.711116
    2018   -0.962177
    2019   -0.728915
    Name: att, dtype: float64



So far, I calculate group-time treatment effects and aggregate ATTs for each group. Now, I would like to check my results with R package's result which is provided by CS. 


```r
%%R
options(warn = -1)
install.packages("did", repos='http://cran.us.r-project.org', quiet=TRUE)
install.packages("ggplot2", repos='http://cran.us.r-project.org', quiet=TRUE)
install.packages("dplyr", repos='http://cran.us.r-project.org', quiet=TRUE)
library(did)
library(ggplot2)
library(dplyr)
df_medicaid <- read.csv('https://raw.githubusercontent.com/mehmet-sari/Exercises_Applications/main/medicaidexpansion.csv')
att <- att_gt(yname = "uninsured",  
              tname = "year",  
              idname = "SID", 
              gname = "AdoptedYear",  
              data = df_medicaid,  
              xformla = NULL,   
              est_method = "dr", 
              control_group = "nevertreated", 
              bstrap = TRUE,  
              biters = 1000,  
              print_details = FALSE)  
summary(att)
```

    R[write to console]: Warning:
    R[write to console]:  package 'did' is in use and will not be installed
    
    R[write to console]: Warning:
    R[write to console]:  package 'ggplot2' is in use and will not be installed
    
    R[write to console]: Warning:
    R[write to console]:  package 'dplyr' is in use and will not be installed
    
    

    
    Call:
    att_gt(yname = "uninsured", tname = "year", idname = "SID", gname = "AdoptedYear", 
        xformla = NULL, data = df_medicaid, control_group = "nevertreated", 
        bstrap = TRUE, biters = 1000, est_method = "dr", print_details = FALSE)
    
    Reference: Callaway, Brantly and Pedro H.C. Sant'Anna.  "Difference-in-Differences with Multiple Time Periods." Forthcoming at the Journal of Econometrics <https://arxiv.org/abs/1803.09015>, 2020. 
    
    Group-Time Average Treatment Effects:
     Group Time ATT(g,t) Std. Error [95% Simult.  Conf. Band]  
      2014 2012   0.1063     0.1603       -0.3665      0.5792  
      2014 2013   0.0539     0.1415       -0.3633      0.4711  
      2014 2014  -0.7857     0.1583       -1.2524     -0.3190 *
      2014 2015  -0.9508     0.1964       -1.5302     -0.3715 *
      2014 2016  -1.0373     0.2066       -1.6464     -0.4281 *
      2014 2017  -1.3097     0.1877       -1.8632     -0.7562 *
      2014 2018  -1.3181     0.2000       -1.9078     -0.7283 *
      2014 2019  -1.3658     0.2178       -2.0081     -0.7234 *
      2015 2012   0.1453     0.2435       -0.5728      0.8634  
      2015 2013  -0.0526     0.1913       -0.6168      0.5117  
      2015 2014   0.9079     0.2032        0.3086      1.5072 *
      2015 2015  -0.3088     0.2014       -0.9027      0.2850  
      2015 2016  -0.4149     0.2486       -1.1480      0.3182  
      2015 2017  -1.1440     0.2427       -1.8597     -0.4284 *
      2015 2018  -1.0426     0.2406       -1.7522     -0.3331 *
      2015 2019  -1.2553     0.2419       -1.9686     -0.5421 *
      2016 2012  -0.2036     0.4881       -1.6431      1.2358  
      2016 2013  -1.2423     0.5726       -2.9309      0.4464  
      2016 2014   0.8321     0.3364       -0.1600      1.8241  
      2016 2015  -0.6599     0.4132       -1.8784      0.5586  
      2016 2016  -0.9904     0.4742       -2.3888      0.4079  
      2016 2017  -3.0455     0.5760       -4.7441     -1.3469 *
      2016 2018  -3.7934     0.5752       -5.4897     -2.0971 *
      2016 2019  -3.1076     0.5142       -4.6240     -1.5912 *
      2018 2012   0.6200     0.3342       -0.3656      1.6056  
      2018 2013   0.1444     0.3131       -0.7790      1.0678  
      2018 2014   1.4227     0.3311        0.4462      2.3991 *
      2018 2015   0.2965     0.3126       -0.6253      1.2182  
      2018 2016  -0.1089     0.3739       -1.2116      0.9937  
      2018 2017  -0.0699     0.4110       -1.2820      1.1422  
      2018 2018  -0.3106     0.3921       -1.4669      0.8457  
      2018 2019  -1.7215     0.3760       -2.8304     -0.6126 *
      2019 2012   0.4850     0.5806       -1.2273      2.1973  
      2019 2013   0.4244     0.7603       -1.8178      2.6666  
      2019 2014   0.1344     0.6859       -1.8884      2.1571  
      2019 2015   0.4431     0.6133       -1.3654      2.2517  
      2019 2016  -0.6723     0.6531       -2.5982      1.2537  
      2019 2017  -0.1399     0.5593       -1.7891      1.5093  
      2019 2018   0.8044     0.5286       -0.7546      2.3633  
      2019 2019  -0.7392     0.4441       -2.0489      0.5705  
    ---
    Signif. codes: `*' confidence band does not cover 0
    
    P-value for pre-test of parallel trends assumption:  0
    Control Group:  Never Treated,  Anticipation Periods:  0
    Estimation Method:  Doubly Robust
    

It seems that the results from R package for CS method are not equal to the results I calculated manually. Numbers are close, trends are similar but somehow there is a slight difference between the results. Lastly I shall check the aggregate ATT results.


```r
%%R 
att <-aggte(att, type = "group")
summary(att)
```

    
    Call:
    aggte(MP = att, type = "group")
    
    Reference: Callaway, Brantly and Pedro H.C. Sant'Anna.  "Difference-in-Differences with Multiple Time Periods." Forthcoming at the Journal of Econometrics <https://arxiv.org/abs/1803.09015>, 2020. 
    
    
    Overall ATT:  
         ATT Std. Error     [95%  Conf. Int.]  
     -1.1425     0.1396   -1.4161     -0.8688 *
    
    
    Group Effects:
     group     ATT Std. Error [95% Simult.  Conf. Band]  
      2014 -1.1279     0.1708       -1.5501     -0.7056 *
      2015 -0.8332     0.1997       -1.3268     -0.3395 *
      2016 -2.7342     0.4132       -3.7557     -1.7127 *
      2018 -1.0161     0.3688       -1.9277     -0.1044 *
      2019 -0.7392     0.4578       -1.8708      0.3924  
    ---
    Signif. codes: `*' confidence band does not cover 0
    
    Control Group:  Never Treated,  Anticipation Periods:  0
    Estimation Method:  Doubly Robust
    

The table below shows the results from my code and the results from R package side by side. Again the R package and my code produce similar but not exactly same results. 

|My Code |       | R Package |    |
| :--- | ---- | :--- | ---- |
group   |  ATT   |  group |  ATT |
2014 |  -1.064193 | 2014 | -1.1279 | 
2015 |  -0.847780 | 2015 | -0.8332 | 
2016 |  -2.711116 | 2016 | -2.7342 | 
2018 |  -0.962177 | 2018 | -1.0161 | 
2019 |  -0.728915 | 2019 | -0.7392 | 

In the next step, I will estimate the treatment effect of Medicaid expansion with covariates. For that, I will use the following parameter which includes propensity score estimation. 

\begin{gather*}
ATT(g,t)=[(\frac{G_{g}}{E\big[G_{g}\big]} - \frac{\frac{p_{g}(X)C}{1-pg(x}}{E\big[\frac{p_{g}(X)C}{1-p_{g}(X)}\big]})(Y_{t}-Y_{g-1})
\end{gather*}


```python

```
