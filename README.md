<div align="center"><img src="VidyaLogo.png"/></div>

Data fetched from Kaggle: https://www.kaggle.com/datasets/gregorut/videogamesales


```python
#Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
%matplotlib inline
```


```python
#Loading data
data = pd.read_csv("VidyaNoRat.csv")
DataRaw = data.shape[0]

data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 16598 entries, 0 to 16597
    Data columns (total 11 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   Rank          16598 non-null  int64  
     1   Name          16598 non-null  object 
     2   Platform      16598 non-null  object 
     3   Year          16327 non-null  float64
     4   Genre         16598 non-null  object 
     5   Publisher     16540 non-null  object 
     6   NA_Sales      16598 non-null  float64
     7   EU_Sales      16598 non-null  float64
     8   JP_Sales      16598 non-null  float64
     9   Other_Sales   16598 non-null  float64
     10  Global_Sales  16598 non-null  float64
    dtypes: float64(6), int64(1), object(4)
    memory usage: 1.4+ MB
    


```python
#Processing data to remove NA and invalid values
data = pd.DataFrame(data)
data = data.drop(data[data['Year'] > 2016].index)
data = data.dropna(axis=0)
data = data.drop("Rank", axis=1)
data['Year'] = data['Year'].astype('int64')

DataProcessed = data.shape[0]
Diff = DataRaw - DataProcessed

print(
    f'We removed {Diff} records from data by processing, leaving us with {DataProcessed} records'
)
```

    We removed 311 records from data by processing, leaving us with 16287 records
    


```python
#Selecting 7th generation consoles
data7 = data[data['Platform'].isin(['X360', 'PS3', 'Wii', 'PSP', 'DS'])]
data7 = data7[data7['Year'] >= 2005]

data7.head()
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
      <th>Name</th>
      <th>Platform</th>
      <th>Year</th>
      <th>Genre</th>
      <th>Publisher</th>
      <th>NA_Sales</th>
      <th>EU_Sales</th>
      <th>JP_Sales</th>
      <th>Other_Sales</th>
      <th>Global_Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wii Sports</td>
      <td>Wii</td>
      <td>2006</td>
      <td>Sports</td>
      <td>Nintendo</td>
      <td>41.49</td>
      <td>29.02</td>
      <td>3.77</td>
      <td>8.46</td>
      <td>82.74</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mario Kart Wii</td>
      <td>Wii</td>
      <td>2008</td>
      <td>Racing</td>
      <td>Nintendo</td>
      <td>15.85</td>
      <td>12.88</td>
      <td>3.79</td>
      <td>3.31</td>
      <td>35.82</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Wii Sports Resort</td>
      <td>Wii</td>
      <td>2009</td>
      <td>Sports</td>
      <td>Nintendo</td>
      <td>15.75</td>
      <td>11.01</td>
      <td>3.28</td>
      <td>2.96</td>
      <td>33.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>New Super Mario Bros.</td>
      <td>DS</td>
      <td>2006</td>
      <td>Platform</td>
      <td>Nintendo</td>
      <td>11.38</td>
      <td>9.23</td>
      <td>6.50</td>
      <td>2.90</td>
      <td>30.01</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Wii Play</td>
      <td>Wii</td>
      <td>2006</td>
      <td>Misc</td>
      <td>Nintendo</td>
      <td>14.03</td>
      <td>9.20</td>
      <td>2.93</td>
      <td>2.85</td>
      <td>29.02</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set_theme(style='darkgrid')
sns.set(font_scale=1.8)

plt.subplots(figsize=(15, 9))
plt.xticks(data7['Year'].unique())
plt.title('Game releases across generation')
plt.ylabel('Games released')

sns.histplot(data=data7,
             x='Year',
             hue='Platform',
             multiple='stack',
             shrink=0.5,
             discrete=True)

plt.show()
```


    
![png](./Vidya_graphs/output_6_0.png)
    



```python
sns.displot(data=data7,
            x='Year',
            hue='Platform',
            col='Platform',
            col_wrap=2,
            legend=False,
            rug=True).set_ylabels('Games released')

plt.show()
```


    
![png](./Vidya_graphs/output_7_0.png)
    



```python
plt.subplots(figsize=(16, 9))

total_sales = sns.barplot(x='Year',
                          y='Global_Sales',
                          data=data7,
                          estimator=sum,
                          hue='Platform',
                          ci=None)

plt.ylabel('Copies of games sold in millions')
plt.title('Games sold across generation')
plt.legend(loc='upper right')
plt.show()
```


    
![png](./Vidya_graphs/output_8_0.png)
    



```python
sales = data7.groupby('Year')['Global_Sales'].sum()
releases = data7.groupby('Year')['Name'].count()

sal_rel_joint = sns.jointplot(data=data7,
                              x=releases,
                              y=sales,
                              kind='reg',
                              height=12)

sal_rel_joint.set_axis_labels('Games released', 'Games sold in millions')
plt.show()
```


    
![png](./Vidya_graphs/output_9_0.png)
    


With the increase of released games, copies of games sold also raise (not a huge discovery).


```python
plt.subplots(figsize=(16, 9))
sales_order = data7.groupby('Genre')['Global_Sales'].sum().sort_values(
    ascending=False).index.values

genre_sales = sns.barplot(x='Global_Sales',
                          y='Genre',
                          data=data7,
                          estimator=sum,
                          order=sales_order,
                          ci=None)

plt.xlabel('Copies of games sold in millions')
plt.ylabel('Genre')
plt.title('Best selling genre')
plt.show()
```


    
![png](./Vidya_graphs/output_11_0.png)
    



```python
plt.subplots(figsize=(15, 14))
sales_order = data7.groupby('Genre')['Global_Sales'].sum().sort_values(
    ascending=False).index.values

genre_sales = sns.barplot(x='Global_Sales',
                          y='Genre',
                          data=data7,
                          hue='Platform',
                          estimator=sum,
                          order=sales_order,
                          ci=None)

plt.xlabel('Copies of games sold in millions')
plt.ylabel('Genre')
plt.title('Games sold by genre')
plt.show()
```


    
![png](./Vidya_graphs/output_12_0.png)
    



```python
genre_section = data7[[
    'Genre', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'
]]
genre_sale = genre_section.groupby('Genre').sum()

plt.subplots(figsize=(16, 10))
sns.heatmap(genre_sale, annot=True, fmt='g', linewidths=0.4)

plt.title('Genre sales by region')
plt.show()
```


    
![png](./Vidya_graphs/output_13_0.png)
    


Action genre generated the highest profits everywhere except Japan, where Role-Playing games dominated.


```python
plt.subplots(figsize=(16, 4))
sales_order = data7.groupby('Platform')['Global_Sales'].sum().sort_values(
    ascending=False).index.values

platform_sales = sns.barplot(x='Global_Sales',
                             y='Platform',
                             data=data7,
                             estimator=sum,
                             order=sales_order,
                             ci=None)

plt.xlabel('Copies of games sold in millions')
plt.ylabel('Platform')
plt.title('Game sales on platforms')
plt.show()
```


    
![png](./Vidya_graphs/output_15_0.png)
    



```python
genre_section = data7[[
    'Platform', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'
]]
genre_sale = genre_section.groupby('Platform').sum()

plt.subplots(figsize=(16, 10))
sns.heatmap(genre_sale, annot=True, fmt='g', linewidths=0.4)

plt.title('Game sales on platforms by region')
plt.show()
```


    
![png](./Vidya_graphs/output_16_0.png)
    


This time, each of the three distinguished regions had their favorite console, for North America it was Microsoft Xbox 360, for Europe - Sony PlayStation 3 and for Japan - Nintendo DS.


```python
plt.subplots(figsize=(16, 9))
sns.set
sales_order = data7.groupby('Publisher')['Global_Sales'].sum().sort_values(
    ascending=False).index.values

publisher_sales = sns.barplot(x='Global_Sales',
                              y='Publisher',
                              data=data7,
                              estimator=sum,
                              order=sales_order[0:10],
                              ci=None)

plt.xlabel('Copies of games sold in millions')
plt.ylabel('Publisher')
plt.title('Top 10 Publishers by game sales')
plt.show()
```


    
![png](./Vidya_graphs/output_18_0.png)
    



```python
plt.subplots(figsize=(16, 9))

publisher_count = sns.countplot(
    y='Publisher',
    data=data7,
    order=data7.groupby('Publisher')['Name'].count().sort_values(
        ascending=False).iloc[:10].index,
    orient='h')

plt.xlabel('Number of games published')
plt.ylabel('Publisher')
plt.title('Top 10 Publishers by games published')
plt.show()
```


    
![png](./Vidya_graphs/output_19_0.png)
    


Electronic Arts published the most games in the 7th generation, but came in second in terms of sales. Interestingly, Nintendo sold the most games while ranking only 9th in terms of number of games published. Quantity of published games didn't grant highest sales. It is worth noting, that releasing one game on multiple platforms results in multiple entries in counting.


```python
Top_10Publishers = [
    'Electronic Arts', 'Ubisoft', 'Activision', 'Namco Bandai Games',
    'Konami Digital Entertainment', 'THQ', 'Sega',
    'Sony Computer Entertainment', 'Nintendo', 'Take-Two Interactive'
]

Top_Publishers_data = data7[data7['Publisher'].isin(Top_10Publishers)]

Publisher_genres = sns.displot(data=Top_Publishers_data,
                               x='Genre',
                               hue='Genre',
                               col='Publisher',
                               col_wrap=3,
                               legend=True,
                               rug=True,
                               palette='colorblind')

Publisher_genres.set(xticklabels=[], ylabel='Games published')
plt.show()
```


    
![png](./Vidya_graphs/output_21_0.png)
    


Electronic Arts dominated Sports genre, publishing over 200 games. Ubisoft took more of a balanced approach, focusing on as many as three genres: Misc, Simulation and Action. Activision mainly made Action and Shooter games. Namco Bandai Games mainly published Action games, but interestingly they published the most Role-Playing games of all 10 competitors - as a Japanese company they were the main source of Role-Playing games loved by their countrymen.


```python
plt.subplots(figsize=(16, 8))
sales_order = data7.groupby('Name')['Global_Sales'].sum().sort_values(
    ascending=False).index.values

Game_sales = sns.barplot(x='Global_Sales',
                         y='Name',
                         data=data7,
                         estimator=sum,
                         order=sales_order[0:10],
                         ci=None,
                         palette='colorblind')

plt.xlabel('Copies of games sold in millions')
plt.ylabel('Game title')
plt.title('Top 10 Games by sales')
plt.show()
```


    
![png](./Vidya_graphs/output_23_0.png)
    


Wii Sports sold more than twice as many copies as Grand Theft Auto V, achieving the title of best-selling game of the 7th generation of consoles. How did Wii Sports win with GTA V from Rockstar Games, which was the most anticipated game of the generation? Most Nintendo Wii consoles were sold bundled with Wii Sports, the game was designed to showcase the assets of the console - mainly quite fun motion controls.
