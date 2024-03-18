# data-visualization-cheat-sheet

## Visualization of the distribution of a numerical value across multiple categories
```python
categories = dataset['category'].unique()
# Divide the categories into groups of five
data_subset_1 = dataset[dataset['category'].isin(categories[:5])]
data_subset_2 = dataset[dataset['category'].isin(categories[5:])]
# Columns of interest with numerical values
columns = ['calories', 'carbohydrate', 'sugar']
# Visualize the distribution of each interesting column across the categories 
for column in columns:
    fig, axs = plt.subplots(1,2, figsize=(12,6), sharey = True)
    sns.kdeplot(data= data_subset_1, x = column, hue ='category', ax=axs[0], fill=True)
    sns.kdeplot(data= data_subset_2, x = column, hue ='category', ax=axs[1], fill=True)
    plt.suptitle(f'Distribution of {column} by category', y= 0.95, size=16)
    plt.show()
```

<div align="center">
<img src="images/distribution of num values across multiple categories .png", alt="distribution of num values across multiple categories" height="350" width="750"></img>
</div>
<hr />


## Change the scale for better visualization
```python
# Changing protein column to a square root scale to better visualization
fig, axs = plt.subplots(1,2, figsize=(12,6), sharey = True)
sns.kdeplot(data= data_subset_1, x = np.sqrt(data_subset_1['protein']), hue ='category', ax=axs[0], fill=True)
sns.kdeplot(data= data_subset_2, x =  np.sqrt(data_subset_2['protein']), hue ='category', ax=axs[1], fill=True)
plt.suptitle('Distribution of protein by category', y= 0.95, size=16)
plt.show()
```
<div align="center">
<img src="images/change of scale.png", alt="change of scale" height="350" width="750"></img>
</div>
<hr />
