import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler


# load data
parts = pd.read_csv('Data/parts.csv')
sets = pd.read_csv('Data/sets.csv')
colors = pd.read_csv('Data/colors.csv')
inventories = pd.read_csv('Data/inventories.csv')
minifigs = pd.read_csv('Data/inventory_minifigs.csv')

##### basic analysis #####

# part analysis
most_common_parts = parts['part_num'].value_counts().idxmax()
print(f"The most common part is: {most_common_parts}")

# set analysis
sets['num_parts'].describe()
print(f"The mean part number of a set is: {sets['num_parts'].mean()}")

# color analysis
most_common_color = colors['rgb'].value_counts().idxmax()
print(f"The most common colour is: {most_common_color}")

# inventory analysis
most_unique_inventory = inventories['set_num'].value_counts().idxmin()
print(f"The set with the most unique inventorylements is: {most_unique_inventory}")


##### analyse how many sets were created over the years #####

sets['year'] = pd.to_numeric(sets['year'], errors='coerce')

# calculate number of sets per year
sets_per_year = sets.groupby('year').size()

# plot
plt.figure(figsize=(10, 5))
plt.plot(sets_per_year)
plt.xlabel('Year')
plt.ylabel('Number of sets')
plt.title('Number of sets per year')
plt.show()


##### analyse the number of parts per set per years #####

sets['year'] = pd.to_numeric(sets['year'], errors='coerce')

# number of parts per set per year
average_parts_per_year = sets.groupby('year')['num_parts'].mean()
plt.figure(figsize=(10, 5))
plt.plot(average_parts_per_year)
plt.xlabel('Year')
plt.ylabel('Mean number of parts per set')
plt.title('Number of parts per set per year')
plt.show()


##### analyse the number of sets for each theme over the years #####

sets['year'] = pd.to_numeric(sets['year'], errors='coerce')

# group the data after year and theme and count the number of sets
theme_counts_per_year = sets.groupby(['year', 'theme_id']).size().reset_index(name='counts')

# for each theme, plot the number of sets over the years
themes = theme_counts_per_year['theme_id'].unique()
theme = pd.read_csv('Data/themes.csv')

for id in themes:
    data = theme_counts_per_year[theme_counts_per_year['theme_id'] == id]
    plt.plot(data['year'], data['counts'], label=theme.loc[theme['id'] == id, 'name'].values[0])

plt.xlabel('Year')
plt.ylabel('Number of sets')
plt.title('Number of sets for each theme over the years')
plt.legend()
plt.show()


##### Linear regression for the number of parts for each set #####

# define the dependend and independend variable
X = sets['year'].values.reshape(-1,1)
y = sets['num_parts'].values.reshape(-1,1)

# split data in training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# create and train model
regressor = LinearRegression()  
regressor.fit(X_train, y_train)

# make prediction on test data
y_pred = regressor.predict(X_test)

# prediction for fucutre year
future_year = [[2025]]
predicted_num_parts = regressor.predict(future_year)
print(f"Predicted number of parts in {future_year[0][0]} is {predicted_num_parts[0][0]}")

# create a plot
plt.scatter(X_test, y_test, color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.scatter(future_year, predicted_num_parts, color='blue')
plt.xlabel('Year')
plt.ylabel('Number of parts')
plt.title('Linear Regression: Number of parts for sets per year')
plt.show()

##### analyse the data with the colour ####
themes = pd.read_csv('Data/themes.csv')

# merge data
sets_with_themes = pd.merge(sets, themes, left_on='theme_id', right_on='id')
full_data = pd.merge(sets_with_themes, colors, left_on='id', right_on='id')

# group data for colour and theme
grouped_data = full_data.groupby(['name_x', 'rgb']).size().reset_index(name='counts')
print(grouped_data)

# show data for specific theme
theme_name = 'Star Wars'  
theme_data = grouped_data[grouped_data['name_x'] == theme_name]

# create barplot
plt.bar(theme_data['rgb'], theme_data['counts'])
plt.title(f'colour for {theme_name}')
plt.xlabel('colour')
plt.ylabel('number')
plt.show()


##### PCA for theme id #####
features = ['num_parts', 'year']  
x = sets.loc[:, features].values

# standardize feature
x = StandardScaler().fit_transform(x)

# perfom PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

# create dataframe with principal components
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

# add theme information
finalDf = pd.concat([principalDf, sets[['theme_id']]], axis=1)
print(finalDf)

# create scatter plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('principal component 1', fontsize=15)
ax.set_ylabel('principal component 2', fontsize=15)
ax.set_title('2-component PCA', fontsize=20)

# choose example theme
theme_id = 1  
colors = ['r', 'b']
for theme, color in zip([theme_id, 'other'], colors):
    indicesToKeep = finalDf['theme_id'] == theme if theme != 'other' else finalDf['theme_id'] != theme_id
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c=color
               , s=50)
ax.legend(['Theme', 'Else'])
ax.grid()

plt.show()
