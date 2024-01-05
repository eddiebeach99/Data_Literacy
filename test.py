import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from collections import defaultdict
from collections import Counter



# load data
parts = pd.read_csv('Data/parts.csv')
sets = pd.read_csv('Data/sets.csv')
sets = sets[sets["year"] <= 2023]
colors = pd.read_csv('Data/colors.csv')
inventories = pd.read_csv('Data/inventories.csv')
inventory_parts = pd.read_csv('Data/inventory_parts.csv')
minifigs = pd.read_csv('Data/inventory_minifigs.csv')
inventory_sets = pd.read_csv('Data/inventory_sets.csv')

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

##### number of sets for top 10 themes over the years #####

# Merge the sets and themes data
data = pd.merge(sets, theme, left_on='theme_id', right_on='id')

# Convert the year to integer
data['year'] = data['year'].astype(int)

# Filter the data for recent years
recent_data = data[data['year'] > 2000]

# Find the top 10 themes
top_themes = recent_data['name_y'].value_counts().nlargest(10).index
print(top_themes)

# Filter the data for the top 10 themes
filtered_data = recent_data[recent_data['name_y'].isin(top_themes)]

# Group the data by year and theme and count the number of sets
grouped_data = filtered_data.groupby(['year', 'name_y']).size().reset_index(name='counts')

# Plot the number of sets over the years for each theme
plt.figure(figsize=(10, 6))
sns.lineplot(data=grouped_data, x='year', y='counts', hue='name_y')
plt.title('Number of Sets Over the Years for Top 10 Themes after the year 2000')
plt.show()


##### linear regression for the themes #####

# Create a new plot
plt.figure(figsize=(10, 6))

# For each theme
for theme in top_themes:
    # Filter the data for the current theme
    theme_data = grouped_data[grouped_data['name_y'] == theme]
    
    # Get the years (X) and counts (y)
    X = theme_data['year'].values.reshape(-1, 1)
    y = theme_data['counts'].values.reshape(-1, 1)
    
    # Fit the linear regression model
    model = LinearRegression().fit(X, y)
    print(f"Linear regression model for {theme}: coefficient = {model.coef_[0][0]}, intercept = {model.intercept_[0]}")
    
    # Get the start and end years for the current theme
    start_year = theme_data['year'].min()
    end_year = theme_data['year'].max()
    
    # Generate a range of years from start to end
    years_range = np.arange(start_year, end_year+1).reshape(-1, 1)
    
    # Predict the counts for the range of years using the fitted model
    counts_pred = model.predict(years_range)
    
    # Plot the original data
    plt.scatter(X, y, label=f'{theme} actual')
    
    # Plot the linear regression line
    plt.plot(years_range, counts_pred, linestyle='--', label=f'{theme} predicted')

# Add labels and legend
plt.xlabel('Year')
plt.ylabel('Number of Sets')
plt.legend()
plt.title('Linear Regression of Number of Sets Over the Years for Top 10 Themes')
plt.show()


##### analyse the data with the colour ####
themes = pd.read_csv('Data/themes.csv')

# merge data
sets_with_themes = pd.merge(sets, themes, left_on='theme_id', right_on='id')
sets_with_themes.info()
inventory_sets_with_themes =pd.merge(inventory_sets, sets_with_themes, left_on='set_num', right_on='set_num')
inventory_sets_with_themes.info()
inventory_parts_with_colours = pd.merge(inventory_parts, colors, left_on='color_id', right_on='id')
inventory_parts_with_colours.info()
full_data =pd.merge(inventory_sets_with_themes, inventory_parts_with_colours, left_on='inventory_id', right_on='inventory_id')
full_data.info()

# Filter the data for recent years
recent_data = full_data[full_data['year'] > 2000]

# show data for specific theme
theme_name = 'Star Wars'  
theme_data = recent_data[recent_data['name_y'] == theme_name]

# create counter
color_counter = Counter()

# count number of parts for each theme for the different sets
for index, row in theme_data.iterrows():
    color_counter[row['name']] += row['quantity_y']

# plot 
colors, counts = zip(*color_counter.most_common(10))
plt.bar(colors, counts)
plt.xlabel('Colour')
plt.ylabel('Number of parts')
plt.title(f'Top 10 Colors for {theme_name} after the year 2000')
plt.xticks(rotation=90)
plt.show()


##### plot the top 10 colors, for each of the top 10 themes ######

# Filter the data for recent years
recent_data = full_data[full_data['year'] > 2000]

# Get the top 10 themes
top_themes = recent_data['name_y'].value_counts().nlargest(10).index

# For each of the top 10 themes, plot the top 10 colors
for theme in top_themes:
    theme_data = recent_data[recent_data['name_y'] == theme]
    
    # create counter
    color_counter = Counter()
    
    # count number of parts for each theme for the different sets
    for index, row in theme_data.iterrows():
        color_counter[row['name']] += row['quantity_y']
    
    # print top 10 colors
    print(f'Top 10 Colors for {theme}')
    for color, count in color_counter.most_common(10):
        print(f'{color}: {count}')
    
    # plot 
    colors, counts = zip(*color_counter.most_common(10))
    plt.bar(colors, counts)
    plt.xlabel('Colour')
    plt.ylabel('Number of parts')
    plt.title(f'Top 10 Colors for {theme} after the year 2000')
    plt.xticks(rotation=90)
    plt.show()

##### plot the top 10 colors, for each of the top 10 themes in a specific year######

# Filter the data for recent years
specific_data = full_data[full_data['year'] == 2020]

# Get the top 10 themes
top_themes = specific_data['name_y'].value_counts().nlargest(10).index

# For each of the top 10 themes, plot the top 10 colors
for theme in top_themes:
    theme_data = recent_data[recent_data['name_y'] == theme]
    
    # create counter
    color_counter = Counter()
    
    # count number of parts for each theme for the different sets
    for index, row in theme_data.iterrows():
        color_counter[row['name']] += row['quantity_y']
    
    # print top 10 colors
    print(f'Top 10 Colors for {theme}')
    for color, count in color_counter.most_common(10):
        print(f'{color}: {count}')
    
    # plot 
    colors, counts = zip(*color_counter.most_common(10))
    plt.bar(colors, counts)
    plt.xlabel('Colour')
    plt.ylabel('Number of parts')
    plt.title(f'Top 10 Colors for {theme} after the year 2000')
    plt.xticks(rotation=90)
    plt.show()