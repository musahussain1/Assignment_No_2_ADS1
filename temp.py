# -*- coding: utf-8 -*-
# Pandas: Importing and manipulating data
import pandas as pd
# Numpy: Numerical computations and array operations
import numpy as np
# Matplotlib: Data visualization
import matplotlib.pyplot as plt
# Seaborn: Statistical data visualization
import seaborn as sns
sns.set()
#loading data
df = pd.read_csv("WorldBank_CO2.csv")
data = df.copy()
#Basic Structure
df.describe()
df.describe(include='all')
df.describe(include='object')
#null values
df.isnull().sum()
#Data types
df.dtypes
df.describe(include = ["O"])
### Ingest and manipulate the data using pandas dataframes.
import pandas as pd

def read_world_bank_data(filename):
    # read in the data and skip the first 4 rows
    df = pd.read_csv(filename)
    
    # drop any columns that are completely empty
    df.dropna(axis=1, how='all', inplace=True)
    
    # rename the 'Country Name' column to 'Country'
    df = df.rename(columns={'Country Name': 'Country'})
    
    # remove any rows that don't have a country name
    df = df[df['Country'].notna()]
    
    # pivot the data to have years as columns and countries as rows
    df_years = df.pivot(index='Country', columns='Indicator Name')
    df_years.columns = df_years.columns.droplevel(0)
    
    # transpose the data to have countries as columns and years as rows
    df_countries = df_years.transpose()
    
    return df_years, df_countries
df_years, df_countries = read_world_bank_data('WorldBank_CO2.csv')
df_years
df_countries
### Explore the statistical properties of a few indicators
# Filter the data to only include rows where the indicator name is "GDP per capita (constant 2010 US$)"
gdp_per_capita = df[df['Indicator Name'] == 'GDP per capita (constant 2010 US$)']

# Print the first 5 rows of the filtered data
print(gdp_per_capita.head()) 
# Select the indicators, countries, and years of interest
indicators = ['GDP per capita (current US$)', 'Life expectancy at birth, total (years)', 'CO2 emissions (metric tons per capita)']
countries = ['AFG', 'AFE', 'AFW']
years = [str(year) for year in range(2010, 2019)]

# Select the relevant columns and filter the data
df = df[['Country Name', 'Country Code', 'Indicator Name'] + years]
df = df.loc[(df['Indicator Name'].isin(indicators)) & (df['Country Code'].isin(countries))]

# Melt the DataFrame to long format
df = df.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name'], var_name='Year', value_name='Value')

# Convert Year to int
df['Year'] = df['Year'].astype(int)

# Compute summary statistics for each indicator and country
summary_stats = df.groupby(['Indicator Name', 'Country Code'])['Value'].describe()

# Print the summary statistics
print(summary_stats)
# Drop unnecessary columns
data = data.drop(columns=['Country Code', 'Indicator Name', 'Indicator Code'])
# Melt the data
data = pd.melt(data, id_vars=['Country Name'], var_name='Year', value_name='Value')
data
data.isnull().sum()
data['Value'].isnull().sum()
data.loc[data['Value'].isnull(),'Value'] == 0
data[data['Value'] == 0]
year_wise=data.groupby('Year')['Value'].sum().reset_index()
year_wise
Year_country_wise=data.groupby(['Country Name','Year'])['Value'].sum().reset_index()
Year_country_wise
# Filter the dataset based on the countries and years you want to include
countries = ['Bhutan', 'Spain', 'Singapore', 'Japan','United States','Bahrain']
years = ['1990', '2000', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
filtered_data = Year_country_wise[(Year_country_wise['Country Name'].isin(countries)) & (Year_country_wise['Year'].isin(years))]

# Set up the plot
fig, ax = plt.subplots(figsize=(60,20 ))

# Plot the line graph for the filtered data
for country in countries:
    country_data = filtered_data[filtered_data['Country Name'] == country]
    ax.plot(country_data['Year'], country_data['Value'], marker='o', linewidth=7.5, label=country)

# Set plot title, labels, and legend
ax.set(title='Year Wise CO2 Emission',
       xlabel='Year',
       ylabel='Emission Values')
ax.legend()

# Show the plot
plt.show()

temp=data.groupby('Country Name')['Value'].sum().reset_index()
temp=temp.sort_values('Value',ascending=False)
temp
plt.figure(figsize=(70,20))
ax=sns.barplot(x='Country Name',y='Value',data=temp.head(10));
plt.xlabel('Country Name')
plt.ylabel('Emission Value')
plt.title('Top 10 countries with high CO2 emission')

def explore_correlations(dataframe):
    # Compute correlation matrix
    corr_matrix = dataframe.corr()

    # Create a heatmap of the correlation matrix
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

# Assuming 'df' is your pandas DataFrame
explore_correlations(df)

import pandas as pd

# Read the Excel file
df_p = pd.read_excel('Population_growth.xlsx')

# Print the DataFrame
print(df_p)

df_p.head()

import matplotlib.pyplot as plt

# Data was taken from world bank websites only those countries are selected which were used in Co2 Emission Indicator
years = [1990, 2000, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]

bahrain = [3.4, 2.8, 3.0, 3.8, 3.8, 3.4, 3.3, 2.1, 0.5, -1.1, -1.0]
us = [1.1, 1.1, 0.7, 0.7, 0.7, 0.7, 0.6, 0.5, 0.5, 1.0, 0.1]
bhutan = [2.6, 2.8, 1.1, 1.0, 0.9, 0.9, 0.8, 0.8, 0.7, 0.7, 0.6]
singapore = [3.9, 1.7, 1.6, 1.3, 1.2, 1.3, 0.1, 0.5, 1.1, -0.3, -4.2]
japan = [0.3, 0.2, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.3, -0.5]
spain = [0.1, 0.4, -0.3, -0.3, -0.1, 0.1, 0.2, 0.4, 0.7, 0.5, 0.1]
# Set up the plot
fig, ax = plt.subplots(figsize=(60,20 ))

# Line plot

plt.plot(years, bahrain, label='Bahrain')
plt.plot(years, us, label='United States')
plt.plot(years, bhutan, label='Bhutan')
plt.plot(years, singapore, label='Singapore')
plt.plot(years, japan, label='Japan')
plt.plot(years, spain, label='Spain')

# Set plot title, labels, and legend
ax.set(title='Year Wise CO2 Emission',
       xlabel='Year',
       ylabel='Emission Values')
ax.legend()

# Title and labels
plt.title('Population growth')
plt.xlabel('Year')
plt.ylabel('Annual growth rate (%)')

# Legend
plt.legend()

# Display the plot
plt.show()
 