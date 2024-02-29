import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew

# Improved load and clean data function
def load_and_clean_data(athletes_path, regions_path):
    """Load athletes and regions datasets, merge, and clean."""
    athletes_df = pd.read_csv(athletes_path)
    regions_df = pd.read_csv(regions_path)
    merged_df = pd.merge(athletes_df, regions_df, on='NOC', how='left')
    
    # Impute missing Age values with median of respective sport/event
    merged_df['Age'] = merged_df.groupby(['Sport', 'Event'])['Age'].transform(lambda x: x.fillna(x.median()))
    
    # Create Medalist binary indicator
    merged_df['Medalist'] = merged_df['Medal'].notnull().astype(int)
    
    return merged_df

# Visualization functions with enhanced color schemes and labels
def plot_age_distribution(data):
    """Plot the distribution of athlete ages with an advanced color scheme."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Age'].dropna(), bins=30, color='#007acc', kde=True, edgecolor='black')
    plt.title('Distribution of Athlete Ages')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()

def plot_gender_distribution(data):
    """Plot the gender distribution among Olympic athletes with labels."""
    gender_counts = data['Gender'].value_counts()
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
    plt.figure(figsize=(8, 8))
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title('Gender Distribution Among Olympic Athletes')
    plt.show()

def plot_medals_over_years(data):
    """Plot the trend of medals won over the years with advanced colors."""
    medals_over_years = data[data['Medalist'] == 1].groupby('Year').size()
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=medals_over_years.index, y=medals_over_years.values, marker='o', linestyle='-', color='#d9534f')
    plt.title('Trend of Medals Won Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Number of Medals')
    plt.show()

def plot_correlation_heatmap(data):
    """Plot the heatmap of the correlation matrix with an advanced color scheme.
    Filters out non-numeric columns before plotting."""
    # Select only numeric columns for correlation calculation
    numeric_cols = data.select_dtypes(include=np.number)
    corr_matrix = numeric_cols.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Numeric Features')
    plt.show()


# Statistical Analysis including kurtosis and skewness
def perform_statistical_analysis(data):
    """Print descriptive statistics, kurtosis, and skewness of the dataset."""
    print(data.describe())
    for col in ['Age', 'Year']:
        print(f"\nKurtosis for {col}: {kurtosis(data[col].dropna())}")
        print(f"Skewness for {col}: {skew(data[col].dropna())}")

# Example usage
if __name__ == "__main__":
    athletes_path = 'all_athlete_games.csv'
    regions_path = 'all_regions.csv'
    data = load_and_clean_data(athletes_path, regions_path)

    # Perform statistical analysis
    perform_statistical_analysis(data)

    # Visualization
    plot_age_distribution(data)
    plot_gender_distribution(data)
    plot_medals_over_years(data)
    plot_correlation_heatmap(data)
