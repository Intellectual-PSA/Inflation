import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Assuming data is stored in CSV format with columns: year, GDP, inflation
df = pd.read_csv('economic_data.csv')

# Plotting GDP over time
plt.figure(figsize=(10,5))
plt.plot(df['year'], df['GDP'], label='GDP')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.title('GDP Over Time')
plt.legend()
plt.show()

# Plotting Inflation over time
plt.figure(figsize=(10,5))
plt.plot(df['year'], df['inflation'], label='Inflation', color='red')
plt.xlabel('Year')
plt.ylabel('Inflation')
plt.title('Inflation Over Time')
plt.legend()
plt.show()

# Analyzing correlation between GDP and Inflation
model = LinearRegression()
model.fit(df[['GDP']], df['inflation'])

print(f'Correlation between GDP and Inflation: {model.score(df[["GDP"]], df["inflation"])}')
