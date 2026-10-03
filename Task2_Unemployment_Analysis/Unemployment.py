import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("Unemployment in India.csv")


df.columns = df.columns.str.strip()

print("First 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset shape:")
print(df.shape)


for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()


df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)


print("\nMissing values:")
print(df.isnull().sum())


print("\nUnemployment Rate Statistics:")
print(df["Estimated Unemployment Rate (%)"].describe())


average_rate = df["Estimated Unemployment Rate (%)"].mean()

print("\nAverage Unemployment Rate:")
print(f"{average_rate:.2f}%")


highest = df.loc[df["Estimated Unemployment Rate (%)"].idxmax()]

print("\nHighest Unemployment Rate:")
print(highest[[
    "Region",
    "Date",
    "Estimated Unemployment Rate (%)",
    "Area"
]])


lowest = df.loc[df["Estimated Unemployment Rate (%)"].idxmin()]

print("\nLowest Unemployment Rate:")
print(lowest[[
    "Region",
    "Date",
    "Estimated Unemployment Rate (%)",
    "Area"
]])


state_average = (
    df.groupby("Region")["Estimated Unemployment Rate (%)"]
    .mean()
    .sort_values(ascending=False)
)

print("\nState-wise Average Unemployment Rate:")
print(state_average)


top10 = state_average.head(10)

plt.figure(figsize=(10, 6))
top10.sort_values().plot(kind="barh")
plt.xlabel("Average Unemployment Rate (%)")
plt.ylabel("Region")
plt.title("Top 10 Regions by Average Unemployment Rate")
plt.tight_layout()
plt.show()


monthly_rate = df.groupby("Date")[
    "Estimated Unemployment Rate (%)"
].mean()

plt.figure(figsize=(12, 6))
plt.plot(monthly_rate.index, monthly_rate.values, marker="o")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.title("Unemployment Rate Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


covid_data = df[
    (df["Date"] >= "2020-03-01") &
    (df["Date"] <= "2020-06-30")
]

print("\nCOVID-19 Period Analysis:")

if len(covid_data) > 0:
    covid_average = covid_data[
        "Estimated Unemployment Rate (%)"
    ].mean()

    print(f"Average unemployment rate during COVID period: {covid_average:.2f}%")


plt.figure(figsize=(8, 5))
sns.histplot(
    df["Estimated Unemployment Rate (%)"],
    kde=True
)
plt.xlabel("Unemployment Rate (%)")
plt.title("Distribution of Unemployment Rate")
plt.tight_layout()
plt.show()


numeric_data = df.select_dtypes(include="number")

plt.figure(figsize=(8, 5))
sns.heatmap(
    numeric_data.corr(),
    annot=True,
    fmt=".2f"
)
plt.title("Correlation Between Numerical Variables")
plt.tight_layout()
plt.show()

print("\n====================================")
print("UNEMPLOYMENT ANALYSIS COMPLETED")
print("====================================")