from src.data_preprocessing import preprocess_pipeline

# Path to your dataset
file_path = "data/raw/World_data_GDP.csv"

# Run preprocessing
df = preprocess_pipeline(file_path)

# Quick checks
print("✅ Shape:", df.shape)
print("\n✅ Columns:", df.columns.tolist()[:10])
print("\n✅ Sample Data:")
print(df.head())

# Check missing values
print("\n✅ Missing Values:", df.isna().sum().sum())