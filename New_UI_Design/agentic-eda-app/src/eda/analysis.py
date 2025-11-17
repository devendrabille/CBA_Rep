def summarize_statistics(df):
    return df.describe()

def detect_missing_values(df):
    return df.isnull().sum()

def clean_data(df):
    # Example cleaning: drop rows with missing values
    return df.dropna()

def analyze_correlation(df):
    return df.corr()

def identify_outliers(df):
    outliers = {}
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        low_bound = q1 - 1.5 * iqr
        high_bound = q3 + 1.5 * iqr
        outliers[col] = df[(df[col] < low_bound) | (df[col] > high_bound)].shape[0]
    return outliers

def perform_eda(df):
    summary = summarize_statistics(df)
    missing_values = detect_missing_values(df)
    cleaned_data = clean_data(df)
    correlation = analyze_correlation(cleaned_data)
    outliers = identify_outliers(cleaned_data)
    
    return {
        "summary": summary,
        "missing_values": missing_values,
        "correlation": correlation,
        "outliers": outliers
    }