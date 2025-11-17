def calculate_data_quality_score(df):
    total_cells = df.size
    missing_count = df.isnull().sum().sum()
    missing_ratio = missing_count / total_cells if total_cells > 0 else 0.0
    
    # Assuming outlier detection is based on IQR method
    outlier_count = 0
    for col in df.select_dtypes(include=['number']).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count += ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    
    outlier_ratio = outlier_count / total_cells if total_cells > 0 else 0.0
    data_quality_score = max(0.0, 100.0 - (missing_ratio * 60.0 + outlier_ratio * 40.0) * 100.0)
    
    return data_quality_score

def compute_metrics(df):
    metrics = {
        'data_quality_score': calculate_data_quality_score(df),
        'num_rows': df.shape[0],
        'num_columns': df.shape[1],
        'missing_values': df.isnull().sum().to_dict(),
    }
    return metrics