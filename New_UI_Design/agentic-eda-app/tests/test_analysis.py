import pytest
import pandas as pd
from src.eda.analysis import (
    summary_statistics,
    handle_missing_values,
    detect_outliers,
    correlation_matrix,
)

def test_summary_statistics():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [5, 6, 7, 8, 9],
        'C': [10, 11, 12, 13, 14]
    })
    result = summary_statistics(df)
    expected = {
        'A': {'mean': 3.0, 'std': 1.5811388300841898, 'min': 1, 'max': 5},
        'B': {'mean': 7.0, 'std': 1.5811388300841898, 'min': 5, 'max': 9},
        'C': {'mean': 12.0, 'std': 1.5811388300841898, 'min': 10, 'max': 14},
    }
    assert result == expected

def test_handle_missing_values():
    df = pd.DataFrame({
        'A': [1, None, 3, 4, None],
        'B': [5, 6, None, 8, 9]
    })
    result = handle_missing_values(df)
    expected = pd.DataFrame({
        'A': [1, 3, 4],
        'B': [5, 6, 8, 9]
    })
    pd.testing.assert_frame_equal(result, expected)

def test_detect_outliers():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 100],
        'B': [5, 6, 7, 8, 9]
    })
    result = detect_outliers(df['A'])
    expected = {'outliers': [100], 'count': 1}
    assert result == expected

def test_correlation_matrix():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [5, 6, 7, 8, 9],
        'C': [10, 11, 12, 13, 14]
    })
    result = correlation_matrix(df)
    expected = {
        'A': {'A': 1.0, 'B': 1.0, 'C': 1.0},
        'B': {'A': 1.0, 'B': 1.0, 'C': 1.0},
        'C': {'A': 1.0, 'B': 1.0, 'C': 1.0},
    }
    pd.testing.assert_frame_equal(result, expected)