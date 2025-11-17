# Agentic EDA Application

## Overview
The Agentic EDA application is a powerful tool designed for exploratory data analysis (EDA) using Streamlit. It allows users to upload datasets, visualize data distributions, and gain insights through AI-generated recommendations. The application integrates various modules for data loading, analysis, visualization, and user interaction.

## Features
- **Data Upload**: Users can upload multiple CSV files for analysis.
- **Exploratory Data Analysis**: The app provides statistical summaries, correlation matrices, and outlier detection.
- **Visualizations**: Generate insightful plots such as heatmaps and boxplots to understand data distributions.
- **AI Insights**: Leverage Azure OpenAI to generate actionable insights and feature engineering ideas based on the uploaded data.
- **Interactive UI**: A user-friendly interface that allows for easy navigation and interaction with the analysis results.

## Project Structure
```
agentic-eda-app
├── src
│   ├── app.py               # Main entry point for the Streamlit application
│   ├── eda                  # Module for exploratory data analysis
│   │   ├── __init__.py
│   │   ├── loader.py        # Handles data loading and validation
│   │   ├── analysis.py      # Contains EDA functions
│   │   ├── plots.py         # Generates visualizations
│   │   └── scoring.py       # Computes data quality scores
│   ├── ui                   # Module for user interface components
│   │   ├── __init__.py
│   │   ├── layout.py        # Defines the layout of the app
│   │   ├── controls.py      # UI controls for user interaction
│   │   └── chat.py          # Manages the chat interface
│   ├── ai                   # Module for AI interactions
│   │   ├── __init__.py
│   │   └── azure_openai.py  # Functions to interact with Azure OpenAI
│   └── config.py            # Configuration settings
├── tests                    # Directory for unit tests
│   ├── test_analysis.py     # Tests for analysis functions
│   └── test_ui.py          # Tests for UI components
├── requirements.txt         # Project dependencies
├── .env.example             # Template for environment variables
├── .gitignore               # Files to ignore in Git
└── README.md                # Project documentation
```

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd agentic-eda-app
   ```
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Set up your environment variables by copying `.env.example` to `.env` and filling in the necessary values.
2. Run the application:
   ```
   streamlit run src/app.py
   ```
3. Open your web browser and navigate to `http://localhost:8501` to access the application.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.