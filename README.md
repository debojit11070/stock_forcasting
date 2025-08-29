# Stock Forecasting Project

## Overview
This repository contains a complete workflow for forecasting Apple Inc. (AAPL) stock prices using various time series models. The project demonstrates data preprocessing, model training, evaluation, and analysis, leveraging Google Colab with a T4 GPU for computational efficiency. The goal is to predict the next 7-day closing prices based on historical data, making it a practical example for financial machine learning applications.

## Project Structure
- `notebooks/`: Jupyter notebooks for each step of the process.
  - `01_preprocessing.ipynb`: Data cleaning and feature engineering.
  - `03_model_training.ipynb`: Model training and comparison.
  - `04_analysis.ipynb`: Testing and analysis of the best model.
  - `05_report.ipynb` (optional): Summary and visualization report.
- `data/`: Raw and processed datasets.
  - `train_AAPL.csv`: Training dataset.
  - `val_AAPL.csv`: Validation dataset.
  - `test_AAPL.csv`: Test dataset.
- `models/`: Saved best model files.
  - `best_model.pkl` or `best_model_pred.csv`: Best performing model or predictions.
- `reports/`: Generated plots and outputs.
  - `model_comparison.png`: Comparison of all models' 7-day forecasts.
  - `test_forecast.png`: Actual vs. predicted plot for the test set.

## Methodology

### Data Preprocessing
- **Source**: Historical AAPL stock data (assumed from a financial API or dataset).
- **Steps** (in `01_preprocessing.ipynb`):
  - Loaded raw data into a pandas DataFrame.
  - Cleaned data by handling missing values and ensuring consistent indexing with dates.
  - Engineered features: `Volume`, 7-day Moving Average (`MA_7`), 1-day Lag (`Lag_1`), and Relative Strength Index (`RSI`).
  - Split data into train (~70%), validation (~15%), and test (~15%) sets, saved as CSV files.

### Model Training
- **File**: `03_model_training.ipynb`
- **Models**:
  - Traditional: ARIMA, SARIMA, Prophet.
  - Deep Learning (T4 GPU-accelerated): GRU, N-BEATS.
  - (Note: TFT, LSTM, and XGBoost were skipped due to issues.)
- **Process**:
  - Converted data to `darts` TimeSeries objects.
  - Trained models on the training set with validation monitoring.
  - Used covariates (`Volume`, `MA_7`, `Lag_1`, `RSI`) where supported.
  - Compared 7-day forecasts using Mean Absolute Error (MAE).
  - Saved the best model based on the lowest MAE.
- **Output**: `best_model.pkl` (for GRU/N-BEATS) or `best_model_pred.csv` (for ARIMA/SARIMA/Prophet), and `model_comparison.png`.

### Model Testing and Analysis
- **File**: `04_analysis.ipynb`
- **Process**:
  - Loaded the best model and applied it to the test set (`test_AAPL.csv`).
  - Generated 7-day predictions and computed MAE and Root Mean Squared Error (RMSE).
  - Visualized actual vs. predicted values for the test set.
- **Output**: `test_forecast.png` and printed metrics.

### Report
- **File**: `05_report.ipynb` (optional)
- **Content**:
  - Introduction: Project goal (forecasting AAPL stock prices).
  - Data Overview: Source and preprocessing details.
  - Methodology: Models used and training approach.
  - Results: Key metrics (MAE, RMSE) and plots.
  - Conclusion: Best model performance, limitations, and future improvements.

## Results
- **Best Model**: Determined by the lowest MAE on the validation set (varies by run; e.g., GRU or Prophet).
- **Test Metrics**: MAE and RMSE reported in `04_analysis.ipynb` output.
- **Visualizations**: `model_comparison.png` shows all models' performance, while `test_forecast.png` highlights the best model’s test set accuracy.

## Requirements
- **Python Libraries**:
  - `darts`
  - `statsmodels`
  - `prophet`
  - `tensorflow`
  - `pandas`
  - `matplotlib`
  - `numpy`
  - `torch` (for GPU support)
- **Environment**: Google Colab with T4 GPU runtime.
- **Installation**: Run `!pip install darts statsmodels prophet tensorflow` in Colab.

## How to Run
1. **Set Up Colab**:
   - Open Google Colab and upload this repository’s notebooks.
   - Change runtime to T4 GPU (Runtime > Change runtime type > T4 GPU).
2. **Run Notebooks**:
   - Execute `01_preprocessing.ipynb` to prepare data.
   - Run `03_model_training.ipynb` to train and compare models.
   - Run `04_analysis.ipynb` to test and analyze the best model.
   - (Optional) Create `05_report.ipynb` for a summary.
3. **Upload Files**:
   - Place `train_AAPL.csv`, `val_AAPL.csv`, and `test_AAPL.csv` in the Colab Files sidebar.
   - Save outputs (`best_model.pkl`, plots) to respective folders.
4. **Save and Download**:
   - Save notebooks to `notebooks/` and plots to `reports/`.
   - Download `models/` files for local storage.

## Limitations
- **Skipped Models**: TFT, LSTM, and XGBoost were excluded due to recurring issues (e.g., covariate mismatches, API errors).
- **Data**: Assumes consistent feature engineering; NaN values were filled with 0, which may affect accuracy.
- **Hyperparameters**: Models use default or reduced epochs (15) for speed; tuning could improve results.

## Future Improvements
- **Hyperparameter Tuning**: Use Optuna to optimize model parameters.
- **Additional Features**: Incorporate sentiment analysis or macroeconomic indicators.
- **Larger Dataset**: Test with more historical data for robustness.
- **Error Handling**: Enhance data validation to prevent NaN losses.

## Acknowledgments
- Built with assistance from xAI’s Grok, leveraging Colab’s T4 GPU for efficiency.
- Data preprocessing and modeling inspired by standard time series forecasting practices.

## Contact
- Feel free to reach out with questions or contributions via [your email or LinkedIn] (add your details here).

---
*Last Updated: August 29, 2025*
