# Twitch Streamers: EDA, Regression Models, and Huber Robust Regression

## Overview

This project analyzes data from **1,000 Twitch channels** to explore relationships between channel activity, audience engagement, popularity, and average viewership. It combines exploratory data analysis (EDA), correlation analysis, outlier detection, and regression modeling to predict **Average viewers**.

Five regression models are compared to evaluate their predictive performance, with particular emphasis on Huber Robust Regression as a method for retaining potentially meaningful extreme observations.

## Objectives

- Explore distributions and relationships among Twitch channel metrics.
- Identify variables associated with average viewership.
- Detect potential outliers using the Interquartile Range (IQR) method.
- Establish a baseline using Linear Regression.
- Compare Linear Regression, Random Forest, Gradient Boosting, XGBoost, and Huber Robust Regression.
- Evaluate model performance using test-set \(R^2\) and RMSE.

## Dataset

The dataset, `twitchdata-update.csv`, contains information on **1,000 Twitch channels** and 11 columns.

The analysis focuses on six numerical predictors:

- `Watch time(Minutes)`
- `Stream time(minutes)`
- `Peak viewers`
- `Followers`
- `Followers gained`
- `Views gained`

The target variable is `Average viewers`.

The columns `Channel`, `Language`, `Mature`, and `Partnered` are excluded from the modeling features. The dataset is not included in this repository; users must obtain the CSV separately and update the file path if necessary.

## Methodology

### 1. Data Exploration and Preprocessing

- Examined dataset dimensions, missing values, and duplicate records.
- Selected six numerical predictors and one target variable.
- Reviewed descriptive statistics and variable distributions.
- Used log-transformed histograms to visualize skewed distributions.

**Interpretation:** Channel activity and audience metrics vary substantially across observations. The distributions exhibit skewness, indicating that a small number of channels may have exceptionally large values. These observations require careful analysis rather than automatic removal.

### 2. Correlation Analysis

Both Pearson and Spearman correlation coefficients were used to examine relationships between the numerical variables.

**Key findings:**

- `Peak viewers` has the strongest Pearson correlation with `Average viewers`, at approximately **0.683**.
- `Peak viewers` also has the strongest Spearman correlation, at approximately **0.728**.
- `Watch time(Minutes)` has a Pearson correlation of approximately **0.476**.
- `Stream time(minutes)` has a negative Pearson correlation of approximately **-0.249**.

**Interpretation:** Peak audience size has the strongest observed association with average viewership among the selected predictors. Differences between Pearson and Spearman correlations indicate that the relationships may be affected by skewness or nonlinear monotonic patterns. Correlation does not establish causation.

### 3. Outlier Analysis

The Interquartile Range (IQR) method was used to flag observations outside the interval defined by \(Q_1 - 1.5 \times IQR\) and \(Q_3 + 1.5 \times IQR\).

**Results:**

- Total observations: 1,000
- Observations flagged in at least one of the seven numerical variables: **288**
- Observations flagged by `Average viewers` alone: **100**

**Interpretation:** The IQR method identifies potentially extreme observations, not necessarily erroneous data. Some flagged observations may correspond to genuinely popular channels with unusually high audience or activity metrics. Removing all 288 observations could discard valuable information and alter the population being analyzed. Therefore, all observations were retained in the primary modeling analysis.

### 4. Regression Modeling

The dataset was divided into training and test sets using an 80:20 split with `random_state=42`.

- Training observations: 800
- Test observations: 200

The following models were fitted and evaluated on the same test set:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor
4. XGBoost Regressor
5. Huber Robust Regression

Predictors were standardized within the Huber regression pipeline. Tree-based models were fitted without standardization.

## Model Performance

The models were evaluated using two metrics.

- **\(R^2\):** Measures how much of the variation in the target variable is explained by the model on the test set. Higher values indicate better performance relative to a mean-based baseline.
- **RMSE (Root Mean Squared Error):** Measures the typical scale of prediction errors while penalizing larger errors more heavily. Lower values indicate better performance.

### Test-Set Results

| Model | Test \(R^2\) | Test RMSE |
|---|---:|---:|
| Huber Robust Regression | **0.751** | **2305.799** |
| XGBoost | 0.644 | 2756.200 |
| Linear Regression | 0.409 | 3554.795 |
| Random Forest | 0.347 | 3736.101 |
| Gradient Boosting | 0.232 | 4050.464 |

### Interpretation of Results

**Huber Robust Regression**

Huber Robust Regression achieved the highest test \(R^2\) of **0.751** and the lowest RMSE of **2305.799** among the five evaluated models. Its \(R^2\) was substantially higher than the Linear Regression baseline of 0.409.

Huber regression uses a robust loss function that reduces the influence of large residuals during model fitting. This makes it a useful approach when extreme observations are potentially meaningful and should be retained.

**XGBoost**

XGBoost achieved a test \(R^2\) of 0.644 and RMSE of 2756.200. It performed better than the Linear Regression, Random Forest, and Gradient Boosting models under the configurations tested, but did not outperform Huber Robust Regression.

**Linear Regression**

Linear Regression achieved a test \(R^2\) of 0.409 and RMSE of 3554.795. This provides a baseline for assessing whether the alternative models improve predictive performance.

**Random Forest and Gradient Boosting**

Random Forest achieved a test \(R^2\) of 0.347, while Gradient Boosting achieved 0.232. Both models performed below the linear baseline on this test set. This demonstrates that ensemble methods do not necessarily outperform simpler or robust alternatives for every dataset and parameter configuration.

### Actual vs. Predicted Values

The notebook includes actual-versus-predicted plots for each model. Predictions closer to the diagonal reference line indicate better agreement between actual and predicted viewership.

These visualizations help assess whether the models capture the general pattern of viewership and whether they struggle to predict values for exceptionally popular channels.

## Key Takeaways

- `Peak viewers` has the strongest observed correlation with `Average viewers` among the selected predictors.
- The IQR method flagged **288 potential outlier observations**, which were retained in the primary analysis.
- Huber Robust Regression achieved the best test-set performance among the five models, with \(R^2 = 0.751\) and RMSE = 2305.799.
- XGBoost was the second-best model according to both reported metrics.
- Robust regression provides a useful alternative to discarding potentially meaningful extreme observations.

## Tools and Technologies

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- Exploratory Data Analysis
- Pearson and Spearman Correlation
- IQR-Based Outlier Detection
- Regression Modeling
- Huber Robust Regression

## Repository Contents

- `Analysis_of_Top_Streamers_on_Twitch.ipynb` — notebook containing data exploration, visualizations, correlation analysis, outlier detection, regression modeling, evaluation metrics, and interpretations.

## How to Run

1. Clone or download this repository.
2. Install the required Python libraries.
3. Obtain the dataset and place `twitchdata-update.csv` in the notebook's working directory.
4. Update `DATA_PATH` in the notebook if the dataset is stored elsewhere.
5. Open the notebook in Google Colab or Jupyter Notebook.
6. Run the cells sequentially.

### Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## Limitations and Future Improvements

- Model evaluation uses a single train-test split; cross-validation could provide a more robust estimate of generalization performance.
- Hyperparameter tuning may improve predictive performance.
- Further residual analysis could investigate prediction errors for highly popular channels.
- Alternative transformations of skewed variables could be explored and evaluated.

## Conclusion

This project combines exploratory data analysis, correlation analysis, outlier detection, and regression modeling to investigate Twitch channel viewership.

The IQR method flagged **288 potential outlier observations**. Rather than discarding these observations and potentially losing critical information about high-performing channels, Huber Robust Regression was explored to retain all observations while reducing the influence of large residuals during model fitting.

Huber Robust Regression achieved a test \(R^2\) of **0.751**, compared with **0.409** for the Linear Regression baseline. Based on the reported test-set \(R^2\) and RMSE, Huber Robust Regression was the strongest-performing model among the five evaluated.

## Author

**Eshita Basak**

GitHub: [eshitab22](https://github.com/eshitab22)
