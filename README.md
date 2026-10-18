# Car Price Prediction with Linear Regression

Explore how vehicle specifications relate to car prices using **Python, exploratory data analysis, and multiple linear regression**. This notebook combines visual analysis with a scikit-learn pipeline that fits feature scaling, categorical encoding, and linear regression on the training set. The model achieves **R² = 0.8715** on a 39-car test split.

The project works with **193 cars across 21 manufacturers**, covering engine performance, vehicle dimensions, fuel economy, and body configuration.

## Project highlights

- **Data exploration:** Inspect feature types, summary statistics, missing values, and duplicate records.
- **Visual analysis:** Explore distributions and price relationships through count plots, density plots, scatter plots, box plots, and manufacturer-level median prices.
- **Consistent preprocessing:** Fit one-hot encoding and min–max feature scaling on the training set, then reuse them for prediction through a single pipeline.
- **Regression modeling:** Use `LinearRegression` to estimate price from multiple vehicle characteristics.
- **Evaluation:** Compare against a mean-price baseline using R², MAE, MSE, and RMSE, with prediction and residual plots.

## Dataset

The project uses [`data.csv`](data.csv), placed alongside the notebook.

| Property | Value |
| --- | --- |
| Vehicles | 193 |
| Manufacturers | 21 |
| Columns | 24: 23 input attributes and the `price` target |
| Categorical input attributes | 10 |
| Numerical input attributes | 13 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Price range in the supplied data | 5,118–45,400 |

### Feature groups

| Group | Examples |
| --- | --- |
| Manufacturer and configuration | `make`, `body-style`, `num-of-doors`, `drive-wheels` |
| Engine and fuel system | `engine-type`, `engine-size`, `horsepower`, `fuel-type`, `fuel-system` |
| Size and weight | `wheel-base`, `length`, `width`, `height`, `curb-weight` |
| Fuel economy | `city-mpg`, `highway-mpg` |
| Prediction target | `price` |

### Dataset background and attribution

The column names and sampled records match the [UCI Automobile dataset](https://archive.ics.uci.edu/dataset/10/automobile), indicating that this CSV is a prepared version of that dataset. Use the supplied `data.csv` for this project’s 193-row, 24-column layout.

The original Automobile dataset contains 205 records and includes information from the **1985 Ward’s Automotive Yearbook**. UCI credits **Jeffrey Schlimmer** and distributes the original dataset under **CC BY 4.0**.

**Citation:** Schlimmer, J. (1985). *Automobile* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5B01C](https://doi.org/10.24432/C5B01C).

## Exploratory analysis

The notebook examines the data from several complementary perspectives:

| Visualization | Question explored |
| --- | --- |
| Count plots | Which categories are most common? |
| Density plots | How are numerical variables distributed? |
| Scatter plots | How do numerical vehicle attributes relate to price? |
| Box plots | How do price distributions differ across vehicle categories? |
| Median-price bar plot | How do manufacturers compare by median price? |

As a numerical companion to these visualizations, the supplied CSV shows the following Pearson correlations with price:

| Attribute | Correlation with price |
| --- | ---: |
| Engine size | 0.889 |
| Curb weight | 0.835 |
| Horsepower | 0.812 |
| Highway MPG | −0.719 |

These descriptive statistics are calculated from `data.csv` in the notebook. Larger engines, heavier cars, and higher horsepower are associated with higher prices in this sample; higher highway fuel economy is associated with lower prices. These are relationships in the dataset, not measures of individual features’ causal effects.

## Modeling workflow

1. **Inspect and visualize:** Review the dataset and explore relationships between vehicle attributes and price.
2. **Separate inputs and target:** Keep the 23 vehicle attributes in `X` and the original price values in `y`.
3. **Split the observations:** Reserve 20% for testing with `random_state=20`, yielding **154 training rows** and **39 test rows**.
4. **Prepare features:** Use a `ColumnTransformer` to scale numerical features with `MinMaxScaler` and encode categories with `OneHotEncoder`.
5. **Train one pipeline:** Fit preprocessing and `LinearRegression` together on the training partition. Category encoding drops one reference category per feature and handles unseen categories during prediction.
6. **Evaluate:** Predict test prices, compare against a mean-price baseline, and inspect prediction errors.

The encoder and scaler learn only from the training set. The target remains in its original units, and the original dataframe is preserved. The fitted pipeline accepts new rows with the same input columns.

## Results

Evaluation uses the fixed **154/39 train/test split** described above.

| Model | R² | MAE | MSE | RMSE |
| --- | ---: | ---: | ---: | ---: |
| Mean-price baseline | -0.0450 | 7,329.37 | 101,119,900.56 | 10,055.84 |
| Linear regression | 0.8715 | 2,396.38 | 12,429,390.22 | 3,525.53 |

Linear regression reduces RMSE by **64.9%** compared with predicting the training set’s mean price for every test car.

| Metric | Interpretation |
| --- | --- |
| R² | Measures model fit relative to predicting the test target’s mean |
| Mean absolute error (MAE) | Average absolute prediction error, in original price units |
| Mean squared error (MSE) | Average squared prediction error |
| Root mean squared error (RMSE) | Error in original price units, with more emphasis on large errors |

The notebook includes an actual-versus-predicted table, a prediction plot, and a residual plot. These metrics describe this specific test split; they are not a cross-validation average.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/K1M1A-K1/Car-Price-Prediction-With-Linear-Regression.git
cd Car-Price-Prediction-With-Linear-Regression
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

The requirements include **NumPy**, **pandas**, **Matplotlib**, **Seaborn**, **scikit-learn**, and **Jupyter**.

### 3. Place the dataset beside the notebook

Keep the filename `data.csv`, matching the notebook’s `pd.read_csv("data.csv")` call.

### 4. Open the notebook

```bash
jupyter notebook price_prediction.ipynb
```

Run all cells from top to bottom. The notebook includes all required imports, fits the complete pipeline, evaluates it, and demonstrates prediction for a held-out car. For a clean rerun, restart the kernel and run all cells.

## Project files

| File | Purpose |
| --- | --- |
| `price_prediction.ipynb` | Exploratory analysis, preprocessing pipeline, regression, and evaluation |
| `data.csv` | Vehicle attributes and target prices |
| `requirements.txt` | Python dependencies |
| `README.md` | Project overview, dataset information, and setup instructions |

## References

- [UCI Machine Learning Repository — Automobile](https://archive.ics.uci.edu/dataset/10/automobile).
- [scikit-learn — OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).
- [scikit-learn — Getting Started](https://scikit-learn.org/stable/getting_started.html).
- [scikit-learn — MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).
- [Car Price Prediction Using Linear Regression — reference project credited in the notebook](https://github.com/roshancyriacmathew/Car-price-prediction-using-Linear-regression-Machine-Learning-Project).

Additional Kaggle tutorials and learning resources are credited in the notebook’s References section.
