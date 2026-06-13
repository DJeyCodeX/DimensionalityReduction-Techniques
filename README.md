# Dimensionality Reduction Techniques

## Project Overview

This project is a hands-on practice file for understanding different dimensionality reduction techniques in machine learning.

The main idea was to learn how we can reduce the number of input features in a dataset without losing too much useful information. This is important because large datasets can have many columns, and not every column adds value to a model. Some features may be repetitive, weak, noisy, or expensive to process.

I worked through both feature selection and feature extraction methods. The first part uses a BigMart sales dataset, and the second part uses image data where each image is converted into pixel-level features.

## Why Dimensionality Reduction Matters

When a dataset has too many features, a model can become harder to train and harder to explain. More features can also increase training time and sometimes reduce performance if the model starts learning noise instead of real patterns.

Dimensionality reduction helps by:

- Removing features that do not add much value
- Reducing repeated information between columns
- Making models faster to train
- Making the data easier to visualize
- Helping reduce overfitting
- Making high-dimensional data, like images, easier to work with

## Datasets Used

### BigMart sales data
It uses `big_mat_sales.csv` dataset. This dataset contains retail product and outlet information, with `Item_Outlet_Sales` as the target variable.

Some of the columns used include:

- `Item_Identifier`
- `Item_Weight`
- `Item_Visibility`
- `Item_MRP`
- `Outlet_Identifier`
- `Outlet_Size`
- `Outlet_Establishment_Year`
- `Item_Outlet_Sales`

This dataset was useful for practicing feature selection methods.

### Image data

The second part uses image data from a zip file and a CSV file containing labels. Each image is read using OpenCV and flattened into a long row of pixel values.

This part was useful because images naturally have many dimensions. Even one image can turn into thousands of pixel features, so dimensionality reduction becomes much more important.

### 1. Missing value treatment

Before applying dimensionality reduction techniques, I handled missing values in the BigMart dataset.

The missing values were treated as follows:

| Column | Treatment |
| --- | --- |
| `Item_Weight` | Filled with median value |
| `Outlet_Size` | Filled with mode value |

This was done because feature selection models and correlation checks work better when the data is clean.

### 2. Separating numerical and categorical columns

I separated the dataset into categorical and numerical columns using pandas data types.

This step helped me understand which techniques could be applied directly and which columns needed encoding first. For example, Random Forest in scikit-learn needs numeric input, so categorical columns had to be converted using one-hot encoding.

## Techniques Covered

### 1. Low Variance Filter

The first technique I used was a low variance filter.

The idea is simple: if a column has almost the same value for most rows, it probably does not help the model much. A feature with very low variance does not provide much separation between observations.

In the BigMart data, `Item_Visibility` had very low variance compared to other numerical columns. This made it a good example of a feature that could be considered for removal.

This method is easy to understand and fast to apply, but it only looks at the feature by itself. It does not check whether the feature is useful for predicting the target.

### 2. High Correlation Filter

The next method was checking correlation between numerical features.

If two columns are highly correlated, they may be carrying almost the same information. In that case, keeping both may not help much, especially for models like linear regression.

I checked correlation between:

- `Item_Weight`
- `Item_Visibility`
- `Item_MRP`
- `Outlet_Establishment_Year`

In this dataset, there were no very highly correlated numerical feature pairs. That was still useful because it showed that feature removal should be based on evidence, not just done automatically.

### 3. Random Forest Feature Importance

I then used a Random Forest Regressor to find important features.

Before training the model, I removed the ID columns:

- `Item_Identifier`
- `Outlet_Identifier`

These columns act more like unique identifiers and are not very meaningful as normal predictive features.

After that, I applied one-hot encoding to categorical columns and trained a `RandomForestRegressor`. Random Forest gives feature importance scores, which helped me see which variables contributed more to predicting `Item_Outlet_Sales`.

I also plotted the top feature importances using a horizontal bar chart.

This was one of the more practical parts of the project because tree-based feature importance is commonly used when working with tabular data.

### 4. SelectFromModel

After using Random Forest manually, I also tried `SelectFromModel` from scikit-learn.

This is useful because it can automatically select features based on model importance scores. Instead of manually looking at the feature importance chart and choosing columns, `SelectFromModel` provides a more structured way to reduce the feature set.

### 5. Backward Feature Elimination

I used Recursive Feature Elimination (RFE) with Linear Regression for backward feature elimination.

The idea is:

- Start with all features
- Train the model
- Remove the least useful feature
- Repeat the process
- Keep the selected number of best features

In this project, I selected 10 features using RFE.

This method is more model-driven than the variance or correlation filters. It tries to understand feature usefulness in relation to model performance. The downside is that it can become slow when the dataset has many features.

### 6. Forward Feature Selection

I also explored forward feature selection using `f_regression`.

This method starts from the opposite direction. Instead of removing features, it identifies features that have stronger statistical relationships with the target.

In the code, I selected variables with an F-value greater than 10. This helped me practice how statistical scores can be used to shortlist useful features.

Forward and backward selection are good learning techniques, but they can be computationally expensive on large datasets.

### 7. Factor Analysis

After feature selection methods, I moved to feature extraction using image data.

For this part:

- Images were loaded from a folder using OpenCV
- Each image was flattened into a long pixel vector
- Pixel columns were created as `pixel0`, `pixel1`, `pixel2`, and so on
- Labels were added from `images_data.csv`

Then I applied Factor Analysis with 3 components.

The goal here was not to select a few original columns, but to create a smaller set of new hidden factors that summarize the original pixel data.

This helped me understand the difference between feature selection and feature extraction:

- Feature selection keeps original columns
- Feature extraction creates new transformed columns

### 8. Principal Component Analysis

The final method covered was PCA.

PCA creates new components that capture as much variance as possible from the original data. This is especially useful for image data, where every pixel can become a separate feature.

In this project, I:

- Visualized sample images
- Fit PCA on the pixel features
- Checked cumulative explained variance
- Used a 95% variance threshold as a reference
- Reduced the data using `PCA(n_components=200)`
- Plotted component-wise and cumulative explained variance

This showed how PCA can compress high-dimensional image data into fewer components while still keeping a large amount of information.

## Feature Selection vs Feature Extraction

One important thing I learned from this project is the difference between selecting features and extracting features.

| Approach | What it does | Examples from this project |
| --- | --- | --- |
| Feature selection | Keeps a smaller set of original columns | Low variance filter, high correlation filter, Random Forest importance, RFE, forward selection |
| Feature extraction | Creates new columns from the original data | Factor Analysis, PCA |

Feature selection is easier to explain because the final features still have their original names. Feature extraction can be more powerful for high-dimensional data, but the new components are harder to interpret directly.

## What I Learned

The biggest learning from this project was that dimensionality reduction is not one single technique. There are many ways to reduce features, and the right method depends on the type of data and the goal.

For normal tabular data, feature selection methods are easier to understand and explain. For image data, projection methods like PCA and Factor Analysis make more sense because the original features are just pixels, and there are too many of them to inspect manually.

I also learned that removing features should not be done blindly. A column with low variance may be removable, but correlation, model importance, and domain understanding should also be considered.

Another useful takeaway was that PCA does not simply "drop columns." It creates new components that capture the main patterns in the data. The explained variance plot helped me understand how many components are needed to keep most of the original information.

## Final Thoughts

This project helped me understand dimensionality reduction from both sides: reducing columns in a structured dataset and compressing high-dimensional image data.

The BigMart sales dataset was useful for learning feature selection methods like variance filtering, correlation filtering, Random Forest importance, RFE, and forward selection. The image dataset made PCA and Factor Analysis easier to appreciate because pixel data quickly becomes very high-dimensional.

Overall, this project gave me a better understanding of how dimensionality reduction can make machine learning workflows cleaner, faster, and easier to manage.
