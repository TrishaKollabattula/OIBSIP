# Iris Flower Classification

**Oasis Infobyte Summer Internship Program (SIP) | Track: Data Science | Task 1**

## Objective
Train machine learning models that classify an iris flower as **Setosa**, **Versicolor** or **Virginica** from its four physical measurements (sepal length, sepal width, petal length, petal width).

## Dataset
The classic Iris dataset, loaded directly from `sklearn.datasets.load_iris()`.
- 150 samples, 4 numeric features, 3 balanced classes (50 each)
- No null values; 1 duplicate row (kept, since the dataset is so small)

## Tech Stack
Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, Jupyter Notebook

## Approach
1. **Exploratory Data Analysis:** shape, dtypes, null and duplicate checks, descriptive statistics, class balance
2. **Visualisation:** pairplot, box plots per feature, correlation heatmap, group means
3. **Feature selection discussion:** which features separate the species best
4. **Train/test split:** 80/20, stratified, `random_state=42`
5. **Models trained:** Logistic Regression, K-Nearest Neighbours (k=5), Decision Tree, Random Forest
6. **Evaluation:** accuracy, confusion matrix, classification report (precision, recall, F1)
7. **5-fold cross-validation** for a fairer comparison on this small dataset
8. **Feature importance** from the Random Forest

## Results

**Single test split (30 samples)**

| Model | Test Accuracy |
|---|---|
| K-Nearest Neighbours | 1.000 |
| Logistic Regression | 0.967 |
| Decision Tree | 0.933 |
| Random Forest | 0.900 |

**5-fold cross-validation**

| Model | Mean Accuracy | Std |
|---|---|---|
| Logistic Regression | 0.973 | 0.025 |
| K-Nearest Neighbours | 0.973 | 0.025 |
| Random Forest | 0.967 | 0.021 |
| Decision Tree | 0.953 | 0.034 |

## Key Findings
- **Petal length and petal width are the most discriminative features.** They are strongly correlated with each other (0.96), and the Random Forest gives them the highest importance (about 0.44 and 0.43). Sepal width is the weakest feature.
- **Setosa is perfectly separable** and is never misclassified by any model. All errors are between Versicolor and Virginica, whose measurements overlap.
- **Best model: K-Nearest Neighbours.** It scored 100% on the test split and tied for the top cross-validation accuracy (97.3%) with Logistic Regression.
- **Caveat:** the test set has only 30 samples, so one flower changes accuracy by about 3.3%. Random Forest ranked last on the test split but scored 96.7% in cross-validation. All four models perform within a few percentage points of each other, so no model is clearly superior.

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn notebook
jupyter notebook iris.ipynb
```
Then choose **Restart** and **Run All**. No dataset download is needed.

## Project Structure
```
DataScience-Task1-IrisFlowerClassification/
├── iris.ipynb      # full analysis, code and observations
└── README.md
```

## Author
**Trisha Kollabattula** | [LinkedIn](https://www.linkedin.com/in/trisha-kollabattula/) | [GitHub](https://github.com/TrishaKollabattula)
