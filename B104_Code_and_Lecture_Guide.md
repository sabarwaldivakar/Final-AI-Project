# B104 project: code and lecture guide

**Project:** Predicting Bank Term Deposits  
**Author:** Divakar Sabarwal  
**Updated:** 13 September 2026

Read this beside [the completed notebook](</Users/divakar/Desktop/AI project/Bank_management2.ipynb>). The [HTML version](</Users/divakar/Desktop/AI project/Bank_management2.html>) contains its executed outputs. This guide is a learning companion, separate from the notebook's character budget.

## 1. What EDA was, and what has now been completed

The previous update completed **exploratory data analysis (EDA)** only. EDA means looking carefully at the data before choosing how to prepare it and train a model. For example, discovering that very few customers subscribed explains why accuracy alone is misleading. Discovering that `pdays=999` has a special meaning explains why we should not treat it like an ordinary number of days.

The notebook now completes the full ML project:

1. Define a business question and prediction target.
2. Load and audit the dataset.
3. Reserve later data for final testing.
4. Explore the earlier development data.
5. Choose features available at the intended prediction time.
6. Encode categories, scale numbers and create one simple history indicator.
7. Train logistic regression and KNN, alongside simple baselines.
8. Compare a small grid of hyperparameter settings using validation data.
9. Refit the selected model and evaluate it on the reserved test set.
10. Explain its coefficients and discuss whether the business should use it.

The implementation uses ordinary pandas operations, plots, small functions, loops and standard scikit-learn components. It does not use neural networks, boosting, SHAP, automated feature selection or complicated cross-validation. `Pipeline` is just a convenient container for the preprocessing and model steps taught in class.

## 2. How to read and run the files

- Open `Bank_management2.ipynb` in Jupyter or your notebook editor.
- Keep the notebook in the AI project folder, with the dataset at `data/dataset.csv`.
- Restart the kernel and run all cells in order. Individual later cells depend on variables created earlier.
- A **Markdown cell** explains the purpose, choices or results. A **code cell** performs the calculation.
- The first code cell prints software versions. [requirements.txt](</Users/divakar/Desktop/AI project/requirements.txt>) records the tested packages.
- The `.html` file is for reading/submission; it cannot execute Python. Editing the notebook does not update HTML until you export again.

From the AI project directory, export an executed notebook with:

```bash
python3 -m nbconvert --to html Bank_management2.ipynb
```

The original EDA notebook is preserved in [the backups folder](</Users/divakar/Desktop/AI project/backups>). The completed notebook condenses that EDA to make space for modelling and discussion.

## 3. Lecture map

These references use **PDF page numbers**, which can differ from the slide number printed on the page. `n.d.` in a citation means the publication year was not established from the supplied material.

| Concept | Lecture and relevant PDF pages | Where we apply it |
|---|---|---|
| Classification and the complete ML process | [Introduction to Machine Learning](</Users/divakar/Desktop/ai and machine learning/lectures/1 - Introduction to Machine Learning.pdf>), pp.15-19, 46-50 | Business framing and the overall sequence |
| Chronological data and early splitting | [Pre-Modeling Steps](</Users/divakar/Desktop/ai and machine learning/lectures/2 - Pre-Modeling Steps.pdf>), pp.4-5 | Earlier fitting data, later validation, latest test |
| Missing information and class imbalance | Pre-Modeling Steps, pp.8-14 | `unknown` categories, class weighting and misleading accuracy |
| One-hot encoding and scaling | Pre-Modeling Steps, pp.19-27 | `OneHotEncoder`, `ColumnTransformer`, `StandardScaler` |
| Leakage and applying transformations | Pre-Modeling Steps, pp.40-41 | Feature timing and fitting preprocessing only on the allowed data |
| Logistic scores and a threshold | [Logistic Regression](</Users/divakar/Desktop/ai and machine learning/lectures/2 - Logistic Regression.pdf>), pp.2-4 | Binary predictions and coefficient explanations |
| Neighbour voting, distance and choosing k | [K-Nearest Neighbors](</Users/divakar/Desktop/ai and machine learning/lectures/5 - K-Nearest Neighbors.pdf>), pp.2-5, 8-9 | KNN comparison and its limitations |
| Regularisation and tuning | [Model Training](</Users/divakar/Desktop/ai and machine learning/lectures/8 - Model Training.pdf>), pp.9-25 | Logistic `C`, KNN `k`, the small grid search |
| Holdout, refitting and final testing | Model Training, pp.27-32 | Model selection followed by final evaluation |
| Classification metrics | Model Training, pp.42-57 | Accuracy, confusion matrix, precision, recall, F1 and ROC-AUC |

The generic shuffled split in Model Training is adapted using the chronological-data guidance in Pre-Modeling Steps. We use two algorithms with dedicated lecture decks. Clustering, linear regression and game-search algorithms would answer different questions and are not added simply to increase the algorithm count.

The initial roadmap proposed more extensions. Your request to stay close to the lectures changes that plan: F1 replaces average precision as the selection metric; one chronological holdout replaces expanding-window validation; logistic coefficients replace permutation importance. These are deliberate simplifications, not missing implementation steps.

## 4. Notebook section 1: the business question and loading

The question is whether customer information and **past campaign history** can help prioritise people likely to subscribe before a campaign is planned. We assume those retained inputs can be known then; the historical dataset cannot independently prove the timing of every value.

`y` is the **target**: the answer we want the model to predict. The customer attributes are **features**: the information the model uses. The outcome is a category, so this is classification rather than regression.

```python
df = pd.read_csv("data/dataset.csv", sep=";")
```

`read_csv` loads a table into a pandas DataFrame named `df`. `sep=";"` matters because this file separates columns with semicolons. Using the wrong separator would produce an incorrectly shaped table.

The audit uses:

| Code | Meaning |
|---|---|
| `df.shape` | Number of rows and columns |
| `df.head(3)` | First three rows, for a quick visual check |
| `df.isna().sum().sum()` | Total ordinary missing cells |
| `df.duplicated().sum()` | Number of exact repeated rows |
| `df.drop_duplicates(keep="first")` | Keep the first occurrence of each exact row |
| `.copy()` | Make a separate table to modify without changing the source table |

The file starts with 41,188 rows. Removing 12 exact duplicates leaves 41,176. It has 21 columns: 20 original inputs and one target. There are no ordinary nulls, but the text `unknown` still means information is unavailable.

This does not prove that every remaining row describes a different customer. Without IDs, we cannot check whether the same person appears in multiple records.

## 5. Notebook section 2: fitting, validation and test

Think of three different roles:

- **Fitting data:** the material the algorithm learns from.
- **Validation data:** the practice assessment used to choose the algorithm and its settings.
- **Test data:** the final assessment after those choices have been made.

```python
development, test = train_test_split(df, test_size=0.20, shuffle=False)
training, validation = train_test_split(development, test_size=0.25, shuffle=False)
```

`shuffle=False` preserves row order. The file is documented as date-ordered, so the model learns from earlier records and is assessed on later ones.

| Partition | Records | Use |
|---|---:|---|
| Fitting (`training`) | 24,705 | Fit each candidate during selection |
| Validation | 8,235 | Compare algorithms and hyperparameters |
| Development | 32,940 | Fitting plus validation; refit the winner here |
| Test | 8,236 | Final evaluation only |

Development is not a fourth separate sample. It contains fitting and validation. The overall proportions are approximately 60%, 20% and 20% because `0.8 × 0.75 = 0.6` and `0.8 × 0.25 = 0.2`.

The `assert` statements check that row indices do not overlap across partitions. They do not establish that customers never repeat.

Earlier EDA examined the entire development partition. The notebook acknowledges this instead of pretending validation had never been seen. No model choice was based on test outcomes. A single validation period is still a limited estimate, especially when the population changes over time.

### Why subscriber percentages differ between sections

| Data scope | Subscriber percentage |
|---|---:|
| Raw full file, before duplicate removal | 11.27% |
| Development: fitting plus validation | 6.38% |
| Fitting subset | 4.81% |
| Validation subset | 11.07% |
| Final test subset | 30.83% |

These are different parts of a chronological dataset, not contradictory calculations. Test prevalence was inspected during final evaluation, after selection. You should not shuffle again to obtain more attractive scores: that would change the question being evaluated.

### Why some inputs are excluded

`duration` reveals how long the call lasted, which cannot be known before it happens. Using it would be prediction-time leakage.

The notebook also excludes current-campaign counts, last-contact channel/date fields and economic indicators. They are not verified campaign-planning snapshots. This is a conservative modelling choice, not a claim that such features are always unusable. With proper timestamped records, some could be suitable in a different project.

## 6. Notebook section 3: understanding the EDA code

```python
eda["subscribed"] = eda["y"].eq("yes").astype(int)
```

`.eq("yes")` checks whether each row says yes. `.astype(int)` converts `True` to 1 and `False` to 0. The mean of a 0/1 column is the proportion of positive rows.

`value_counts()` counts each label. For categorical fields, `.eq("unknown").mean().mul(100)` calculates the percentage of unknown values.

The first figure shows class counts and missing-information percentages. Its practical lesson is that an always-no classifier looks accurate while finding no subscribers. Unknown categories should be handled deliberately instead of being missed by `isna()`.

The numeric summary temporarily replaces `pdays=999` with `NaN` so it can describe actual intervals separately. That replacement affects the summary table only; it is not the model's transformation.

```python
rates = eda.groupby(col)["subscribed"].agg(["size", "mean"])
```

`groupby` puts records with the same category together. `size` counts them; `mean` calculates their response rate. A `for` loop draws the same kind of figure for `job` and `poutcome`, avoiding duplicated code.

Sample sizes matter: a high rate in a small group is less persuasive than a similar rate supported by many records. Prior success has a 37.50% rate in development, but only 168 examples. Only six fitting records have a non-sentinel `pdays` value, so history patterns are especially poorly represented in the earliest data.

`np.array_split` divides the ordered development rows into five equal-sized groups. The final EDA plot shows how response rates change across them. These groups are not exact months or equal-duration periods.

The EDA does **not** establish that being a student causes subscription, or that the bank should target a group without further checks.

## 7. Notebook section 4: preprocessing and feature engineering

### The simple feature function

`prepare_features(data)` is a small function: it takes a table, selects the approved columns, adds a history indicator and returns the result. Naming it once lets the notebook apply exactly the same rules to every partition.

```python
X["previously_contacted"] = X["pdays"].ne(999).astype(int)
X["pdays"] = X["pdays"].replace(999, 0)
```

`.ne(999)` means “not equal to 999”. The indicator keeps the meaning of the original sentinel:

| Original `pdays` | New `pdays` | `previously_contacted` |
|---:|---:|---:|
| 999 | 0 | 0 |
| 0 | 0 | 1 |
| 6 | 6 | 1 |

Both first rows now have a numeric interval of zero, but the flag distinguishes them. This is a basic example of feature engineering: representing information in a more useful way.

`X` conventionally means predictors. `y` means the target. The target mapping is fixed: `{"no": 0, "yes": 1}`. The target and all excluded fields are absent from `X`.

### StandardScaler

Standardisation subtracts the fitting mean and divides by the fitting standard deviation. The notebook applies it to age, previous contact count and the adjusted day interval.

Without scaling, a large numerical range can dominate KNN distances and affect the strength of logistic regularisation. Standardisation changes the units; it does **not** make an arbitrary distribution normally distributed or remove outliers.

During selection, the validation data use the means and standard deviations learned from fitting data. They are not scaled using their own statistics.

### OneHotEncoder

Models need numeric inputs. One-hot encoding represents categories with indicator columns, avoiding an invented ranking such as `admin=1`, `student=2`, `retired=3`.

- `drop="first"` omits one reference category from each field. A job coefficient can then be read relative to the omitted job.
- `handle_unknown="ignore"` lets a later unseen category be represented without an exception. It produces all zeros for that field, which can resemble the reference category; that is a limitation.
- The literal value `unknown` is a learned category when it appears in fitting data. It is different from a category that the encoder has never encountered.

The same encoding is used for both algorithms. Dropping a reference category also changes distances for KNN. This is one reason not to claim KNN has been exhaustively or optimally configured.

### ColumnTransformer and Pipeline

`ColumnTransformer` applies different instructions to different columns: scale numeric values, encode categories and pass the binary flag through unchanged.

`Pipeline` places that transformer before the classifier. Calling `fit` learns the transformations and model in order. Calling `predict_proba` applies those already learned transformations before generating scores. These are standard scikit-learn tools used to enforce the lecture's rule against learning from validation or test data. [Leakage guidance](https://scikit-learn.org/1.6/common_pitfalls.html).

No numeric imputation is needed for this particular file. The prototype would need a defined policy for genuinely missing future inputs before deployment. The encoder uses its default sparse representation where appropriate; you do not need to manually manage sparse matrices.

## 8. Notebook section 5: models and tuning

### Logistic regression

Despite its name, logistic regression is a classifier here. It learns a weighted sum of encoded features, then applies the sigmoid function from the lecture to produce a score between 0 and 1.

**Learned parameters** are coefficients and the intercept. **Hyperparameters** are settings chosen before fitting, such as `C` and the class-weighting option.

`C` controls inverse regularisation strength: **smaller C means stronger regularisation** in scikit-learn. `max_iter=1000` is an optimisation iteration limit, not a number of models or epochs that must all be used. `class_weight="balanced"` gives more weight to the rare class during fitting; it does not create synthetic rows or force predictions to be 50/50. [LogisticRegression documentation](https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.LogisticRegression.html).

### K-nearest neighbours

KNN stores fitting examples. For a new record, it finds `k` nearby profiles in the transformed feature space. This notebook uses Euclidean distance and equal votes. The subscriber score is the proportion of those neighbours labelled yes.

With `k=5`, three yes votes produce a score of `3/5=0.6`, so the record is predicted yes at the fixed 0.5 threshold. With a rare positive class, most neighbourhoods can vote no even if headline accuracy looks high.

Larger `k` averages over more neighbours and usually smooths decisions. It does not guarantee better results. One-hot dimensions, mixed-feature distances and changing populations can weaken similarity. [KNeighborsClassifier documentation](https://scikit-learn.org/1.6/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).

### The grid search is an ordinary loop

The notebook evaluates:

- Six logistic settings: three values of `C`, each with two weighting choices.
- Four KNN settings: `k=5, 15, 25, 51`.
- Two baselines: everyone no, and everyone yes.

The `candidates` dictionary connects each readable model name with a classifier. The loop builds a pipeline, fits it, predicts validation scores and saves its metrics. A dictionary stores a named collection of values; a list stores the rows of the results table.

```python
model.fit(X_train, y_train)
probability = model.predict_proba(X_valid)[:, 1]
```

`fit` learns from training examples. `predict_proba` returns two columns, one for each class. Because targets are mapped to 0 and 1, `[:, 1]` selects the score for the subscriber class.

`evaluate` converts scores to predictions using `>=0.5`, then calculates the metrics. `**evaluate(...)` inserts that returned metric dictionary into the row containing the model name; it is ordinary Python dictionary unpacking, not an ML technique.

The results are sorted by F1 without rounding first. `round(4)` changes only the displayed precision. The top row determines the selected model. The named hyperparameters are selected on validation data; the test set is not used here.

### What the validation results say

The winner is **logistic regression, C=0.1, balanced class weights**:

| Validation measure | Result |
|---|---:|
| Precision | 0.1391 |
| Recall | 0.6349 |
| F1 | 0.2283 |
| ROC-AUC | 0.6071 |

The matching unweighted model has F1 0.0253 but slightly higher ROC-AUC. Weighting therefore helps the chosen F1 objective, not every aspect of performance. Other balanced values of C have almost identical F1; the evidence does not establish a meaningful difference between them.

All tested KNN settings have zero validation F1. This does not mean KNN is universally useless. It means these settings, this representation, this time split and this threshold did not capture subscribers successfully. We report the failure instead of discarding it from the comparison.

The selected model's fitting F1 is 0.0993. Validation is higher, but its subscriber prevalence and period differ. You cannot diagnose overfitting by comparing these two numbers as though the populations were identical.

## 9. Notebook section 6: evaluation and the meaning of the scores

The winner is refitted on **all development data**. Its hyperparameters stay fixed, but its coefficients and preprocessing statistics are learned again using more observations. That is the refit-then-test process in the lecture.

`final_model = selected` gives the selected pipeline another variable name; it is not a copy. Calling `fit` replaces its fitted state. The earlier validation metrics have already been saved separately.

The notebook then scores the latest test records once as a final assessment. It does not change the model afterwards to improve the displayed result.

### Confusion matrix: the actual counts

| Actual outcome | Predicted no | Predicted yes |
|---|---:|---:|
| Did not subscribe | 2,308 true negatives | 3,389 false positives |
| Subscribed | 564 false negatives | 1,975 true positives |

`confusion_matrix(...).ravel()` turns the 2-by-2 table into `tn, fp, fn, tp` in that order. The positive label is subscription.

| Metric | Meaning | Test result |
|---|---|---:|
| Accuracy | Fraction of all predictions that are correct | 52.00% |
| Precision | Fraction of flagged records that are subscribers: `TP/(TP+FP)` | 36.82% |
| Recall | Fraction of subscribers captured: `TP/(TP+FN)` | 77.79% |
| F1 | Harmonic mean of precision and recall | 0.4998 |
| ROC-AUC | How well scores rank subscribers above non-subscribers | 0.6837 |

F1 is `2 × precision × recall / (precision + recall)`. It is a useful compromise here because both wasted contacts and missed subscribers matter, while actual financial costs are unavailable. It is not a direct calculation of business value.

An always-no model has higher test accuracy (69.17%) but captures no subscribers. An always-yes model captures all subscribers and already obtains F1 0.4713. Comparing both baselines prevents us from exaggerating the winner's advantage.

The ROC curve varies a hypothetical threshold to show true-positive versus false-positive rates. Drawing this curve does not change the actual 0.5 decision threshold. AUC 0.684 is not “68.4% accuracy”, and the probability-like scores are not guaranteed calibrated probabilities, particularly after class weighting.

Test F1 is higher than validation F1 partly in a setting with much higher subscriber prevalence, and the model was refitted on later development records. The comparison does not prove it will keep improving on future campaigns.

## 10. Notebook section 7: explaining coefficients

The selected model is logistic regression, so the notebook can inspect `coef_`. This is the collection of learned feature weights. `get_feature_names_out()` retrieves the names after encoding; names beginning `category__` are category indicators, and `numeric__` identifies scaled numeric fields.

The code selects the ten largest **absolute** coefficients and then plots their signed values. Absolute value selects large effects in either direction; the sign shows whether the term increases or decreases the fitted score.

Reference categories are:

| Field | Reference |
|---|---|
| Job | `admin.` |
| Marital status | `divorced` |
| Education | `basic.4y` |
| Default, housing, loan | `no` |
| Previous campaign outcome | `failure` |

For example, the student job coefficient is about **+0.927** relative to `admin.`, and prior success is about **+0.908** relative to failure. These are changes in the linear **log-odds** score, not percentage-point increases in subscription probability. The sigmoid maps the combined score to a probability-like value.

The history flag has a coefficient around +1.033; unknown default information is around -0.415. Numeric coefficients instead refer to one development standard deviation. Different feature units and overlapping history information mean the chart should not be read as a definitive ranking of business importance.

Rare groups can get unstable weights. Changing someone's job or past history is not a demonstrated intervention that would cause a subscription. The coefficient section explains the particular selected logistic model; if future edits select a different model family, that section must also be adapted.

## 11. Notebook section 8: the business conclusion

At the fixed threshold, the model flags **5,364 of 8,236 test records**, about **65.13%**. Those records include **1,975 of 2,539 subscribers**, about **77.79%**. It leaves 2,872 records unflagged and misses 564 observed subscribers.

The historical response rate among flagged records is 36.82%, compared with 30.83% overall: approximately six percentage points higher. That is a modest concentration of subscribers. It does not demonstrate the number of calls saved, unique people targeted, new subscriptions caused, or profit earned.

The recommendation is **not to deploy automatically**. Reasons include the weak validation discrimination, substantial change between periods, many false positives, old data, missing customer IDs, uncertain feature timestamps and unknown operating costs.

The next business investigation would obtain current data, verify feature availability and contact eligibility, compare subgroup outcomes, estimate costs, and then consider a controlled pilot against the existing campaign policy. Any pilot is a recommendation only; no bank system has been deployed or contacted.

This is critical evaluation: explaining what the evidence supports and what it does not. A complete assignment does not require pretending that the model is excellent.

## 12. What to check before submission

The ML notebook is complete and its outputs were executed. You should still review it in your own words and confirm administrative requirements:

- Confirm the current Canvas deadline; the supplied brief's September 2025 date is old.
- Confirm this dataset was not used in any module exercise not included in the local files.
- Confirm the university's exact character-counting rule. The reported count covers code plus Markdown source; HTML bytes and embedded image data are a different measure.
- Check the Harvard reference format against the institution's guide and replace `n.d.` if lecture dates are known.
- Check the AI-use disclosure against the tutor's instructions.
- Open the exported HTML and verify that the notebook, figures and references are readable.

The AI Cup agent, online assessments and participation are separate module components. They are not completed by this bank-marketing notebook. The files have not been submitted to Canvas.

## 13. Questions you should be able to answer

1. Why is this classification rather than regression?
2. Why is current-call duration excluded?
3. Why preserve chronological order rather than shuffle?
4. How do development, fitting, validation and test differ?
5. Why can an always-no model have high accuracy but zero value for finding subscribers?
6. Why is `pdays=999` handled with a flag?
7. Why encode categories and standardise numeric inputs?
8. What do C, class weights and k control?
9. How was the winner chosen without test-set tuning?
10. What do 36.82% precision and 77.79% recall mean for this business?
11. Why does a positive coefficient not prove causation?
12. Why is the final recommendation cautious despite the improved test F1?

Every answer is supported by the notebook results and the lecture connections above. Use the guide to understand the reasoning, rather than memorising isolated lines of Python.
