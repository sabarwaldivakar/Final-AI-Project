# B104 project roadmap: completion record

**Updated:** 13 September 2026  
**Project:** Predicting Bank Term Deposits

The end-to-end machine-learning notebook is complete. Following your request, the implementation uses logistic regression, KNN, a simple chronological holdout and a manual parameter grid based on the supplied lectures.

## Deliverables

- [Executed notebook](</Users/divakar/Desktop/AI project/Bank_management2.ipynb>)
- [HTML export for review/submission](</Users/divakar/Desktop/AI project/Bank_management2.html>)
- [Code and lecture study guide](</Users/divakar/Desktop/AI project/B104_Code_and_Lecture_Guide.md>)
- [Tested package versions](</Users/divakar/Desktop/AI project/requirements.txt>)
- [Earlier versions and original roadmap](</Users/divakar/Desktop/AI project/backups>)

## Completed stages

- [x] Business question, target and dataset provenance.
- [x] Duplicate audit and chronological data separation.
- [x] Development-set EDA, unknown categories, class imbalance and changing response rates.
- [x] Conservative campaign-planning feature selection.
- [x] One-hot encoding, numerical scaling and a prior-contact indicator.
- [x] Logistic regression and KNN comparison, with always-no/always-yes baselines.
- [x] Manual grid: six logistic settings and four KNN settings.
- [x] Model selected by validation F1 at a fixed 0.5 threshold.
- [x] Refit on development data and final evaluation on reserved test records.
- [x] Coefficient explanation, business implications and limitations.
- [x] Harvard-style references and AI-assistance disclosure.
- [x] Notebook executed, plots inspected and HTML exported.

The initial roadmap's random forest, expanding-window validation, average precision and permutation-importance extensions were replaced with the simpler lecture-based approach. The original plan is backed up; it no longer describes the final implementation.

## Result

Selected model: **logistic regression, C=0.1, balanced class weights**.

| Measure | Validation | Test |
|---|---:|---:|
| Precision | 0.1391 | 0.3682 |
| Recall | 0.6349 | 0.7779 |
| F1 | 0.2283 | 0.4998 |
| ROC-AUC | 0.6071 | 0.6837 |

The test model flags 5,364 records and captures 1,975 subscribers, while missing 564 and flagging 3,389 non-subscribers. Performance is modest relative to simple baselines, and class proportions change substantially between periods. The recommendation is **no automatic deployment**: obtain recent, timestamped data and assess costs and subgroup outcomes before considering a controlled pilot.

## Verification and remaining review

The notebook contains **19,486 characters of code and Markdown source**, excluding outputs and notebook metadata. All code cells executed successfully; the final saved outputs contain no error or warning streams. Checks confirmed separate partitions, finite model values and scaling statistics learned from development rather than test data. All five figures are embedded in the HTML.

The browser's security policy blocked opening the local HTML file for an in-browser preview. The exported document was instead checked structurally, and the notebook's plot images were inspected directly. Open the HTML locally for a final visual review.

- [ ] Review the code and explanations using the study guide.
- [ ] Confirm the official character-counting rule, current deadline and dataset eligibility.
- [ ] Check institution-specific Harvard formatting and AI-disclosure requirements.
- [ ] Open the HTML and submit it through the correct Canvas folder when ready.

The notebook has not been submitted. AI Cup performance, online assessments and class participation are separate assessment components and remain outside this bank-marketing project.
