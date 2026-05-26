# Mastercard Hidden Entrepreneur Solution

## 1. Objective
Build an explainable, production-friendly classifier that flags whether a card behaves like:
- `0`: regular consumer
- `1`: hidden entrepreneur

Business cards were used as proxy positive labels and consumer cards as proxy negative labels.

## 2. Data and cleaning
- Observation window: 2025-10-01 00:00:00 to 2026-03-31 23:59:53
- Business transactions: 2,997,593
- Consumer transactions: 9,832,487
- Business cards: 25,000
- Consumer cards: 80,000
- Merchant reference coverage after merge: 100%
- Exact duplicate rows removed: business=0, consumer=0
- Non-positive amount rows removed: business=0, consumer=0

## 3. Leakage prevention
The pipeline intentionally excludes:
- `card_tier` because business cards are directly marked as `Business`
- `bank_name` because it can capture issuer-specific bias instead of transactional behavior
- raw `merchant_name` / `merchant_id` from the model to avoid merchant memorization

Instead, the model uses stable aggregated behavioral signals at `card_number` level.

## 4. Engineered card-level features
Mandatory features included:
- transaction behavior: counts and amount statistics
- merchant behavior: merchant diversity and top merchant concentration
- MCC behavior
- time behavior: hour, night, weekend
- digital behavior: online, tokenized, recurring
- geography: international share and country diversity
- velocity: average transactions per observed day

Additional explainable enrichment after merge:
- `recurring_capable_ratio`: share of transactions at merchants capable of recurring billing

## 5. Model comparison
```
              model  precision  recall  f1_score  roc_auc    tn  fp  fn   tp
      Random Forest     1.0000  0.9998    0.9999      1.0 16000   0   1 4999
Logistic Regression     0.9996  1.0000    0.9998      1.0 15998   2   0 5000
            XGBoost     0.9998  0.9992    0.9995      1.0 15999   1   4 4996
```

## 6. Primary model: XGBoost
- Precision: 0.9998
- Recall: 0.9992
- F1-score: 0.9995
- ROC-AUC: 1.0000
- Confusion matrix: TN=15999, FP=1, FN=4, TP=4996

## 7. Business meaning of model errors
- False Positive: a normal consumer is flagged as entrepreneur. Business impact: unnecessary outreach, friction, or over-monitoring.
- False Negative: a hidden entrepreneur stays in consumer segment. Business impact: missed migration to business products, underpriced servicing, and weaker portfolio intelligence.

If business goal is acquisition of micro-business clients, recall can be prioritized by lowering the threshold.
If business goal is minimizing friction for true consumers, precision should remain the main KPI.

## 8. Key business signals found in the data
```
                feature  consumer_mean  business_mean  business_vs_consumer_pct
        recurring_ratio         0.0314         0.1547                  392.6752
recurring_capable_ratio         0.0740         0.3363                  354.4595
             avg_amount     54770.1359    167642.4478                  206.0837
     top_merchant_ratio         0.1607         0.3678                  128.8737
           online_ratio         0.4687         0.8484                   81.0113
    international_ratio         0.2503         0.3122                   24.7303
              txn_count       122.9061       119.9037                   -2.4428
   avg_transaction_hour        14.9289        11.6863                  -21.7203
       unique_merchants        36.7505        17.0873                  -53.5046
 weekend_activity_ratio         0.3452         0.1291                  -62.6014
```

Observed pattern in this dataset:
- hidden entrepreneurs are much more online-heavy
- they transact earlier in the day and have materially higher night activity
- they have higher recurring behavior and higher concentration on a top merchant
- they have much larger ticket sizes
- they are slightly more international
- merchant diversity is lower, not higher, which suggests concentration in a narrow set of business-service merchants rather than broad retail spend

## 9. Top feature importance
- `avg_transaction_hour`: consumer mean=14.9289, business mean=11.6863
- `online_ratio`: consumer mean=0.4687, business mean=0.8484
- `weekend_activity_ratio`: consumer mean=0.3452, business mean=0.1291
- `top_merchant_ratio`: consumer mean=0.1607, business mean=0.3678
- `tokenized_ratio`: consumer mean=0.3848, business mean=0.5846
- `recurring_ratio`: consumer mean=0.0314, business mean=0.1547
- `total_amount`: consumer mean=6642511.7084, business mean=18769161.7454
- `recurring_capable_ratio`: consumer mean=0.0740, business mean=0.3363

## 10. Example: why the model thinks a card is a hidden entrepreneur
Card: `5486023773118595`
- Predicted probability: 1.0000
- Actual label: 1
- Predicted label: 1
- `tokenized_ratio`=0.6140, SHAP=4.1100, pushes_to_hidden_entrepreneur
- `online_ratio`=0.9123, SHAP=2.3245, pushes_to_hidden_entrepreneur
- `avg_transaction_hour`=11.4386, SHAP=2.2903, pushes_to_hidden_entrepreneur
- `weekend_activity_ratio`=0.1404, SHAP=1.3436, pushes_to_hidden_entrepreneur
- `top_merchant_ratio`=0.3596, SHAP=0.5689, pushes_to_hidden_entrepreneur

## 11. Presentation storyline ideas
1. Start from the business problem: consumer cards used for business-like spend create pricing and servicing blind spots.
2. Show data design and leakage controls first to establish model credibility.
3. Use 4-5 business signals with simple charts: online ratio, avg amount, recurring ratio, top merchant concentration, transaction hour.
4. Compare Logistic Regression, Random Forest, and XGBoost to show why gradient boosting is chosen as the operational model.
5. Add one SHAP slide: explain a single flagged customer in plain business language.
6. End with an action slide: score consumer portfolio, review top-risk cards, route strongest cases to business card migration campaigns.

## 12. Recommendations
1. Validate on an out-of-time sample before production because the current dataset is very separable and likely semi-synthetic.
2. Add threshold tuning by business objective: growth, compliance review, or customer experience.
3. Expand labels beyond business cards vs consumer cards by using confirmed SME onboarding or merchant-linked ground truth.
4. Monitor drift in online share, time-of-day behavior, and recurring ecosystems after launch.
