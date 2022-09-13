## Customer churn

* From kaggle sliced competition s1 e07
* used xgboost to predict bank customer churn (leaves bank or not)
  * Evaluated via LogLoss - my private score was 0.10121

### Data dictionary

*'id': unique identifier for the customer

*'attrition_flag': whether the customer is churned (0 = no; 1 = yes)

*'customer_age': age of the customer

*'gender': gender of the customer

*'education_level': education level of the customer

*'income_category': income range of the customer

*'total_relationship_count': number of relationships

*'months_inactive_12_mon': number of months the customer is inactive in the past 12 months

*'credit_limit': the customer's credit limit

*'total_revolving_bal': the customer's total revolving balance

*'total_amt_chng_q4_q1': the amount the balance changed from Q4 to Q1

*'total_trans_amt': the value of all the customer's transactions in the period

*'total_trans_ct': the number of all of the customer's transactions

*'total_ct_chng_q4_q1': the difference in number of the customer's transactions from Q4 to Q1

*'avg_utilization_ratio': the customer's average utilization ratio during the period
