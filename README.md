# credit-risk-automation-aws

# TL;DR:

Fully automated loan-risk prediction pipeline using Lending Club data. Mirrors how banks and fintechs ingest raw loan files, preprocess features, train risk models, and generate actionable approval strategies. Built entirely on AWS using S3 and SageMaker.


## Real-World Problem Context
Banks and fintech lenders process thousands of loan applications every day. Manual underwriting is slow, inconsistent, and costly. Modern lenders automate three critical steps:

1. Ingest raw applicant data from internal systems or partner platforms  
2. Run the data through a risk engine that handles cleaning, feature preparation, and model scoring  
3. Generate approval strategies aligned with institutional risk appetite  

**credit-risk-automation-aws** simulates this exact pipeline end-to-end on AWS: raw CSV ingestion, automated preprocessing, XGBoost credit-risk modeling, and strategy dashboards used for lending decisions.

## Key Highlights
* End-to-end loan risk automation: CSV → S3 → SageMaker → predictions → dashboards  
* Handles real messy borrower data: missing values, categorical encoding, inconsistent formats  
* XGBoost binary classifier for predicting loan default risk  
* Batch scoring pipeline outputs actionable approval lists  
* Two lending strategies: conservative (lower default rate) and optimal (higher approvals)  
* Architecture mirrors real-world lending risk systems

## Table of Contents
1. [Cost](#cost)  
2. [Project Goal](#project-goal)  
3. [Datasets](#datasets)  
4. [Architecture](#architecture)  
5. [Setup](#setup)  
   * [S3](#s3)  
   * [SageMaker](#sagemaker)  
6. [Usage](#usage)  
7. [Results](#results)  
8. [Next Steps / Extensions](#next-steps--extensions)  
9. [Acknowledgments](#acknowledgments)

## Cost
All AWS services used (S3 and SageMaker) are Free Tier eligible.  
Delete unused notebooks, artifacts, and endpoints after execution to avoid charges.

## Project Goal
Implement a reproducible credit-risk automation pipeline that:

* ingests raw Lending Club CSV files into S3  
* preprocesses and structures data for modeling  
* trains an XGBoost model to classify loan default risk  
* produces borrower approval lists for lending strategy decisions  
* stores model artifacts, predictions, and dashboards back into S3  

This represents the core of an automated underwriting engine used by modern financial institutions.

## Datasets
**Source:** [Lending Club Loan Data](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv)

* **Size:** ~260,722 rows after preprocessing  
* **Important fields**  
  * `loan_status` (target: Fully Paid = 0, Charged Off = 1)  
  * `loan_amnt`, `term`, `int_rate`, `grade`, `emp_length`  
  * Payment history, credit behavior, and borrower attributes  

The dataset’s scale, imbalance, and inconsistencies make it representative of production lending environments.

## Architecture

<img width="506" height="335" alt="image" src="https://github.com/user-attachments/assets/851e51ef-0418-45f4-868f-82c51f969071" />


**Flow Explanation**  
* **S3 (Raw Data):** Acts as the data ingestion layer similar to internal bank data lakes  
* **SageMaker Notebooks:** Perform cleaning, preprocessing, and feature engineering  
* **SageMaker Training (XGBoost):** Trains the credit-risk model  
* **SageMaker Batch Transform:** Generates predictions at scale  
* **Dashboards:** Provide credit policy teams with approval strategies and expected default rates  
* **Loop-Back to S3:** All outputs stored for audit, regulatory review, and future retraining  

This architecture follows the same pattern used by underwriting engines in banks, consumer lenders, and fintech loan originators.

## Setup
### S3
SageMaker Studio automatically manages project buckets.

**Buckets used & screenshots:**
<img width="1277" height="467" alt="image" src="https://github.com/user-attachments/assets/4afbd278-0d5c-421d-83f4-daab28ac3607" />

**Raw input bucket**
<img width="1918" height="513" alt="image" src="https://github.com/user-attachments/assets/5449b7c1-9248-42c8-a0c5-accefddf23e5" />

**Processed data bucket**
<img width="1912" height="525" alt="image" src="https://github.com/user-attachments/assets/bf7d9285-b613-4ff6-ba95-bc5170b6e5ae" />

**Folders**
<img width="1919" height="660" alt="image" src="https://github.com/user-attachments/assets/145d83d2-32d4-43eb-8a08-7644c4c2c488" />

**predictions/folder**
<img width="1919" height="717" alt="image" src="https://github.com/user-attachments/assets/15e126d5-4c21-48fa-960c-02a6bd92def6" />

**training_code/folder(Python code goes here)**
<img width="1918" height="480" alt="image" src="https://github.com/user-attachments/assets/c7a4156b-c8f3-46fd-97c1-0be9eef50466" />

**visualizations/folder**
<img width="1911" height="643" alt="image" src="https://github.com/user-attachments/assets/15c0d385-dcdb-4758-9fc4-00fe336d1a61" />

**json files**
<img width="1917" height="565" alt="image" src="https://github.com/user-attachments/assets/90c06b01-f803-45fb-aa97-5ec1faacd0c7" />

**Folders contain**  

* Raw data
  
* Preprocessed datasets

* Model artifacts
  
* Batch predictions
    
* Final dashboards
  

### SageMaker
Ran each notebook in sequence:

1. `1_data_inspection.ipynb`  
   Explore missing values, distribution shifts, correlations.

2. `2_data_preprocessing.ipynb`  
   Clean inconsistencies, encode categorical features, export final training data.

3. `3_xgboost_training.ipynb`  
   Train the binary classifier using SageMaker’s built-in XGBoost.

4. `4_predictions_and_evaluation.ipynb`  
   Run batch inference, compute probabilities, generate evaluation metrics.

5. `5_final_dashboard.ipynb`  
   Produce strategy thresholds, approval lists, and business dashboards.

All generated artifacts and CSVs are saved back into S3.

## Usage
* Load and preprocess the Lending Club dataset  
* Train the XGBoost binary classifier (Charged Off vs Fully Paid)  
* Score the dataset using batch transform  
* Compute borrower default probabilities  
* Apply decision thresholds for lending strategies  
* Export risk-stratified borrower lists:

## Results

**Dashboard visuals summarize the financial impact of each strategy**

**probability_distribution**  
<img width="4168" height="1772" alt="graph1_probability_distribution" src="https://github.com/user-attachments/assets/f9c3f0eb-398f-405a-9a72-6f6a98492fed" />

**approval_volume**
<img width="2969" height="1773" alt="graph2_approval_volume" src="https://github.com/user-attachments/assets/421763de-3c6b-4fe2-a688-6f9f554bb2b7" />

**default_rate**
<img width="2968" height="1773" alt="graph3_default_rate" src="https://github.com/user-attachments/assets/14d2189b-f8ed-4d61-adf3-30424b33f37d" />

**risk_vs_reward**
<img width="6557" height="2383" alt="graph4_risk_vs_reward" src="https://github.com/user-attachments/assets/1e539934-8c66-44c1-89cd-e9f231c7dd47" />

These outputs reflect real credit-policy trade-offs between approval volume and default risk.

### Model Performance
* **AUC-ROC:** 0.70 on held-out data  
* Well calibrated for threshold optimization

### Strategy Outcomes
**Ultra-Safe (<50% probability of default)**  
* Approved: 257,638  
* Expected default rate: 19.75%

**Optimal (<35% probability of default)**  
* Approved: 227,678  
* Expected default rate: 16.88%

## Next Steps / Extensions
* Add real-time inference for instant loan approval flows  
* Build a SageMaker Pipeline for fully automated retraining  
* Integrate Feature Store for production-grade feature consistency  
* Add model drift monitoring and alerts  
* Use A/B lending strategies for risk-reward experimentation  
* Scale to full Lending Club dataset (~2M+ rows)

## Acknowledgments
* [Lending Club](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv)  
* AWS S3 and SageMaker components  
* Project aligned to responsible, transparent lending practices

---
**credit-risk-automation-aws** — Approve more. Lose less.  
*Because banks shouldn't have to choose between growth and safety.*
