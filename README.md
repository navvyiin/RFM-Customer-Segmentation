# Corporate RFM Segmentation Suite

A comprehensive Streamlit-based analytics application designed to deliver advanced RFM segmentation, K-Means clustering, lifetime value forecasting, and churn risk modelling. Built for data-driven organisations seeking granular customer intelligence, segment profiling, and targeted marketing optimisation.

# Key Features
### 1. Automated RFM Analysis 
- Cleanses and transforms transactional data. 
- Computes Recency, Frequency, and Monetary values per customer. 
- Provides descriptive statistics and exploratory diagnostics.

### 2. Dynamic K-Means Segmentation 
- Interactive selection of cluster ranges and K values. 
- Elbow and Silhouette analysis for optimal cluster tuning. 
- Automated segment naming using behavioural z-scores.

### 3. Customer Lifetime Value (LTV) Forecasting 
- Uses frequency, order value, churn risk, and business horizon. 
- Applies profit margin assumptions to estimate long-term value. 
- Generates segment-level and customer-level LTV distributions.

### 4. Churn Probability Modelling 
- Logistic regression with fallback heuristic modelling. 
- Detects at-risk customers using recency thresholds. 
- Visualises churn distribution and segment-level risk.

### 5. Segment Intelligence Dashboard 
- Radar-style behavioural profiles for each segment. 
- Segment-level KPIs including LTV, churn, recency, and revenue. 
- Automated, context-aware marketing recommendations.

### 6. Executive-Level Reporting 
- Interactive dashboards for leadership and CRM teams. 
- Revenue distribution, customer distribution, and KPI summaries. 
- Exportable CSV outputs for downstream analytics or warehousing. 

### 7. Streamlit UI with Corporate Styling 
- Clean teal-white theme with KPI cards and responsive layouts. 
- Sidebar-driven controls for all analytical parameters. 
- Optimised for clarity, performance, and professional review.

# Data Requirements
Upload a CSV containing the following mandatory fields:

| Column Name   | Description                                   |
| ------------- | --------------------------------------------- |
| `CustomerID`  | Unique identifier for each customer           |
| `InvoiceNo`   | Transaction or order ID                       |
| `InvoiceDate` | Transaction timestamp (YYYY-MM-DD or similar) |
| `Quantity`    | Quantity purchased                            |
| `UnitPrice`   | Price per unit                                |

Optional files should not include personally identifiable information (PII).

# Project Structure
- main_rfm_segmentation.py - Main Streamlit application
- requirements.txt - Python dependencies
- README.md - Project documentation
- This is the workflow of the project - https://www.figma.com/board/tCxlM8oWazPmMYLDTl5XAx/RFM-Segmentation-Workflow?node-id=0-1&t=8R0FOKkxZ65CZ8hq-1

# Intended Users
- Marketing & CRM teams seeking actionable segmentation.
- Data analysts developing retention, upsell, and win-back strategies.
- Product leadership monitoring engagement cohorts.
- Business stakeholders needing evidence-based customer insights.

# Security & Compliance 
- The application performs local, in-session computation only.
- Users are responsible for ensuring uploaded datasets contain no PII.
- Suitable for sandbox, analysis, and non-production pipelines.

# License
This project can be licensed under MIT, Apache-2.0, or as preferred by your organisation.

# Contributions
Contributions, enhancements, and feature extensions are welcome.
Please open an issue or submit a pull request with context and proposed changes.
