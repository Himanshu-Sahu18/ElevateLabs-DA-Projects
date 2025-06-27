# Customer Lifetime Value (LTV) Prediction Model

This project implements a machine learning model to predict customer lifetime value based on purchase behavior. The model uses advanced feature engineering and multiple machine learning algorithms to provide accurate LTV predictions and customer segmentation.

## Project Structure

- `ltv_prediction.py`: Main script containing the LTV prediction model
- `online_retail.csv`: Input dataset containing customer purchase history
- `requirements.txt`: Python dependencies
- `output/`: Directory containing model outputs and visualizations

## Model Architecture

### 1. Data Preprocessing

- Handles missing values and outliers
- Removes returns (negative quantities)
- Processes date formats
- Calculates transaction amounts
- Removes statistical outliers using IQR method

### 2. Feature Engineering

The model uses several sophisticated features:

#### Basic RFM Features

- **Recency**: Days since last purchase
- **Frequency**: Number of transactions
- **Monetary**: Total amount spent

#### Advanced Features

- **Customer Tenure**: Length of customer relationship
- **Purchase Frequency**: Transactions per year
- **Average Order Value**: Average transaction amount
- **Product Diversity**: Number of unique products purchased
- **Purchase Regularity**: Ratio of unique purchase dates to total purchases
- **Monetary Per Day**: Average daily spending
- **Transaction Value Statistics**: Mean and standard deviation
- **Quantity Statistics**: Total and average quantities

### 3. LTV Calculation

The model uses a sophisticated LTV calculation:

```
LTV = (Average Order Value × Purchase Frequency × Customer Lifespan) × Profit Margin
```

Where:

- Profit Margin is set to 20%
- Customer Lifespan is calculated from tenure
- Purchase Frequency is normalized to yearly basis

### 4. Machine Learning Models

The project implements two models:

#### XGBoost

- Parameters:
  - n_estimators: 200
  - learning_rate: 0.1
  - max_depth: 6
  - min_child_weight: 1
  - subsample: 0.8
  - colsample_bytree: 0.8

#### Random Forest

- Parameters:
  - n_estimators: 200
  - max_depth: 10
  - min_samples_split: 5
  - min_samples_leaf: 2

### 5. Model Performance

Latest Results:

- XGBoost:
  - MAE: $5.40
  - RMSE: $41.76
  - R² Score: 0.9753
- Random Forest:
  - MAE: $1.41
  - RMSE: $9.90
  - R² Score: 0.9986

### 6. Customer Segmentation

Customers are segmented into four tiers based on predicted LTV:

- **Platinum**: Top 25% of customers
- **Gold**: 25-50th percentile
- **Silver**: 50-75th percentile
- **Bronze**: Bottom 25% of customers

## Setup Instructions

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the model:

```bash
python ltv_prediction.py
```

## Output Analysis (Kindly check the output folder)

### 1. Model Outputs

- `customer_ltv_predictions.csv`: Contains:
  - Original RFM features
  - Advanced features
  - Predicted LTV from both models
  - Customer segments from both models

### 2. Visualizations

The model generates several visualizations in the `output/` directory:

#### Feature Importance

- Shows the relative importance of each feature
- Helps understand what drives customer value
- Available for both XGBoost and Random Forest models

#### LTV Distribution

- Histogram of predicted LTV values
- Kernel Density Estimation (KDE) overlay
- Shows the distribution of customer value

#### Customer Segments

- Box plots showing LTV distribution by segment
- Helps identify value ranges for each segment
- Available for both models' predictions

#### Actual vs Predicted

- Scatter plot comparing actual vs. predicted LTV
- Perfect prediction line for reference
- Shows model accuracy

## Business Applications

### 1. Marketing Strategy

- Target high-value customers with premium offers
- Develop retention strategies for at-risk customers
- Optimize marketing spend based on customer value

### 2. Customer Service

- Prioritize support for high-value customers
- Customize service levels based on customer segment
- Identify opportunities for upselling

### 3. Product Development

- Understand product preferences of high-value customers
- Develop products targeting specific segments
- Optimize product mix based on customer value

### 4. Pricing Strategy

- Develop segment-specific pricing
- Identify price sensitivity by segment
- Optimize pricing for maximum customer value

## Future Improvements

### 1. Model Enhancements

- Add time-series features
- Implement customer churn prediction
- Add demographic features
- Include seasonal patterns

### 2. Feature Engineering

- Add product category analysis
- Include customer interaction data
- Add website/app usage metrics
- Incorporate social media engagement

### 3. Business Integration

- Real-time prediction API
- Automated reporting system
- Integration with CRM systems
- Automated marketing triggers
