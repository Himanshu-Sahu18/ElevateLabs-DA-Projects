import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

def load_and_preprocess_data(file_path):
    """Load and preprocess the retail data."""
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Convert InvoiceDate to datetime with correct format (DD-MM-YYYY)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d-%m-%Y %H:%M')
    
    # Calculate total amount for each transaction
    df['TotalAmount'] = df['Quantity'] * df['Price']
    
    # Remove returns (negative quantities) and outliers
    df = df[df['Quantity'] > 0]
    df = df[df['TotalAmount'] > 0]
    
    # Remove outliers (transactions with very high amounts)
    Q1 = df['TotalAmount'].quantile(0.25)
    Q3 = df['TotalAmount'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[df['TotalAmount'] <= (Q3 + 1.5 * IQR)]
    
    return df

def calculate_advanced_features(df, current_date):
    """Calculate advanced features for each customer."""
    print("Calculating advanced features...")
    
    # Basic RFM features
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': [
            lambda x: (current_date - x.max()).days,  # Recency
            lambda x: (x.max() - x.min()).days,  # Customer Tenure
            lambda x: len(x.unique())  # Number of unique purchase dates
        ],
        'Invoice': [
            'count',  # Frequency
            'nunique'  # Number of unique invoices
        ],
        'TotalAmount': [
            'sum',  # Monetary
            'mean',  # Average transaction value
            'std'  # Standard deviation of transaction values
        ],
        'Quantity': [
            'sum',  # Total quantity
            'mean'  # Average quantity per transaction
        ],
        'StockCode': 'nunique'  # Number of unique products purchased
    })
    
    # Flatten the multi-level columns
    rfm.columns = ['_'.join(col).strip() for col in rfm.columns.values]
    
    # Rename columns for clarity
    rfm = rfm.rename(columns={
        'InvoiceDate_<lambda_0>': 'Recency',
        'InvoiceDate_<lambda_1>': 'Tenure',
        'InvoiceDate_<lambda_2>': 'UniquePurchaseDates',
        'Invoice_count': 'Frequency',
        'Invoice_nunique': 'UniqueInvoices',
        'TotalAmount_sum': 'Monetary',
        'TotalAmount_mean': 'AvgTransactionValue',
        'TotalAmount_std': 'TransactionValueStd',
        'Quantity_sum': 'TotalQuantity',
        'Quantity_mean': 'AvgQuantity',
        'StockCode_nunique': 'UniqueProducts'
    })
    
    # Handle edge cases and calculate additional features
    rfm['Tenure'] = rfm['Tenure'].replace(0, 1)  # Replace 0 tenure with 1 to avoid division by zero
    rfm['Frequency'] = rfm['Frequency'].replace(0, 1)  # Replace 0 frequency with 1
    
    # Calculate additional features with proper handling of edge cases
    rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']
    rfm['PurchaseFrequency'] = rfm['Frequency'] / (rfm['Tenure'] / 365)  # Purchases per year
    rfm['MonetaryPerDay'] = rfm['Monetary'] / rfm['Tenure']
    rfm['ProductDiversity'] = rfm['UniqueProducts'] / rfm['Frequency']
    rfm['PurchaseRegularity'] = rfm['UniquePurchaseDates'] / rfm['Frequency']
    
    # Fill NaN values with 0
    rfm = rfm.fillna(0)
    
    # Replace infinite values with large finite numbers
    rfm = rfm.replace([np.inf, -np.inf], 0)
    
    # Calculate LTV using a more sophisticated approach
    # LTV = (Average Order Value × Purchase Frequency × Customer Lifespan) × Profit Margin
    # Assuming a 20% profit margin
    profit_margin = 0.20
    rfm['LTV'] = (rfm['AvgOrderValue'] * rfm['PurchaseFrequency'] * (rfm['Tenure'] / 365)) * profit_margin
    
    # Handle any remaining infinite values in LTV
    rfm['LTV'] = rfm['LTV'].replace([np.inf, -np.inf], 0)
    
    return rfm

def prepare_features_for_model(rfm):
    """Prepare features for the model."""
    print("Preparing features for model...")
    
    # Select features for the model
    features = [
        'Recency', 'Tenure', 'Frequency', 'Monetary', 'AvgOrderValue',
        'PurchaseFrequency', 'MonetaryPerDay', 'ProductDiversity',
        'PurchaseRegularity', 'UniqueProducts', 'AvgTransactionValue',
        'TransactionValueStd', 'TotalQuantity', 'AvgQuantity'
    ]
    
    X = rfm[features].copy()
    y = rfm['LTV'].copy()
    
    # Final check for any remaining infinite values
    X = X.replace([np.inf, -np.inf], 0)
    y = y.replace([np.inf, -np.inf], 0)
    
    return X, y

def train_and_evaluate_models(X, y):
    """Train and evaluate multiple models."""
    print("Training and evaluating models...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_scaled = scaler.transform(X)  # Scale the entire dataset
    
    # Define models to evaluate
    models = {
        'XGBoost': xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Make predictions for test set
        y_pred_test = model.predict(X_test_scaled)
        
        # Make predictions for entire dataset
        y_pred_all = model.predict(X_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2 = r2_score(y_test, y_pred_test)
        
        results[name] = {
            'model': model,
            'scaler': scaler,
            'metrics': {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            },
            'predictions': y_pred_all,  # Use predictions for all customers
            'test_predictions': y_pred_test,
            'test_data': (X_test, y_test)
        }
        
        print(f"{name} Performance:")
        print(f"MAE: ${mae:.2f}")
        print(f"RMSE: ${rmse:.2f}")
        print(f"R2 Score: {r2:.4f}")
    
    return results

def segment_customers(rfm, predictions, model_name):
    """Segment customers based on predicted LTV."""
    print(f"Segmenting customers using {model_name} predictions...")
    
    # Add predictions to RFM dataframe
    rfm[f'PredictedLTV_{model_name}'] = predictions
    
    # Create segments based on LTV percentiles
    percentiles = [0, 0.25, 0.5, 0.75, 1.0]
    labels = ['Bronze', 'Silver', 'Gold', 'Platinum']
    rfm[f'Segment_{model_name}'] = pd.qcut(
        rfm[f'PredictedLTV_{model_name}'],
        q=percentiles,
        labels=labels
    )
    
    return rfm

def create_advanced_visualizations(rfm, results):
    """Create advanced visualizations."""
    print("Creating advanced visualizations...")
    
    # 1. Feature Importance
    for model_name, result in results.items():
        if hasattr(result['model'], 'feature_importances_'):
            plt.figure(figsize=(12, 6))
            importances = pd.Series(
                result['model'].feature_importances_,
                index=rfm.columns[:len(result['model'].feature_importances_)]
            ).sort_values(ascending=True)
            importances.plot(kind='barh')
            plt.title(f'Feature Importance - {model_name}')
            plt.tight_layout()
            plt.savefig(f'output/feature_importance_{model_name}.png')
            plt.close()
    
    # 2. LTV Distribution by Segment
    for model_name in results.keys():
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            x=f'Segment_{model_name}',
            y=f'PredictedLTV_{model_name}',
            data=rfm
        )
        plt.title(f'LTV Distribution by Customer Segment - {model_name}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'output/segment_distribution_{model_name}.png')
        plt.close()
    
    # 3. Actual vs Predicted LTV (using test set)
    for model_name, result in results.items():
        plt.figure(figsize=(10, 6))
        plt.scatter(
            result['test_data'][1],
            result['test_predictions'],
            alpha=0.5
        )
        plt.plot(
            [result['test_data'][1].min(), result['test_data'][1].max()],
            [result['test_data'][1].min(), result['test_data'][1].max()],
            'r--'
        )
        plt.title(f'Actual vs Predicted LTV - {model_name}')
        plt.xlabel('Actual LTV ($)')
        plt.ylabel('Predicted LTV ($)')
        plt.tight_layout()
        plt.savefig(f'output/actual_vs_predicted_{model_name}.png')
        plt.close()
    
    # 4. Customer Value Distribution
    for model_name in results.keys():
        plt.figure(figsize=(10, 6))
        sns.histplot(
            rfm[f'PredictedLTV_{model_name}'],
            bins=50,
            kde=True
        )
        plt.title(f'Distribution of Predicted Customer Lifetime Value - {model_name}')
        plt.xlabel('Predicted LTV ($)')
        plt.tight_layout()
        plt.savefig(f'output/ltv_distribution_{model_name}.png')
        plt.close()

def main():
    # Load and preprocess data
    df = load_and_preprocess_data('online_retail.csv')
    
    # Set current date as the latest date in the dataset
    current_date = df['InvoiceDate'].max()
    
    # Calculate advanced features
    rfm = calculate_advanced_features(df, current_date)
    
    # Prepare features for model
    X, y = prepare_features_for_model(rfm)
    
    # Train and evaluate models
    results = train_and_evaluate_models(X, y)
    
    # Segment customers for each model
    for model_name, result in results.items():
        rfm = segment_customers(rfm, result['predictions'], model_name)
    
    # Create visualizations
    create_advanced_visualizations(rfm, results)
    
    # Save results
    rfm.to_csv('output/customer_ltv_predictions.csv')
    print("\nResults have been saved to the 'output' directory.")

if __name__ == "__main__":
    main() 