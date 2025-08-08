#!/usr/bin/env python3
"""
Transaction Categorizer
Uses a pre-trained model to categorize new transactions.
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import warnings
warnings.filterwarnings('ignore')

class TransactionFeatureExtractor:
    """Handles all feature extraction for transaction data"""
    
    def extract_amount_features(self, df):
        """Extract numerical features from transaction amounts"""
        features = pd.DataFrame(index=df.index)
        
        # Convert amount to absolute value for features
        abs_amount = df['amount'].abs()
        
        features['amount'] = abs_amount
        features['log_amount'] = np.log1p(abs_amount)  # log(1 + amount) to handle zeros
        features['is_small'] = (abs_amount < 10).astype(int)
        features['is_medium'] = ((abs_amount >= 10) & (abs_amount < 100)).astype(int)
        features['is_large'] = (abs_amount >= 100).astype(int)
        features['is_very_large'] = (abs_amount >= 500).astype(int)
        
        return features
    
    def clean_description(self, text):
        """Clean and normalize transaction descriptions"""
        if pd.isna(text):
            return ""
        
        # Convert to uppercase and remove extra spaces
        text = str(text).upper().strip()
        
        # Remove common transaction codes and numbers
        text = re.sub(r'#\d+', '', text)  # Remove #123 style codes
        text = re.sub(r'\b\d{2,}\b', '', text)  # Remove standalone numbers (2+ digits)
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        return text.strip()
    
    def extract_keyword_features(self, df):
        """Extract keyword-based features from descriptions"""
        features = pd.DataFrame(index=df.index)
        
        # Clean descriptions
        clean_desc = df['description'].apply(self.clean_description)
        
        # Define keyword categories
        keyword_groups = {
            'food_keywords': ['STARBUCKS', 'MCDONALDS', 'RESTAURANT', 'CAFE', 'PIZZA', 'BURGER', 
                             'FOOD', 'DINING', 'KITCHEN', 'GRILL', 'BAKERY', 'DELI', 'SUBWAY'],
            'gas_keywords': ['SHELL', 'EXXON', 'BP', 'CHEVRON', 'GAS', 'FUEL', 'STATION', 'MOBIL'],
            'grocery_keywords': ['GROCERY', 'SUPERMARKET', 'WALMART', 'TARGET', 'SAFEWAY', 'KROGER'],
            'shopping_keywords': ['AMAZON', 'STORE', 'SHOP', 'RETAIL', 'PURCHASE', 'MALL'],
            'transport_keywords': ['UBER', 'LYFT', 'TAXI', 'TRANSIT', 'PARKING', 'TOLL'],
            'bills_keywords': ['ELECTRIC', 'WATER', 'PHONE', 'INTERNET', 'INSURANCE', 'PAYMENT'],
            'atm_keywords': ['ATM', 'WITHDRAWAL', 'CASH', 'DEPOSIT'],
            'bank_keywords': ['BANK', 'CREDIT', 'TRANSFER', 'FEE', 'INTEREST']
        }
        
        # Create binary features for keyword groups
        for group_name, keywords in keyword_groups.items():
            pattern = '|'.join(keywords)
            features[group_name] = clean_desc.str.contains(pattern, case=False, na=False).astype(int)
        
        return features
    
    def _extract_amount_features_wrapper(self, X):
        """Wrapper method for amount feature extraction (needed for pickling)"""
        return self.extract_amount_features(X)
    
    def _extract_keyword_features_wrapper(self, X):
        """Wrapper method for keyword feature extraction (needed for pickling)"""
        return self.extract_keyword_features(X)
    
    def _get_clean_descriptions(self, df):
        """Extract clean descriptions (needed for pickling)"""
        return df['description'].apply(self.clean_description)
    
    def build_feature_pipeline(self, df):
        """Build the complete feature extraction pipeline"""
        
        # Amount features
        amount_transformer = FunctionTransformer(
            self._extract_amount_features_wrapper,
            validate=False
        )
        
        # Keyword features  
        keyword_transformer = FunctionTransformer(
            self._extract_keyword_features_wrapper,
            validate=False
        )
        
        # Text features using TF-IDF on cleaned descriptions
        text_transformer = Pipeline([
            ('extract_text', FunctionTransformer(self._get_clean_descriptions, validate=False)),
            ('tfidf', TfidfVectorizer(
                max_features=100, 
                ngram_range=(1, 2), 
                stop_words='english',
                min_df=2
            ))
        ])
        
        # Combine all features
        feature_pipeline = FeatureUnion([
            ('amount_features', amount_transformer),
            ('keyword_features', keyword_transformer),
            ('text_features', text_transformer)
        ])
        
        return feature_pipeline

def categorize_transactions(model_file, input_file, output_file):
    """Categorize transactions using a pre-trained model"""
    
    # Load the trained model
    print(f"Loading model from {model_file}...")
    try:
        model_data = joblib.load(model_file)
        model = model_data['model']
        feature_pipeline = model_data['feature_pipeline']
        categories = model_data['categories']
        print(f"Model loaded successfully. Available categories: {categories}")
    except Exception as e:
        raise ValueError(f"Error loading model: {e}")
    
    # Load new transactions
    print(f"Loading transactions from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Validate required columns
    required_cols = ['description', 'amount']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Clean data
    original_len = len(df)
    df = df.dropna(subset=['description'])
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df = df.dropna(subset=['amount'])
    
    if len(df) < original_len:
        print(f"Removed {original_len - len(df)} rows with missing/invalid data")
    
    print(f"Categorizing {len(df)} transactions...")
    
    # Extract features using the same pipeline from training
    X = feature_pipeline.transform(df)
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Add predictions to dataframe
    df_result = df.copy()
    df_result['category'] = predictions
    df_result['confidence'] = np.max(probabilities, axis=1)
    
    # Add individual category probabilities for reference
    for i, category in enumerate(model.classes_):
        df_result[f'prob_{category}'] = probabilities[:, i]
    
    # Sort by confidence (lowest first) to help identify uncertain predictions
    # df_result = df_result.sort_values('confidence')
    
    # Save results
    df_result.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Print summary statistics
    print(f"\n=== Categorization Summary ===")
    category_counts = pd.Series(predictions).value_counts()
    for category, count in category_counts.items():
        print(f"  {category}: {count} transactions")
    
    avg_confidence = df_result['confidence'].mean()
    low_confidence_count = (df_result['confidence'] < 0.5).sum()
    
    print(f"\n=== Confidence Statistics ===")
    print(f"Average confidence: {avg_confidence:.3f}")
    print(f"Low confidence predictions (<0.5): {low_confidence_count} transactions")
    
    if low_confidence_count > 0:
        print(f"\nNote: {low_confidence_count} predictions have low confidence.")
        print("Consider reviewing these manually. They are sorted at the top of the output file.")
    
    # Show some example predictions
    print(f"\n=== Sample Predictions ===")
    sample_df = df_result.head(5)[['description', 'amount', 'category', 'confidence']]
    for _, row in sample_df.iterrows():
        print(f"'{row['description'][:40]}...' (${row['amount']:.2f}) → {row['category']} ({row['confidence']:.3f})")
    
    return df_result

def main():
    parser = argparse.ArgumentParser(
        description='Categorize transactions using a pre-trained model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python categorize_transactions.py my_model.pkl new_transactions.csv results.csv
  
Required CSV format for input:
  description,amount
  "DUNKIN DONUTS #456",-3.25
  "CHEVRON GAS STATION",-38.50
  
Output will include:
  - All original columns
  - category: The predicted category
  - confidence: Model confidence (0-1)
  - prob_[category]: Probability for each category
        """
    )
    parser.add_argument('model_file', help='Pre-trained model file (.pkl)')
    parser.add_argument('input_file', help='CSV file with transactions to categorize')
    parser.add_argument('output_file', help='Output CSV file for categorized transactions')
    parser.add_argument('--min-confidence', type=float, default=0.0,
                       help='Only output predictions with confidence >= this value')
    
    args = parser.parse_args()
    
    try:
        results = categorize_transactions(args.model_file, args.input_file, args.output_file)
        
        # Filter by confidence if specified
        if args.min_confidence > 0.0:
            low_conf_count = (results['confidence'] < args.min_confidence).sum()
            if low_conf_count > 0:
                print(f"\nFiltering out {low_conf_count} predictions below confidence threshold {args.min_confidence}")
                high_conf_results = results[results['confidence'] >= args.min_confidence]
                filtered_output = args.output_file.replace('.csv', '_high_confidence.csv')
                high_conf_results.to_csv(filtered_output, index=False)
                print(f"High-confidence results saved to {filtered_output}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
