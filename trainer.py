#!/usr/bin/env python3
"""
Transaction Model Trainer
Trains a classification model on categorized transactions and saves it for later use.
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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
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
        text = re.sub(r'\|', '', text)  # Remove pipe char
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
            'grocery_keywords': ['GROCERY', 'SUPERMARKET', 'WALMART', 'TARGET', 'SAFEWAY', 'KROGER', 'PRIME'],
            'shopping_keywords': ['AMAZON', 'STORE', 'SHOP', 'RETAIL', 'PURCHASE', 'MALL'],
            'transport_keywords': ['UBER', 'LYFT', 'TAXI', 'TRANSIT', 'PARKING', 'TOLL'],
            'beer_keywords': ['BEER'],
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

def train_model(csv_file, model_file):
    """Train the transaction categorization model"""
    print(f"Loading training data from {csv_file}...")
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # Validate required columns
    required_cols = ['description', 'category', 'amount']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Clean data - handle empty categories
    print("Processing data...")
    
    # Handle empty/null categories by replacing with "Uncategorized"
    df['category'] = df['category'].fillna('Uncategorized')
    df['category'] = df['category'].replace('', 'Uncategorized')  # Replace empty strings
    df['category'] = df['category'].str.strip()  # Remove whitespace
    df['category'] = df['category'].replace('', 'Uncategorized')  # Replace empty after strip
    
    # Remove rows with missing description or amount
    df = df.dropna(subset=['description'])
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df = df.dropna(subset=['amount'])
    
    print(f"Training on {len(df)} transactions with {df['category'].nunique()} categories")
    
    # Show category distribution
    category_counts = df['category'].value_counts()
    print(f"Category distribution:")
    for category, count in category_counts.items():
        print(f"  {category}: {count} transactions")
    
    # Warn about rare categories (but don't exclude them)
    rare_categories = category_counts[category_counts < 5].index.tolist()
    if rare_categories:
        print(f"\nNote: Categories with < 5 examples may have lower accuracy: {rare_categories}")
        print("The model will still train on these categories.")
    
    # Check if we have only uncategorized data
    if len(category_counts) == 1 and 'Uncategorized' in category_counts:
        print("\nWarning: All transactions are uncategorized!")
        print("The model will learn to predict 'Uncategorized' for everything.")
        print("Consider manually categorizing some transactions for better results.")
    
    # Build feature extractor and pipeline
    feature_extractor = TransactionFeatureExtractor()
    feature_pipeline = feature_extractor.build_feature_pipeline(df)
    
    # Extract features
    print("Extracting features...")
    X = feature_pipeline.fit_transform(df)
    y = df['category']
    
    # Build complete model pipeline
    print("Training Random Forest model...")
    model = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),  # Don't center sparse matrices
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Handle special case where we only have one category
    if len(category_counts) == 1:
        print(f"\nOnly one category found: {list(category_counts.keys())[0]}")
        print("Training a simple classifier (no train/test split needed)")
        
        # Train on all data
        model.fit(X, y)
        print("Model trained successfully (100% accuracy expected for single category)")
        
        # Skip validation since there's only one category
        use_cv = False
    else:
        # Check if we can do stratified split (need at least 2 samples per class)
        class_counts = y.value_counts()
        min_class_count = class_counts.min()
        
        if min_class_count < 2:
            print(f"Warning: Some categories have only 1 transaction:")
            rare_categories = class_counts[class_counts == 1].index.tolist()
            print(f"  {rare_categories}")
            print("Using regular train/test split instead of stratified split")
            
            # Use regular split when we have rare categories
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            use_stratify = False
        else:
            # Use stratified split when possible
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            use_stratify = True
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel trained successfully!")
        print(f"Validation accuracy: {accuracy:.3f}")
        
        use_cv = True
    
    # Cross-validation score (handle rare categories and single category)
    if use_cv and len(category_counts) > 1:
        if use_stratify and min_class_count >= 5:  # Need at least 5 for 5-fold CV
            cv_scores = cross_val_score(model, X, y, cv=5)
            print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        elif min_class_count >= 3:
            cv_scores = cross_val_score(model, X, y, cv=3)
            print(f"3-fold cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        else:
            print("Skipping cross-validation due to rare categories (< 3 samples per class)")
    
    # Print detailed classification report (only if we have test data)
    if use_cv and len(category_counts) > 1:
        print("\nDetailed Performance Report:")
        print(classification_report(y_test, y_pred))
    
    # Save model and feature pipeline
    model_data = {
        'model': model,
        'feature_pipeline': feature_pipeline,
        'categories': sorted(df['category'].unique()),
        'feature_extractor': feature_extractor
    }
    
    joblib.dump(model_data, model_file)
    print(f"\nModel saved to {model_file}")
    print("You can now use this model with the categorize_transactions.py script")

def main():
    parser = argparse.ArgumentParser(
        description='Train a transaction categorization model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_model.py my_transactions.csv my_model.pkl
  
Required CSV format (your format is supported):
  Account,Date,Payee,description,category,amount,Split_Amount,Cleared
  
The script will use only: description, category, amount
Empty categories will be treated as "Uncategorized"
        """
    )
    parser.add_argument('training_file', help='CSV file with training data (description, category, amount)')
    parser.add_argument('model_file', help='Output file to save the trained model (.pkl)')
    
    args = parser.parse_args()
    
    try:
        train_model(args.training_file, args.model_file)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
