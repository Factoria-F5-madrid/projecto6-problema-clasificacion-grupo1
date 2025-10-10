"""
Simple test with real data
"""

import sys
import os
sys.path.append('backend')

import pandas as pd
import numpy as np

def test_data_loading():
    """Test loading real data"""
    
    print("🧪 TESTING DATA LOADING")
    print("=" * 40)
    
    try:
        # Load data directly
        df = pd.read_csv('backend/data/processed/cleaned_tweets.csv')
        print(f"✅ Data loaded: {df.shape[0]} samples, {df.shape[1]} features")
        print(f"✅ Columns: {list(df.columns)}")
        print(f"✅ Target distribution:")
        print(df['class_label'].value_counts())
        
        # Show sample data
        print(f"\n✅ Sample data:")
        print(df[['clean_tweet_improved', 'class_label']].head())
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_data_loading()
