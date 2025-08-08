import pandas as pd
import numpy as np
from apyori import apriori
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import warnings
import itertools

CSV_PATH = 'data/Electronics.csv'
JSON_PATH = 'data/Electronics_5.json'
TARGET_DF_SIZE = 10000  
CHUNK_SIZE = 5000      
K_RECOMMENDATIONS = 10 
MIN_SUPPORT = 0.00005  
MIN_CONFIDENCE = 0.1   
MIN_LIFT = 1.0         
CHUNK_SIZE_TFIDF = 2000 
MIN_UNITS_REQUIRED = 200  

# --- Step 1: Dataset Acquisition & Initial Sampling (Memory Optimized) ---
print("Step 1: Dataset Acquisition & Initial Sampling (Memory Optimized)...")
transactions_df_chunks = pd.read_csv(CSV_PATH, names=['asin', 'reviewerID', 'overall', 'unixReviewTime'], chunksize=CHUNK_SIZE)
transactions_df_prelim = pd.concat(chunk for chunk in transactions_df_chunks)
TRANSACTION_ASIN_SAMPLE_SIZE = min(len(transactions_df_prelim), TARGET_DF_SIZE * 2)
transactions_df = transactions_df_prelim.sample(n=TRANSACTION_ASIN_SAMPLE_SIZE, random_state=42)
transactions_df['overall'] = transactions_df['overall'].astype(np.float16)
transactions_df['unixReviewTime'] = transactions_df['unixReviewTime'].astype(np.int32)
transactions_df['reviewerID'] = transactions_df['reviewerID'].astype('category')
transactions_df['asin'] = transactions_df['asin'].astype('category')
print(f"Pre-sampled transactions_df shape for ASINs: {transactions_df.shape}")
print(f"Memory usage of pre-sampled transactions_df: {transactions_df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

# Load and filter JSON data in chunks
print("Processing JSON chunks and filtering for relevant ASINs...")
relevant_asins = set(transactions_df['asin'].unique())
json_data_chunks = pd.read_json(JSON_PATH, lines=True, chunksize=CHUNK_SIZE)
json_data = pd.concat([chunk[['asin', 'reviewText', 'summary', 'style']] for chunk in json_data_chunks if chunk['asin'].isin(relevant_asins).any()], ignore_index=True)
json_data = json_data[json_data['asin'].isin(relevant_asins)].sample(n=min(len(json_data), TARGET_DF_SIZE * 5), random_state=42)
json_data['asin'] = json_data['asin'].astype('category')
print(f"Stopping JSON read early after processing {len(json_data)} rows.")
print(f"Filtered data_df_json_combined shape before merge: {json_data.shape}")
print(f"Memory usage of data_df_json_combined: {json_data.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

# Merge datasets and derive units required
print("Attempting merge of transactions and JSON data...")
data_df = pd.merge(transactions_df, json_data, on='asin', how='inner')
# Estimate units required as 2x review count, minimum 200
data_df['units_required'] = data_df.groupby('asin')['reviewerID'].transform('count') * 2
data_df['units_required'] = data_df['units_required'].clip(lower=MIN_UNITS_REQUIRED)
print(f"Initial merged data_df shape: {data_df.shape}")
print(f"Columns in data_df after merge: {list(data_df.columns)}")
print(f"Memory usage of initial merged data_df: {data_df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
data_df = data_df.sample(n=TARGET_DF_SIZE, random_state=42).reset_index(drop=True)
print(f"Final sampled data_df shape: {data_df.shape}")
print(f"Memory usage of final sampled data_df: {data_df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

# --- Step 2: Data Cleaning & Preprocessing ---
print("Step 2: Data Cleaning & Preprocessing...")
data_df = data_df.rename(columns={'reviewerID': 'user_id', 'asin': 'item_id', 'overall': 'rating'})
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    data_df['rating'] = data_df['rating'].fillna(data_df['rating'].mean())
    data_df['reviewText'] = data_df['reviewText'].fillna('No review')
    data_df['summary'] = data_df['summary'].fillna('No summary')
if 'style' in data_df.columns:
    data_df['style_format'] = data_df['style'].apply(lambda x: str(x).replace(':', '') if isinstance(x, dict) else str(x))
    data_df['style_format'] = data_df['style_format'].fillna('')
    data_df = data_df.drop(columns=['style'])
data_df = data_df.drop_duplicates(subset=['user_id', 'item_id'])
print(f"Data cleaned and preprocessed. Final shape: {data_df.shape}")
print(f"Memory usage after preprocessing: {data_df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

# --- Step 3: Data Transformation for Apriori ---
print("Step 3: Data Transformation for Apriori...")
user_item_counts = data_df['user_id'].value_counts()
valid_users = user_item_counts[user_item_counts >= 2].index
data_df_filtered = data_df[data_df['user_id'].isin(valid_users)]
transactions = data_df_filtered.groupby('user_id')['item_id'].apply(list).tolist()
print(f"Number of transactions (users with 2+ items): {len(transactions)}")

# --- Step 4: Apply Apriori Algorithm ---
print("Step 4: Applying Apriori Algorithm...")
results = list(apriori(transactions, min_support=MIN_SUPPORT, min_confidence=MIN_CONFIDENCE, min_lift=MIN_LIFT, min_length=2))
print(f"Number of association rules generated: {len(results)}")

# --- Step 5: Feature Engineering with TF-IDF (Chunked) ---
print("Step 5: Feature Engineering with TF-IDF...")
data_df['combined_text'] = data_df['reviewText'] + " " + data_df['summary'] + " " + data_df['style_format']

tfidf = TfidfVectorizer(stop_words='english', max_features=2000, ngram_range=(1, 2))
chunk_size_tfidf = CHUNK_SIZE_TFIDF
tfidf_matrices = []
for i in range(0, len(data_df), chunk_size_tfidf):
    chunk = data_df.iloc[i:i + chunk_size_tfidf]
    tfidf_matrix_chunk = tfidf.fit_transform(chunk['combined_text'])
    tfidf_matrices.append(tfidf_matrix_chunk)
    print(f"Processed TF-IDF chunk {i//chunk_size_tfidf + 1}")

from scipy.sparse import vstack
tfidf_matrix = vstack(tfidf_matrices)
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

item_id_to_idx = {item_id: idx for idx, item_id in enumerate(data_df['item_id'].unique())}
idx_to_item_id = {idx: item_id for idx, item_id in enumerate(data_df['item_id'].unique())}

# Chunked content similarity
print("Step 6: Computing Chunked Content Similarity...")
def chunked_cosine_similarity(matrix, chunk_size=1000):
    n_samples = matrix.shape[0]
    similarity_dict = {}
    for i in range(0, n_samples, chunk_size):
        start_i = i
        end_i = min(i + chunk_size, n_samples)
        chunk_i = matrix[start_i:end_i]
        for j in range(0, n_samples, chunk_size):
            start_j = j
            end_j = min(j + chunk_size, n_samples)
            chunk_j = matrix[start_j:end_j]
            chunk_sim = cosine_similarity(chunk_i, chunk_j)
            for idx_i, local_i in enumerate(range(start_i, end_i)):
                if local_i not in similarity_dict:
                    similarity_dict[local_i] = {}
                for idx_j, local_j in enumerate(range(start_j, end_j)):
                    if local_i != local_j:
                        similarity_dict[local_i][local_j] = chunk_sim[idx_i, idx_j]
    return similarity_dict

content_similarity = chunked_cosine_similarity(tfidf_matrix)
print(f"Content similarity computed for {len(content_similarity)} items")

# --- Step 7: Recommendation Functions ---
print("Step 7: Generating Recommendations...")

# Top 10 based on Mean Rating with Units Required
def get_top_10_recommendations(data_df, k=K_RECOMMENDATIONS):
    top_10_df = data_df.groupby('item_id')['rating'].mean().sort_values(ascending=False).head(k).reset_index()
    top_10_df = top_10_df.merge(data_df[['item_id', 'units_required']].drop_duplicates(), on='item_id', how='left')
    top_10_df['units_required'] = top_10_df['units_required'].fillna(MIN_UNITS_REQUIRED)
    return top_10_df.rename(columns={'rating': 'score'})[['item_id', 'units_required', 'score']]

# --- Step 8: Generate and Display Top 10 Recommendations ---
print("Step 8: Generating and Displaying Top 10 Recommendations...")
top_10_recs = get_top_10_recommendations(data_df)
print("\nTop 10 Products with Recommended Stock Units:")
for _, row in top_10_recs.iterrows():
    print(f"Recommend Product-{row['item_id']} / Units Required-{int(row['units_required'])}")

# --- Step 9: Save and Visualize ---
output_file = 'top_10_product_recommendations.csv'
try:
    top_10_recs.to_csv(output_file, index=False)
    print(f"Recommendations saved to '{output_file}'")
except PermissionError:
    alt_output_file = 'top_10_product_recommendations_alt.csv'
    top_10_recs.to_csv(alt_output_file, index=False)
    print(f"Permission denied. Saved to '{alt_output_file}'")
except Exception as e:
    print(f"Error saving file: {e}. Recommendations not saved.")

if not top_10_recs.empty:
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Plot score (average rating) on the left y-axis
    sns.barplot(x='item_id', y='score', data=top_10_recs, ax=ax1, color='skyblue', label='Average Rating')
    ax1.set_xlabel('Item ID')
    ax1.set_ylabel('Average Rating', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Create a second y-axis for units required
    ax2 = ax1.twinx()
    ax2.plot(top_10_recs['item_id'], top_10_recs['units_required'], color='red', marker='o', label='Units Required')
    ax2.set_ylabel('Units Required', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Top 10 Products: Rating vs Units Required', fontsize=14)
    plt.xticks(rotation=90, ha='right')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
    plt.tight_layout()
    image_file = 'top_10_product_recommendations_plot.png'
    plt.savefig(image_file)
    plt.show()
    print(f"Recommendation plot saved to '{image_file}'")
else:
    print("No plot generated due to empty recommendations.")