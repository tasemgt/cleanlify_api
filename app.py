import os
import re
import json
import string
import requests
from io import StringIO
from datetime import datetime
import sqlite3

import cherrypy
from cherrypy.lib import static
from dotenv import load_dotenv

import numpy as np
import pandas as pd
from spellchecker import SpellChecker
from fuzzywuzzy import process
from rapidfuzz import fuzz
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score


# Load environment variables from .env
load_dotenv()

import google.generativeai as genai


# Configure your Gemini API key here
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Google Knowledge Graph API settings
GOOGLE_KG_API_KEY = os.getenv("GOOGLE_KG_API_KEY")
GOOGLE_KG_API_URL = os.getenv("GOOGLE_KG_API_URL", "https://kgsearch.googleapis.com/v1/entities:search")

# CORS tool to allow cross-origin requests
def cors():
    cherrypy.response.headers["Access-Control-Allow-Origin"] = "*"
    cherrypy.response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    cherrypy.response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    cherrypy.response.headers["Access-Control-Max-Age"] = "86400"

cherrypy.tools.cors = cherrypy.Tool('before_handler', cors)


# Directory to store uploaded files
UPLOAD_DIR = os.path.join(os.getcwd(), 'uploads')

DB_PATH = "hci.db"

spell = SpellChecker(distance=2)  # small edit distance

# 
def str_to_bool(s: str) -> bool:
    if isinstance(s, bool):
        return s
    s = s.strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    elif s in ("false", "0", "no", "n", "f"):
        return False
    else:
        raise ValueError(f"Cannot parse boolean value from: {s}")

# ---------- Helpers Functions ----------
def preprocess_list(items):
    cleaned = []
    for item in items:
        # Detect tuple (value, row_id)
        if isinstance(item, (list, tuple)) and len(item) == 3:
            val, row_id, short_text_col = item[0], item[1], item[2]
        else:
            val, row_id = item, None

        if pd.isna(val):
            cleaned.append((np.nan, row_id, short_text_col))
            continue

        s = str(val).strip().lower()
        s = re.sub(r'[^\w\s]', '', s)
        cleaned.append((s, row_id, short_text_col) if row_id is not None else s)
    return cleaned

# --- Preprocess helper ---
def preprocess_text(text):
    text = str(text).lower().strip()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text

def correct_phrase(text):
    """Apply spell correction to each word."""
    words = text.split()
    corrected = [spell.correction(w) or w for w in words]
    return ' '.join(corrected)

# ---------- Cluster step ----------
def cluster_brands(preprocessed_list, threshold, match_mode):
    clusters = []
    # Example simplified clustering logic
    while preprocessed_list:
        base = preprocessed_list.pop(0)
        cluster = [base]
        base_value = base[0]

        rest = []
        for item in preprocessed_list:
            value = item[0]

             # Get the function from fuzz by name
            func = getattr(fuzz, match_mode, None)
            
            if callable(func):  #partial_token_sort_ratio
                similarity = func(base_value, value) # Use fuzzy matching ratio levenshtein distance
                if similarity >= threshold:
                    cluster.append(item)
                else:
                    rest.append(item)
            else:
                raise AttributeError(f"'{match_mode}' is not a valid fuzz function")

        preprocessed_list = rest
        clusters.append(cluster)
    
    # print("Clusters formed:", clusters)  # Debugging output

    return clusters


# ---------- Process clusters ----------
def process_clusters(clusters):
    cluster_dict = {}
    brand_to_cluster = {}

    for idx, cluster in enumerate(clusters, start=1):
        # cluster is list of tuples (value, row_id) added short_text_col
        cluster = list(cluster)
        # Extract only the values for suggestion logic
        values = [c[0] for c in cluster]
        total = len(values)

        suggestion = None
        suggestion_mode = None
        confidence = 1.0 if total == 1 else None

        if total == 1:
            suggestion = values[0]
            suggestion_mode = "single_member"
        else:
            # majority vote
            freq = Counter(values)
            most_common, count = freq.most_common(1)[0]
            suggestion = most_common
            suggestion_mode = "majority"
            confidence = count / total

            # If all distinct, use spell checking heuristics
            if len(set(values)) == total:
                valid_dict_words = [w for w in values if w in spell]
                if valid_dict_words:
                    freq_valid = Counter(valid_dict_words)
                    suggestion, count = freq_valid.most_common(1)[0]
                    suggestion_mode = "spell_checker"
                    confidence = count / total if count else 0.9
                else:
                    corrected = [correct_phrase(w) or w for w in values]
                    freq_corrected = Counter(corrected)
                    suggestion, count = freq_corrected.most_common(1)[0]
                    suggestion_mode = "spell_checker"
                    confidence = 0.9

        # Build member dicts: each member has a single integer row_id
        members = []
        for val, row_id, short_text_col in cluster:
            # ensure row_id is a native int if present
            member_row_id = int(row_id) if row_id is not None and not pd.isna(row_id) else None
            members.append({
                "value": val,
                "row_ids": member_row_id,  # per your request: single int, not a list
                "short_text_col": short_text_col  # if you want to keep track of the column name
            })

        cluster_tag = f"cluster_{idx}"
        cluster_dict[cluster_tag] = {
            'members': members,
            'suggestion': suggestion,
            'suggestion_mode': suggestion_mode,
            'original_suggestion_mode': suggestion_mode,
            'confidence': round(confidence, 2) if confidence is not None else None
        }

        # Map each cleaned value -> cluster_tag (last assignment wins, but values in same cluster are consistent)
        for v in values:
            brand_to_cluster[v] = cluster_tag

    return cluster_dict, brand_to_cluster

# ---------- KMeans Auto Clustering for columns direct ----------
def kmeans_auto_cluster(pairs, min_k=2, max_k=20):
    """
    Cluster list of (value, row_id, short_text_col) tuples using MiniBatchKMeans.
    Automatically selects best number of clusters via silhouette score.

    Args:
        pairs: list of tuples (value, row_id, short_text_col)
        min_k: minimum number of clusters to try
        max_k: maximum number of clusters to try

    Returns:
        clusters: list of lists of tuples grouped into clusters
    """
    # Extract just the brand values (first element of tuple)
    # Extract just the brand values (first element of tuple)
    values = [p[0] for p in pairs]

    if len(values) < min_k:
        # Not enough data to cluster
        return [pairs]

    # Vectorize using character n-gram TF-IDF
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2,4))
    X = vectorizer.fit_transform(values)

    best_k = min_k
    best_score = -1
    best_labels = None

    # Try different k values
    for k in range(min_k, min(max_k, len(values)) + 1):
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        labels = kmeans.labels_

        if len(set(labels)) == 1:
            continue  # skip degenerate case

        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    # If no valid clustering found, return everything as one cluster
    if best_labels is None:
        return [pairs]

    # Group tuples back into clusters
    clusters = []
    for cluster_id in set(best_labels):
        cluster_items = [pairs[i] for i in range(len(pairs)) if best_labels[i] == cluster_id]
        clusters.append(cluster_items)

    return clusters



# ---------- STEP 3: Full Category Cluster and suggestion pipeline ----------
def clean_category_dict(category_brand_dict, threshold=75, match_mode="ratio", isML=False):

    all_cluster_info = {}
    all_brand_to_suggestion = {}
    all_cluster_info_stats = {}

    for category, brand_list in category_brand_dict.items():
        # brand_list is expected to be list of (value, row_id) just added short_text_col
        cleaned_pairs = preprocess_list(brand_list)  # returns list of (cleaned_value, row_id), just added short_text_col

        # filter out NaNs (preprocess_list marks them with np.nan)
        cleaned_pairs = [p for p in cleaned_pairs if not pd.isna(p[0])]

        if isML:
            # Use ML-based clustering (e.g. KMeans)
            print('Using ML for category text clustering:::', isML)
            clusters = kmeans_auto_cluster(cleaned_pairs)
        else:
            # Use fuzzy matching based clustering
            # Cluster using tuple-aware function
            print('Not using ML for category short text clustering:', isML)
            clusters = cluster_brands(cleaned_pairs, threshold, match_mode) # Adjust threshold as needed

        cluster_dict, brand_to_cluster = process_clusters(clusters)

        # Map cleaned value -> suggestion
        brand_to_suggestion = {}
        for cleaned_val, cluster_tag in brand_to_cluster.items():
            suggestion = cluster_dict[cluster_tag]['suggestion']
            brand_to_suggestion[cleaned_val] = suggestion

        all_cluster_info[category] = cluster_dict
        all_brand_to_suggestion[category] = brand_to_suggestion


    #     cleaned = df[col].replace(r'^\s*$', np.nan, regex=True)

    #     # Total non-null values
    #     non_null_count = int(cleaned.count())
    #     # Total unique values (excluding nulls)
    #     unique_count = cleaned.nunique()
        
    #     print('MAPPING: ', 'Mike')

    #     all_cluster_info_stats[col] = {
    #         'num_of_before_unique': unique_count,
    #         'total_values': non_null_count
    #     }
    #     # print(df.head())

    # print('ALL CLUSTER INFO STATS: ', all_cluster_info_stats)

    return all_cluster_info, all_brand_to_suggestion



# ---------- Clustering Helper Functions for Column direct cleaning  ----------

def preprocess_list_cols(items):
    """Lowercase, strip, remove punctuation; keep NaN as-is."""
    cleaned = []
    for item in items:
        if pd.isna(item):
            cleaned.append(item)
            continue
        item = str(item).strip().lower()
        item = re.sub(r'[^\w\s]', '', item)
        cleaned.append(item)
    return cleaned



def cluster_items(values_with_meta, threshold, match_mode):
    """Cluster items (with row ids and column names) by fuzzy similarity."""
    clusters = []
    for item in values_with_meta:
        if pd.isna(item['value']):
            continue
        found_cluster = False
        for cluster in clusters:
            # Use fuzzy matching ratio to determine similarity
            # Get the function from fuzz by name
            func = getattr(fuzz, match_mode, None)
            
            if callable(func):  #partial_token_sort_ratio
                if any(func(item['value'], member['value']) >= threshold for member in cluster):
                    cluster.append(item)
                    found_cluster = True
                    break
            else:
                raise AttributeError(f"'{match_mode}' is not a valid fuzz function")
        if not found_cluster:
            clusters.append([item])
    return clusters


def process_clusters_cols(clusters):
    """Generate dict with suggestions, members containing row_ids and col names."""
    cluster_dict = {}
    value_to_cluster = {}

    for idx, cluster in enumerate(clusters, start=1):
        total = len(cluster)
        suggestion = None
        suggestion_mode = None
        confidence = 1.0 if total == 1 else None

        # Extract just the text values
        text_values = [m['value'] for m in cluster]

        if total == 1:
            suggestion = text_values[0]
            suggestion_mode = "single_member"
        else:
            freq = Counter(text_values)
            most_common, count = freq.most_common(1)[0]
            suggestion = most_common
            suggestion_mode = "majority"
            confidence = count / total

            # If all distinct → try dictionary or spellcheck
            if len(set(text_values)) == total:
                valid_dict_words = [word for word in text_values if word in spell]
                if valid_dict_words:
                    freq_valid = Counter(valid_dict_words)
                    suggestion, count = freq_valid.most_common(1)[0]
                    suggestion_mode = "spell_checker"
                    confidence = count / total if count else 0.9
                else:
                    corrected = [correct_phrase(b) or b for b in text_values]
                    freq_corrected = Counter(corrected)
                    suggestion, count = freq_corrected.most_common(1)[0]
                    suggestion_mode = "spell_checker"
                    confidence = 0.9

        cluster_tag = f"cluster_{idx}"
        cluster_dict[cluster_tag] = {
            "members": cluster,  # already dicts with value, row_ids, short_text_col
            "suggestion": suggestion,
            "suggestion_mode": suggestion_mode,
            "original_suggestion_mode": suggestion_mode,
            "confidence": round(confidence, 2) if confidence is not None else None
        }

        for member in cluster:
            value_to_cluster[member['value']] = cluster_tag

    return cluster_dict, value_to_cluster


# ---------- KMeans Clustering for Short Text Columns ----------
def cluster_items_kmeans(values_with_meta, min_k=2, max_k=10, use_minibatch=True):
    """Cluster items (with row ids and column names) using KMeans with auto k selection."""
    if not values_with_meta:
        return []

    texts = [item["value"] for item in values_with_meta]

    # Vectorize using character n-grams (better for short text + typos)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
    X = vectorizer.fit_transform(texts)

    best_k = None
    best_score = -1
    best_labels = None

    # Avoid trying more clusters than data points
    max_k = min(max_k, len(texts))

    # Search for optimal k using silhouette score
    for k in range(min_k, max_k + 1):
        if k >= len(texts):  # can't have more clusters than points
            break
        try:
            if use_minibatch:
                model = MiniBatchKMeans(n_clusters=k, random_state=42, n_init="auto")
            else:
                model = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = model.fit_predict(X)

            if len(set(labels)) == 1:
                continue  # silhouette score fails with 1 cluster
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        except Exception:
            continue

    # Fallback: if silhouette fails, use single cluster
    if best_labels is None:
        best_labels = np.zeros(len(texts))

    # Group values by cluster
    clusters = []
    for cluster_id in set(best_labels):
        cluster_items = [
            values_with_meta[i] for i, label in enumerate(best_labels) if label == cluster_id
        ]
        clusters.append(cluster_items)

    return clusters



# ---------- Clustering Short Text Columns ----------
def cluster_short_text(data, columns, threshold, match_mode, isML=False):

    """Main pipeline for short text clustering with metadata tracking."""
    # Convert input into DataFrame
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, dict):
        df = pd.DataFrame(data)
    elif isinstance(data, list) and all(isinstance(row, dict) for row in data):
        df = pd.DataFrame(data)
    else:
        raise ValueError("Unsupported data format")

    all_cluster_info = {}
    all_cluster_info_stats = {}

    for col in columns:
        if col not in df.columns:
            continue

        cleaned_values = preprocess_list_cols(df[col])

        # Add metadata to each value
        values_with_meta = [
            {"value": val, "row_ids": idx, "short_text_col": col}
            for idx, val in enumerate(cleaned_values)
            if isinstance(val, str) and val.strip() != ""
        ]

        if isML:
            # Use ML-based clustering (e.g. KMeans)
            print('Using ML for short text clustering:', isML)                      
            clusters = cluster_items_kmeans(values_with_meta, min_k=2, max_k=8)
        else:
            # Use fuzzy matching based clustering           
            print('Not using ML for short text clustering:', isML)                      
            clusters = cluster_items(values_with_meta, threshold=threshold, match_mode=match_mode)

        cluster_info, mapping = process_clusters_cols(clusters)

        # print('CLUSTERS: ', clusters)

        all_cluster_info[col] = cluster_info

        # mapping[col] = {
        #     "mapping": mapping
        # }
        cleaned = df[col].replace(r'^\s*$', np.nan, regex=True)

        # Total non-null values
        non_null_count = int(cleaned.count())
        # Total unique values (excluding nulls)
        unique_count = cleaned.nunique()
        
        # print('MAPPING: ', 'Mike')

        all_cluster_info_stats[col] = {
            'num_of_before_unique': unique_count,
            'total_values': non_null_count
        }
        # print(df.head())

    # print('ALL CLUSTER INFO STATS: ', all_cluster_info_stats)

    return all_cluster_info, all_cluster_info_stats


# Google K Graph canonicalization Helper Function
# def get_canonical_name(query, context_type=""):
def get_canonical_name(query):
    params = {
        "query": query,
        "key": GOOGLE_KG_API_KEY,
        "limit": 5,
        "indent": True
    }
    response = requests.get(GOOGLE_KG_API_URL, params=params)
    data = response.json()

    if "itemListElement" not in data or not data["itemListElement"]:
        return None

    candidates = []
    for item in data["itemListElement"]:
        result = item.get("result", {})
        types = result.get("@type", [])
        name = result.get("name", "")
        if not name:
            continue

        # # If context_type provided, only keep matching types
        # if context_type and context_type not in types:
        #     continue

        score = fuzz.partial_ratio(query.lower(), name.lower())
        candidates.append((name, score))

    if not candidates:
        return None

    # Pick the candidate with the highest fuzzy match score
    canonical_name = max(candidates, key=lambda x: x[1])[0]
    return canonical_name



# ---------- API CLASS!!!! ------------------------------------------------------------------------------
# Class to handle the Cleanlify API
class CleanlifyAPI:
    def __init__(self):
        self.data_df = None
        self.filtered_text_df = None
        self.text_columns_to_clean = []
        self.cleaned_text_df = None
        self.original_filename = None
        self.analysis_result = None


    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.cors()
    def analyze(self, file, limit="100"):
        """Upload + Analyze CSV: Suggest fields, flag short text columns, and mark important categorical columns."""
        contents = file.file.read().decode("utf-8")
        self.data_df = pd.read_csv(StringIO(contents))
        self.original_filename = file.filename

        # Replace unnamed columns with blank
        self.data_df.columns = ["" if str(col).startswith("Unnamed") else col for col in self.data_df.columns]

        result = []
        total_rows = int(len(self.data_df))

        for col in self.data_df.columns:
            series = self.data_df[col].dropna().astype(str).str.strip()
            values = series[series != ""]
            unique_values = values.unique().tolist()
            num_missing = int(total_rows - len(values))

            num_numeric = int(sum(values.apply(lambda x: x.replace(".", "", 1).isdigit())))
            pct_numeric = float(num_numeric / len(values)) if len(values) else 0
            pct_unique = float(len(unique_values) / len(values)) if len(values) else 0

            if pct_numeric > 0.8:
                col_type = "numeric"
            elif pct_unique < 0.5 and len(unique_values) < 20:
                col_type = "categorical"
            else:
                col_type = "text"

            # Short text detection for text columns
            if col_type == "text":
                avg_words = values.apply(lambda x: len(x.split())).mean()
                is_short_text = bool(avg_words < 3)
            else:
                is_short_text = False

            # Determine if column is important
            if col_type == "categorical":
                # Heuristic: low unique count, low missing percentage
                pct_missing = num_missing / total_rows if total_rows > 0 else 1
                important = (len(unique_values) > 1) and (len(unique_values) < 20) and (pct_missing < 0.5)
            else:
                important = False

            sample_values = unique_values if col_type == "categorical" else unique_values[:5]

            result.append({
                "name": str(col),
                "type": str(col_type),
                "missingCount": int(num_missing),
                "uniqueCount": int(len(unique_values)),
                "sampleValues": [str(v) for v in sample_values],
                "isShortText": bool(is_short_text),
                "important": bool(important)
            })

        self.analysis_result = result

        # Apply row limit to raw data preview
        try:
            if limit != "all":
                row_limit = int(limit)
                limited_df = self.data_df.head(row_limit)
            else:
                limited_df = self.data_df
        except ValueError:
            limited_df = self.data_df.head(100)

        raw_data = limited_df.fillna("").astype(str).to_dict(orient="records")


        return {
            "filename": str(self.original_filename),
            "shape": (int(self.data_df.shape[0]), int(self.data_df.shape[1])),
            "columns": result,
            "rawData": raw_data
        }
    #  End of analyze method

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def group_by_category(self, threshold=75, match_mode="ratio", ml_mode=False):
        
        threshold =  int(threshold) if isinstance(threshold, str) else threshold
        ml_mode = str_to_bool(ml_mode)

        print("Running group_by_category with threshold:", threshold)
        print("Running group_by_category with mode:", match_mode)
        print("Running group_by_category with mlmode:", ml_mode)

        try:
            payload = cherrypy.request.json

            all_columns = payload.get("allColumns")
            category_col = payload.get("categoryColumn")
            raw_data = payload.get("rawData")
            filename = payload.get("filename")
            selected_cols = payload.get("selectedColumns")
            use_category = payload.get("useCategory", False)

            if not use_category:
                return {"error": "Category column not selected or use_category flag is False."}

            df = pd.DataFrame(raw_data)

            if category_col not in df.columns:
                return {"error": f"Category column '{category_col}' not found in dataset."}

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            long_text_cols = [
                col for col in df.select_dtypes(include=[object]).columns
                if df[col].dropna().map(len).mean() > 50
            ]

            short_text_cols = [
                col for col in selected_cols
                if col not in numeric_cols and col not in long_text_cols
            ]

            # Reset index for row tracking
            df_reset = df.reset_index(drop=False).rename(columns={'index': 'row_id'})

            # Melt short text columns
            df_long = df_reset.melt(
                id_vars=['row_id', category_col],
                value_vars=short_text_cols,
                var_name='short_text_col_name',
                value_name='short_text_value'
            )

            # Drop NaN values from processing but keep track of row_id and category
            df_long_clean = df_long.dropna(subset=['short_text_value']).copy()

            # Filter out NaN and empty strings
            filtered_df_long_clean = df_long_clean[df_long_clean['short_text_value'].notna() & (df_long_clean['short_text_value'].str.strip() != "")]

            # Group by category and compute required stats
            all_cluster_info_stats = {}

            grouped = filtered_df_long_clean.groupby('category')['short_text_value']

            # print('GROUPED:', grouped)

            for category, values in grouped:
                total_count = int(values.count())
                unique_count = values.nunique()
                all_cluster_info_stats[category] = {
                    'num_of_before_unique': unique_count,
                    'total_values': total_count
                }

            # print(df_long_clean)
            # print(all_cluster_info_stats)

            # Build category -> list of (value, row_id) tuples (preserve order)
            category_brand_dict = {}
            for cat, grp in df_long_clean.groupby(category_col):
                values = grp['short_text_value'].astype(str).tolist()
                row_ids = grp['row_id'].astype(int).tolist()
                short_text_col_name = grp['short_text_col_name'].astype(str).tolist()
                category_brand_dict[cat] = list(zip(values, row_ids, short_text_col_name))

                # print('CAT BRAND DICT', category_brand_dict)

            # Run your cleaning and clustering pipeline
            all_cluster_info, all_brand_to_suggestion = clean_category_dict(category_brand_dict, threshold, match_mode=match_mode, isML=ml_mode)

            # print("All cluster info:", all_cluster_info)

            # Now build the groupedData response structure including row_ids and cluster info
            grouped_data = {}

            for category, cluster_dict in all_cluster_info.items():
                grouped_data[category] = {}
                # Collect all row_ids of values in this category (for quick reference)
                row_ids = df_long_clean[df_long_clean[category_col] == category]['row_id'].astype(int).tolist()
                # grouped_data[category]['row_ids'] = row_ids

                # Add all cluster info under category (cluster_dict already contains members with single int row_ids)
                for cluster_tag, cluster_data in cluster_dict.items():
                    grouped_data[category][cluster_tag] = cluster_data

            return {
                "status": "success",
                "useCategory": True,
                "categoryColumn": category_col,
                "rawData": raw_data,
                "filename": filename,
                "dfLongClean": df_long_clean.to_dict(orient="records"),
                "groupedData": grouped_data,
                "groupedDataStats": all_cluster_info_stats
            }

        except Exception as e:
            cherrypy.log(f"Error in /group_by_category: {e}")
            return {"error": str(e)}
        # End of group_by_category method


    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.cors()
    def group_in_column(self, threshold=75, match_mode="ratio", ml_mode=False):
        """
        API endpoint to group values in given columns without regrouping the whole dataset.
        Returns which columns were selected for grouping.
        """
        # Short-circuit preflight request
        if cherrypy.request.method == 'OPTIONS':
            cherrypy.response.status = 200
            return ""
        
        threshold =  int(threshold) if isinstance(threshold, str) else threshold
        ml_mode = bool(ml_mode) if isinstance(ml_mode, str) else ml_mode

        print("Running group_in_column with threshold:", threshold)
        print("Running group_in_column with mode:", match_mode)

        try:
            # 1. Parse incoming request
            input_data = cherrypy.request.json

            data = input_data.get("rawData")  # Can be list of dicts or column-oriented dict
            selected_columns = input_data.get("selectedColumns", [])
            useCategory = False

            if not data or not isinstance(selected_columns, list) or not selected_columns:
                raise ValueError("Missing or invalid 'data' or 'selectedColumns' in request.")

            # 2. Run the clustering/grouping
            df = pd.DataFrame(data)
            grouped_data, grouped_data_stats  = cluster_short_text(df, selected_columns, threshold=threshold, match_mode=match_mode, isML=ml_mode)

            print("Cluster results:", grouped_data_stats)  # Debugging output

            # 3. Prepare response
            response = {
                "status": "success",
                "useCategory": useCategory,
                "rawData": data,
                "selectedColumns": selected_columns,
                "groupedData": grouped_data,
                "groupedDataStats": grouped_data_stats
            }

            return response

        except Exception as e:
            cherrypy.response.status = 400
            return {"error": str(e)}
        

    # Advanced suggestion API endpoints

    @cherrypy.expose
    @cherrypy.tools.json_in()   # Expect JSON input
    @cherrypy.tools.json_out()  # Return JSON output
    @cherrypy.tools.cors()
    def use_llm(self):
        """
        POST /use_google_gk
        Body:
        {
            "members": ["name1", "name2", ...],
            "context": "Hospital organization in the UK"
        }
        """

        # Short-circuit preflight request
        if cherrypy.request.method == 'OPTIONS':
            cherrypy.response.status = 200
            return ""

        # Get request data
        try:
            input_data = cherrypy.request.json
            members = input_data.get("members", [])
            context = input_data.get("context", "Organization")
        except Exception:
            cherrypy.response.status = 400
            return {"error": "Invalid JSON input"}

        if not members or not isinstance(members, list):
            cherrypy.response.status = 400
            return {"error": "'members' must be a non-empty list"}

        # Prepare prompt for Gemini
        prompt = f"""
        You are an AI that standardizes texts/phrases (brands, organisations, places, people, etc.).
        Given the following list of texts/phrases:
        {members}
        And providing this extra information about them: {context}.
        Return for me the most appropriate match to these texts.
        """

        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            suggestion = response.text.strip()
        except Exception as e:
            cherrypy.response.status = 500
            return {"error": f"Gemini API error: {str(e)}"}

        # Return response in desired structure
        return {
            "suggestion": suggestion,
            "members": members
        }
    # End of LLM Google Gemini function


    # Google Knowledge Graph canonicalization endpoint
    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.cors()
    def use_google_kg(self):
        """
        POST /use_google_kg
        Body:
        {
            "members": ["name1", "name2", ...],
            "context_type": "Hospital"  # optional, must match Google KG @type exactly
        }
        """

        # Short-circuit preflight request
        if cherrypy.request.method == 'OPTIONS':
            cherrypy.response.status = 200
            return ""

        try:
            input_data = cherrypy.request.json
            members = input_data.get("members", [])
            # context_type = input_data.get("context_type", "")
        except Exception:
            cherrypy.response.status = 400
            return {"error": "Invalid JSON input"}

        if not members or not isinstance(members, list):
            cherrypy.response.status = 400
            return {"error": "'members' must be a non-empty list"}

        canonical_name = None
        for name in members:
            # result = get_canonical_name(name, context_type)
            result = get_canonical_name(name)
            if result:
                canonical_name = result
                break

        if not canonical_name:
            cherrypy.response.status = 404
            return {"error": "No canonical name found", "members": members}

        return {
            "suggestion": canonical_name,
            "members": members
        }


    @cherrypy.expose
    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def apply_cleaning(self):
        try:
            payload = cherrypy.request.json

            df_long_clean_data = payload.get("dfLongClean", [])
            grouped_data = payload.get("groupedData", {})
            grouped_data_stats = payload.get("groupedDataStats", {})
            raw_data = payload.get("rawData", [])
            filename = payload.get("filename", "cleaned_data.csv")
            use_category = payload.get("useCategory", False)

            df_raw = pd.DataFrame(raw_data)

            # Track manual corrections count per column
            manual_corrections_count = {}

            if use_category:
                # ------------- EXISTING USE_CATEGORY LOGIC -------------
                df_long_clean = pd.DataFrame(df_long_clean_data)

                df_long_clean["short_text_col_name_cleaned"] = df_long_clean["short_text_col_name"] + "_cleaned"
                df_long_clean["short_text_value_cleaned"] = df_long_clean["short_text_value"]

                for category, clusters in grouped_data.items():
                    for cluster_id, cluster_info in clusters.items():
                        suggestion = cluster_info.get("suggestion")
                        members = cluster_info.get("members", [])
                        exceptions = {ex["index"]: ex["value"] for ex in cluster_info.get("exceptions", [])}

                        for idx, member in enumerate(members):
                            cleaned_value = exceptions.get(idx, suggestion)
                            row_id = member["row_ids"]
                            col_name = member["short_text_col"]

                            mask = (
                                (df_long_clean["row_id"] == row_id) &
                                (df_long_clean["short_text_col_name"] == col_name)
                            )
                            df_long_clean.loc[mask, "short_text_value_cleaned"] = cleaned_value

                            if idx in exceptions:
                                manual_corrections_count[col_name] = manual_corrections_count.get(col_name, 0) + 1

                # Pivot original and cleaned
                df_wide_original = df_long_clean.pivot(index="row_id", columns="short_text_col_name", values="short_text_value").reset_index()
                df_wide_cleaned = df_long_clean.pivot(index="row_id", columns="short_text_col_name", values="short_text_value_cleaned").reset_index()

                df_wide_cleaned.columns = [
                    col if col == "row_id" else f"{col}_cleaned"
                    for col in df_wide_cleaned.columns
                ]

                df_final = df_raw.copy()
                for col in list(df_raw.columns):
                    if col != "row_id" and col in df_wide_original.columns:
                        cleaned_col = f"{col}_cleaned"
                        if cleaned_col in df_wide_cleaned.columns:
                            insert_pos = df_final.columns.get_loc(col) + 1
                            df_final.insert(insert_pos, cleaned_col, df_wide_cleaned[cleaned_col])

                cleaned_columns = df_long_clean["short_text_col_name"].unique()

                # Build summary
                summary = []
                for category_name, clusters in grouped_data.items():
                    stats = grouped_data_stats.get(category_name, {})
                    total_vals = stats.get("total_values", 0)
                    num_of_before_unique = stats.get("num_of_before_unique", 0)
                    # manual_corr = sum(
                    #     1 for col, count in manual_corrections_count.items() if col == category_name
                    # )

                    num_of_clusters = len(clusters)
                    num_of_majority = 0
                    total_num_of_single = 0
                    num_of_spell_check = 0
                    num_of_global_manual = 0
                    num_of_gkg = 0
                    num_of_llm = 0

                    after_cleaned_values = set()

                    for cluster_info in clusters.values():
                        suggestion = cluster_info.get("suggestion")
                        suggestion_mode = cluster_info.get("suggestion_mode", "")
                        if suggestion: # only add non-null suggestions
                            after_cleaned_values.add(suggestion)

                            members = cluster_info.get("members", [])

                            if suggestion_mode == "majority":
                                num_of_majority += len(members)
                                # print('MAJORITY MEMBERS: ', members)
                            elif suggestion_mode == "single_member":
                                total_num_of_single += len(members)
                            elif suggestion_mode == "spell_checker":
                                num_of_spell_check += len(members)
                            elif suggestion_mode == "custom":
                                num_of_global_manual += len(members)
                            elif suggestion_mode == "google_kg":
                                num_of_gkg += len(members)
                            elif suggestion_mode == "llm_suggest":
                                num_of_llm += len(members)

                    num_of_after_unique = len(after_cleaned_values)
                    accepted = num_of_majority + num_of_spell_check + total_num_of_single
                    rejected = num_of_global_manual + num_of_gkg + num_of_llm
                    acceptance_ratio = round(accepted / (accepted + rejected), 3) if (accepted + rejected) > 0 else 0.0

                    summary.append({
                        "column": category_name,
                        "total_values": total_vals,
                        "num_of_before_unique": num_of_before_unique,
                        "num_of_after_unique": num_of_after_unique,
                        # "manual_corrections": manual_corr,
                        "num_of_clusters": num_of_clusters,
                        "num_of_majority": num_of_majority,
                        "total_num_of_single": total_num_of_single,
                        "num_of_spell_check": num_of_spell_check,
                        "num_of_global_manual": num_of_global_manual,
                        "num_of_gkg": num_of_gkg,
                        "num_of_llm": num_of_llm,
                        "acceptance_ratio": acceptance_ratio * 100 # as percentage
                    })

                total_acceptance_ratio = 0
                ave_acceptance_ratio = 0
                
                for sum in summary:
                    total_acceptance_ratio += sum['acceptance_ratio']
                if len(summary) > 0:
                    ave_acceptance_ratio = round(total_acceptance_ratio / len(summary), 3)

                # Final return
                return {
                    "filename": filename,
                    "timestamp": datetime.now().isoformat(),  # ISO 8601 format: "2025-08-27T14:45:30.123456"
                    "status": "success",
                    "originalData": df_raw.to_dict(orient="records"),
                    "cleanedData": df_final.to_dict(orient="records"),
                    "acceptance_ratio": ave_acceptance_ratio,
                    "useCategory": True,
                    "summary": summary
                }

                # return {
                #     "status": "success",
                #     "originalData": df_raw.to_dict(orient="records"),
                #     "cleanedData": df_final.to_dict(orient="records"),
                #     "summary": summary
                # }

            else:
                # ------------- NO CATEGORY CASE -------------
                df_final = df_raw.copy()

                # Assume grouped_data is structured as: {col_name: {cluster_id: {suggestion, members[], exceptions[]}}}
                for col_name, clusters in grouped_data.items():
                    cleaned_col_name = f"{col_name}_cleaned"
                    df_final[cleaned_col_name] = df_raw[col_name]  # start with original

                    for cluster_id, cluster_info in clusters.items():
                        suggestion = cluster_info.get("suggestion")
                        suggestion_mode = cluster_info.get("suggestion_mode")
                        original_suggestion_mode = cluster_info.get("original_suggestion_mode")
                        members = cluster_info.get("members", [])
                        exceptions = {ex["index"]: ex["value"] for ex in cluster_info.get("exceptions", [])}

                        for idx, member in enumerate(members):
                            row_id = member["row_ids"]
                            cleaned_value = exceptions.get(idx, suggestion)

                            mask = (df_final.index == row_id)
                            df_final.loc[mask, cleaned_col_name] = cleaned_value

                            if idx in exceptions:
                                manual_corrections_count[col_name] = manual_corrections_count.get(col_name, 0) + 1

                    # Insert cleaned col next to original
                    insert_pos = df_final.columns.get_loc(col_name) + 1
                    col_data = df_final.pop(cleaned_col_name)
                    df_final.insert(insert_pos, cleaned_col_name, col_data)

                # Build summary
                # Enhanced summary
                summary = []

                for col in grouped_data.keys():
                    col_clusters = grouped_data[col]
                    stats = grouped_data_stats.get(col, {})
                    
                    total_vals = stats.get("total_values", 0)
                    num_of_before_unique = stats.get("num_of_before_unique", 0)
                    
                    manual_corr = manual_corrections_count.get(col, 0)

                    # Initialize counters
                    num_of_clusters = len(col_clusters)
                    num_of_majority = 0
                    total_num_of_single = 0
                    num_of_spell_check = 0
                    num_of_global_manual = 0
                    num_of_gkg = 0
                    num_of_llm = 0

                    after_cleaned_values = set()

                    for cluster_info in col_clusters.values():
                        members = cluster_info.get("members", [])
                        suggestion_mode = cluster_info.get("suggestion_mode", "")
                        original_suggestion_mode = cluster_info.get("original_suggestion_mode", "")
                        suggestion = cluster_info.get("suggestion")

                        if suggestion is not None: # only add non-null suggestions
                            after_cleaned_values.add(suggestion)

                            # Count suggestion modes
                            if suggestion_mode == "majority":
                                num_of_majority += len(members)
                            elif suggestion_mode == "single_member":
                                total_num_of_single += len(members)
                            elif suggestion_mode == "spell_checker":
                                num_of_spell_check += len(members)
                            elif suggestion_mode == "custom":
                                num_of_global_manual += len(members)
                            elif suggestion_mode == "google_kg":
                                num_of_gkg += len(members)
                            elif suggestion_mode == "llm_suggest":
                                num_of_llm += len(members)

                    # Calculate post-cleaning unique count
                    num_of_after_unique = len(after_cleaned_values)

                    # Calculate acceptance ratio
                    accepted = num_of_majority + num_of_spell_check + total_num_of_single
                    rejected = num_of_global_manual + num_of_gkg + num_of_llm
                    acceptance_ratio = round(accepted / (accepted + rejected), 3) if (accepted + rejected) > 0 else 0.0

                    summary.append({
                        "column": col,
                        "total_values": total_vals,
                        "num_of_before_unique": num_of_before_unique,
                        "num_of_after_unique": num_of_after_unique,
                        "manual_corrections": manual_corr,
                        "num_of_clusters": num_of_clusters,
                        "num_of_majority": num_of_majority,
                        "total_num_of_single": total_num_of_single,
                        "num_of_spell_check": num_of_spell_check,
                        "num_of_global_manual": num_of_global_manual,
                        "num_of_gkg": num_of_gkg,
                        "num_of_llm": num_of_llm,
                        "acceptance_ratio": acceptance_ratio * 100 # as percentage
                    })

                total_acceptance_ratio = 0
                ave_acceptance_ratio = 0
                
                for sum in summary:
                    total_acceptance_ratio += sum['acceptance_ratio']
                if len(summary) > 0:
                    ave_acceptance_ratio = round(total_acceptance_ratio / len(summary), 3)

                
                return {
                    "status": "success",
                    "filename": filename,
                    "timestamp": datetime.now().isoformat(),  # ISO 8601 format: "2025-08-27T14:45:30.123456"
                    "originalData": df_raw.to_dict(orient="records"),
                    "cleanedData": df_final.to_dict(orient="records"),
                    "acceptance_ratio": ave_acceptance_ratio,
                    "summary": summary
                }

                # return {
                #     "status": "success",
                #     "originalData": df_raw.to_dict(orient="records"),
                #     "cleanedData": df_final.to_dict(orient="records"),
                #     "summary": summary
                # }

        except Exception as e:
            cherrypy.log(f"Error in /apply_cleaning: {e}")
            return {"error": str(e)}
        

    # Database and User Management Endpoints
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.expose
    def register(self):
        data = cherrypy.request.json
        required_fields = ["name", "email", "password"]
        if not all(data.get(f) for f in required_fields):
            return {"status": "error", "message": "Missing required fields"}

        # Required
        name = data["name"]
        email = data["email"]
        password = data["password"]

        # Optional
        job_title = data.get("job_title")
        location = data.get("location")
        organisation = data.get("organisation")
        bio = data.get("bio")
        member_plan = data.get("member_plan")
        profession = data.get("profession")

        # Auto-generate date
        create_date = datetime.now().isoformat()

        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users (
                        name, email, password, job_title, location,
                        organisation, bio, create_date, member_plan, profession
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    name, email, password, job_title, location,
                    organisation, bio, create_date, member_plan, profession
                ))
                conn.commit()
                user_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            return {"status": "error", "message": "Email already exists"}

        return {
            "status": "success",
            "message": "User registered",
            "user": {
                "id": user_id,
                "name": name,
                "email": email,
                "job_title": job_title,
                "location": location,
                "organisation": organisation,
                "bio": bio,
                "create_date": create_date,
                "member_plan": member_plan,
                "profession": profession
            }
        }

    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.expose
    @cherrypy.tools.cors()
    def login(self):
        # Short-circuit preflight request
        if cherrypy.request.method == 'OPTIONS':
            cherrypy.response.status = 200
            return ""
        
        data = cherrypy.request.json
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return {"status": "error", "message": "Email and password are required"}

        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Fetch user
            cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
            user_row = cursor.fetchone()
            if not user_row:
                return {"status": "error", "message": "User not found"}

            if user_row["password"] != password:
                return {"status": "error", "message": "Incorrect password"}

            user = dict(user_row)
            user_id = user["id"]
            user.pop("password", None)

            # Fetch all cleans for the user
            cursor.execute("""
                SELECT * FROM cleans WHERE user_id = ?
            """, (user_id,))
            cleans = []
            clean_rows = cursor.fetchall()

            for clean_row in clean_rows:
                clean = dict(clean_row)
                clean_id = clean["clean_id"]

                # Fetch associated summaries
                cursor.execute("""
                    SELECT * FROM summaries WHERE clean_id = ?
                """, (clean_id,))
                summaries = [dict(row) for row in cursor.fetchall()]

                clean["summaries"] = summaries
                cleans.append(clean)

            user["cleans"] = cleans

        return {
            "status": "success",
            "user": user
        }
    

    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.expose
    @cherrypy.tools.cors()
    def clean_save(self):
        # Short-circuit preflight request
        if cherrypy.request.method == 'OPTIONS':
            cherrypy.response.status = 200
            return ""
        data = cherrypy.request.json

        # Required fields
        user_id = data.get("user_id")
        file_name = data.get("file_name")
        cleaning_mode = data.get("cleaning_mode")  # 'category' or 'direct'
        acceptance_ratio = data.get("acceptance_ratio")
        summaries = data.get("summaries", [])

        if not all([user_id, file_name, cleaning_mode, summaries]):
            return {"status": "error", "message": "Missing required fields"}

        clean_date = datetime.now().isoformat()

        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()

                # Insert clean record
                cursor.execute("""
                    INSERT INTO cleans (
                        file_name, clean_date, cleaning_mode, acceptance_ratio, user_id
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    file_name, clean_date, cleaning_mode, acceptance_ratio, user_id
                ))
                clean_id = cursor.lastrowid

                # Insert summary rows
                inserted_summary_ids = []
                for s in summaries:
                    cursor.execute("""
                        INSERT INTO summaries (
                            clean_id, column, total_values, num_of_before_unique,
                            num_of_after_unique, manual_corrections, num_of_clusters,
                            num_of_majority, total_num_of_single, num_of_spell_check,
                            num_of_global_manual, num_of_gkg, num_of_llm, acceptance_ratio
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        clean_id,
                        s.get("column"),
                        s.get("total_values"),
                        s.get("num_of_before_unique"),
                        s.get("num_of_after_unique"),
                        s.get("manual_corrections"),
                        s.get("num_of_clusters"),
                        s.get("num_of_majority"),
                        s.get("total_num_of_single"),
                        s.get("num_of_spell_check"),
                        s.get("num_of_global_manual"),
                        s.get("num_of_gkg"),
                        s.get("num_of_llm"),
                        s.get("acceptance_ratio")
                    ))
                    inserted_summary_ids.append(cursor.lastrowid)

                conn.commit()

        except Exception as e:
            return {"status": "error", "message": str(e)}

        return {
            "status": "success",
            "message": "Clean and summaries saved",
            "clean_id": clean_id,
            "summary_ids": inserted_summary_ids
        }
    
    @cherrypy.tools.json_out()
    @cherrypy.expose
    @cherrypy.tools.cors()
    def cleans_all(self, user_id=None):
        # Short-circuit preflight request
        if cherrypy.request.method == 'OPTIONS':
            cherrypy.response.status = 200
            return ""
        if not user_id:
            return {"status": "error", "message": "user_id is required in query parameters"}

        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Fetch all cleans by the user
                cursor.execute("""
                    SELECT * FROM cleans WHERE user_id = ?
                    ORDER BY clean_date DESC
                """, (user_id,))
                cleans = []

                clean_rows = cursor.fetchall()
                for clean_row in clean_rows:
                    clean = dict(clean_row)
                    clean_id = clean["clean_id"]

                    # Fetch summaries for this clean
                    cursor.execute("""
                        SELECT * FROM summaries WHERE clean_id = ?
                    """, (clean_id,))
                    summaries = [dict(row) for row in cursor.fetchall()]
                    clean["summaries"] = summaries

                    cleans.append(clean)

            return {
                "status": "success",
                "user_id": user_id,
                "cleans": cleans
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}



    # Handle CORS preflight requests
    @cherrypy.expose
    @cherrypy.tools.cors()
    def OPTIONS(self, *args, **kwargs):
        cherrypy.response.status = 200
        return ""
    

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    config = {
        "/": {
            "tools.sessions.on": True,
            "tools.response_headers.on": True,
            "tools.cors.on": True,
            "tools.response_headers.headers": [("Content-Type", "application/json")]
        },
        "/download_cleaned": {
            "tools.response_headers.on": True,
            "tools.response_headers.headers": [("Content-Type", "text/csv")]
        }
    }

    cherrypy.quickstart(CleanlifyAPI(), "/", config)
