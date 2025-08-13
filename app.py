import os
import re
import json
import string
import requests
from io import StringIO

import cherrypy
from cherrypy.lib import static

import numpy as np
import pandas as pd
from spellchecker import SpellChecker
from fuzzywuzzy import process
from rapidfuzz import fuzz
from collections import Counter, defaultdict
from difflib import SequenceMatcher

import google.generativeai as genai

# Configure your Gemini API key here
GEMINI_API_KEY = ""  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

# Google Knowledge Graph API settings
GOOGLE_KG_API_KEY = ""
GOOGLE_KG_API_URL = "https://kgsearch.googleapis.com/v1/entities:search"

# CORS tool to allow cross-origin requests
def cors():
    cherrypy.response.headers["Access-Control-Allow-Origin"] = "*"
    cherrypy.response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    cherrypy.response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    cherrypy.response.headers["Access-Control-Max-Age"] = "86400"

cherrypy.tools.cors = cherrypy.Tool('before_handler', cors)


# Directory to store uploaded files
UPLOAD_DIR = os.path.join(os.getcwd(), 'uploads')

spell = SpellChecker(distance=2)  # small edit distance

# ---------- STEP 0: Helpers ----------
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

# --- Similarity helper ---
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def correct_phrase(text):
    """Apply spell correction to each word."""
    words = text.split()
    corrected = [spell.correction(w) or w for w in words]
    return ' '.join(corrected)


# ---------- STEP 1: Clustering ----------
def cluster_brands(preprocessed_list, threshold=90):
    clusters = []
    # Example simplified clustering logic
    while preprocessed_list:
        base = preprocessed_list.pop(0)
        cluster = [base]
        base_value = base[0]

        rest = []
        for item in preprocessed_list:
            value = item[0]
            similarity = fuzz.ratio(base_value, value)
            if similarity >= threshold:
                cluster.append(item)
            else:
                rest.append(item)
        preprocessed_list = rest
        clusters.append(cluster)
    
    # print("Clusters formed:", clusters)  # Debugging output

    return clusters


# ---------- STEP 2: Process clusters ----------
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
            'confidence': round(confidence, 2) if confidence is not None else None
        }

        # Map each cleaned value -> cluster_tag (last assignment wins, but values in same cluster are consistent)
        for v in values:
            brand_to_cluster[v] = cluster_tag

    return cluster_dict, brand_to_cluster


# ---------- STEP 3: Full pipeline ----------
def clean_category_dict(category_brand_dict):
    all_cluster_info = {}
    all_brand_to_suggestion = {}

    for category, brand_list in category_brand_dict.items():
        # brand_list is expected to be list of (value, row_id) just added short_text_col
        cleaned_pairs = preprocess_list(brand_list)  # returns list of (cleaned_value, row_id), just added short_text_col

        # filter out NaNs (preprocess_list marks them with np.nan)
        cleaned_pairs = [p for p in cleaned_pairs if not pd.isna(p[0])]

        # Cluster using tuple-aware function
        clusters = cluster_brands(cleaned_pairs, threshold=75)

        cluster_dict, brand_to_cluster = process_clusters(clusters)

        # Map cleaned value -> suggestion
        brand_to_suggestion = {}
        for cleaned_val, cluster_tag in brand_to_cluster.items():
            suggestion = cluster_dict[cluster_tag]['suggestion']
            brand_to_suggestion[cleaned_val] = suggestion

        all_cluster_info[category] = cluster_dict
        all_brand_to_suggestion[category] = brand_to_suggestion

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



def cluster_items(values_with_meta, threshold=90):
    """Cluster items (with row ids and column names) by fuzzy similarity."""
    clusters = []
    for item in values_with_meta:
        if pd.isna(item['value']):
            continue
        found_cluster = False
        for cluster in clusters:
            if any(fuzz.ratio(item['value'], member['value']) >= threshold for member in cluster):
                cluster.append(item)
                found_cluster = True
                break
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
            "confidence": round(confidence, 2) if confidence is not None else None
        }

        for member in cluster:
            value_to_cluster[member['value']] = cluster_tag

    return cluster_dict, value_to_cluster


# ---------- Clustering Short Text Columns ----------
def cluster_short_text(data, columns, threshold=90):
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

        clusters = cluster_items(values_with_meta, threshold=threshold)
        cluster_info, mapping = process_clusters_cols(clusters)

        all_cluster_info[col] = cluster_info

        # mapping[col] = {
        #     "mapping": mapping
        # }

    return all_cluster_info


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
        from io import StringIO
        import pandas as pd

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
    def group_by_category(self):
        try:
            payload = cherrypy.request.json

            all_columns = payload.get("allColumns")
            category_col = payload.get("categoryColumn")
            raw_data = payload.get("rawData")
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

            print(df_long_clean)

            # Build category -> list of (value, row_id) tuples (preserve order)
            category_brand_dict = {}
            for cat, grp in df_long_clean.groupby(category_col):
                values = grp['short_text_value'].astype(str).tolist()
                row_ids = grp['row_id'].astype(int).tolist()
                short_text_col_name = grp['short_text_col_name'].astype(str).tolist()
                category_brand_dict[cat] = list(zip(values, row_ids, short_text_col_name))

                # print(category_brand_dict)

            # Run your cleaning and clustering pipeline
            all_cluster_info, all_brand_to_suggestion = clean_category_dict(category_brand_dict)

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
                "rawData": raw_data,
                "dfLongClean": df_long_clean.to_dict(orient="records"),
                "groupedData": grouped_data
            }

        except Exception as e:
            cherrypy.log(f"Error in /group_by_category: {e}")
            return {"error": str(e)}
        # End of group_by_category method


    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.cors()
    def group_in_column(self):
        """
        API endpoint to group values in given columns without regrouping the whole dataset.
        Returns which columns were selected for grouping.
        """
        # Short-circuit preflight request
        if cherrypy.request.method == 'OPTIONS':
            cherrypy.response.status = 200
            return ""
        
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
            grouped_data = cluster_short_text(df, selected_columns)

            # print("Cluster results:", grouped_data)  # Debugging output

            # 3. Prepare response
            response = {
                "status": "success",
                "useCategory": useCategory,
                "rawData": data,
                "selectedColumns": selected_columns,
                "groupedData": grouped_data
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
        You are an AI that standardizes organization names.
        Given the following list of names:

        {members}

        These names refer to the same {context}.
        Return ONLY the single correct official name without extra words.
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
            raw_data = payload.get("rawData", [])
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
                summary = []
                for col in cleaned_columns:
                    total_vals = df_long_clean[df_long_clean["short_text_col_name"] == col].shape[0]
                    manual_corr = manual_corrections_count.get(col, 0)
                    summary.append({"column": col, "total_values": total_vals, "manual_corrections": manual_corr})

                return {
                    "status": "success",
                    "originalData": df_raw.to_dict(orient="records"),
                    "cleanedData": df_final.to_dict(orient="records"),
                    "summary": summary
                }

            else:
                # ------------- NO CATEGORY CASE -------------
                df_final = df_raw.copy()

                # Assume grouped_data is structured as: {col_name: {cluster_id: {suggestion, members[], exceptions[]}}}
                for col_name, clusters in grouped_data.items():
                    cleaned_col_name = f"{col_name}_cleaned"
                    df_final[cleaned_col_name] = df_raw[col_name]  # start with original

                    for cluster_id, cluster_info in clusters.items():
                        suggestion = cluster_info.get("suggestion")
                        members = cluster_info.get("members", [])
                        exceptions = {ex["index"]: ex["value"] for ex in cluster_info.get("exceptions", [])}

                        for idx, member in enumerate(members):
                            row_id = member["row_ids"]
                            cleaned_value = exceptions.get(idx, suggestion)

                            print('Suggestion:', cleaned_value)

                            mask = (df_final.index == row_id)
                            df_final.loc[mask, cleaned_col_name] = cleaned_value

                            if idx in exceptions:
                                manual_corrections_count[col_name] = manual_corrections_count.get(col_name, 0) + 1

                    # Insert cleaned col next to original
                    insert_pos = df_final.columns.get_loc(col_name) + 1
                    col_data = df_final.pop(cleaned_col_name)
                    df_final.insert(insert_pos, cleaned_col_name, col_data)

                # Build summary
                summary = []
                for col in grouped_data.keys():
                    total_vals = df_raw[col].shape[0]
                    manual_corr = manual_corrections_count.get(col, 0)
                    summary.append({"column": col, "total_values": total_vals, "manual_corrections": manual_corr})

                return {
                    "status": "success",
                    "originalData": df_raw.to_dict(orient="records"),
                    "cleanedData": df_final.to_dict(orient="records"),
                    "summary": summary
                }

        except Exception as e:
            cherrypy.log(f"Error in /apply_cleaning: {e}")
            return {"error": str(e)}


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
