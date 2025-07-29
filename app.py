import cherrypy
import os
import json
import pandas as pd
from cherrypy.lib import static
from io import StringIO
from spellchecker import SpellChecker

UPLOAD_DIR = os.path.join(os.getcwd(), 'uploads')

class CleanlifyAPI:
    def __init__(self):
        self.data = None
        self.filtered_text_df = None
        self.text_columns_to_clean = []
        self.cleaned_text_df = None
        self.original_filename = None
        self.analysis_result = None

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def index(self):
        return {"message": "Welcome to the re-implemented Cleanlify API!"}
    

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def analyze(self, file, limit="100"):
        """Upload + Analyze CSV: Suggest fields and flag problematic text columns, with row limit."""
        import pandas as pd
        from io import StringIO

        contents = file.file.read().decode("utf-8")
        self.data = pd.read_csv(StringIO(contents))
        self.original_filename = file.filename

        # Rename Unnamed columns to empty string
        self.data.columns = ["" if col.startswith("Unnamed") else col for col in self.data.columns]

        result = []
        total_rows = len(self.data)

        for col in self.data.columns:
            series = self.data[col].dropna().astype(str).str.strip()
            values = series[series != ""]
            unique_values = values.unique().tolist()
            num_missing = total_rows - len(values)

            # Determine type
            num_numeric = sum(values.apply(lambda x: x.replace(".", "", 1).isdigit()))
            pct_numeric = num_numeric / len(values) if len(values) else 0
            pct_unique = len(unique_values) / len(values) if len(values) else 0

            if pct_numeric > 0.8:
                col_type = "numeric"
            elif pct_unique < 0.5 and len(unique_values) < 20:
                col_type = "categorical"
            else:
                col_type = "text"

            # Mark problematic: text fields with high uniqueness
            is_problematic = col_type == "text" and pct_unique > 0.7

            result.append({
                "name": col,
                "type": col_type,
                "missingCount": int(num_missing),
                "uniqueCount": len(unique_values),
                "sampleValues": unique_values[:5],
                "isProblematic": is_problematic
            })

        self.analysis_result = result

        # Determine how many rows to return
        if limit.lower() == "all":
            limited_df = self.data.copy()
        else:
            try:
                row_limit = int(limit)
                limited_df = self.data.head(row_limit)
            except ValueError:
                limited_df = self.data.head(100)  # fallback

        raw_data = limited_df.fillna("").to_dict(orient="records")

        return {
            "filename": self.original_filename,
            "shape": self.data.shape,
            "columnsAnalyzed": result,
            "rawData": raw_data
        }



    # @cherrypy.expose
    # @cherrypy.tools.json_out()
    # def upload_and_preview(self, file):
    #     filename = file.filename
    #     contents = file.file.read().decode("utf-8")

    #     self.data = pd.read_csv(StringIO(contents))

    #     preview = self.data.head(5).fillna("").to_dict(orient="records")

    #     return {
    #         "filename": filename,
    #         "shape": self.data.shape,
    #         "columns": self.data.columns.tolist(),
    #         "preview": preview
    #     }

    # @cherrypy.expose
    # @cherrypy.tools.json_out()
    # def suggest_text_columns(self):
    #     if self.data is None:
    #         raise cherrypy.HTTPError(400, "No data uploaded")

    #     # Drop numeric columns
    #     non_numeric = self.data.select_dtypes(exclude=['number'])

    #     # Filter: keep short free-text style columns
    #     text_cols = []
    #     for col in non_numeric.columns:
    #         unique_vals = non_numeric[col].dropna().unique()
    #         avg_len = non_numeric[col].dropna().astype(str).map(len).mean()

    #         if len(unique_vals) > 10 and avg_len < 50:
    #             text_cols.append(col)

    #     self.filtered_text_df = non_numeric[text_cols].copy()
    #     self.text_columns_to_clean = text_cols

    #     return {
    #         "text_columns_suggested": text_cols,
    #         "filtered_preview": self.filtered_text_df.head(5).fillna("").to_dict(orient="records")
    #     }

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def clean_text(self):
        if self.filtered_text_df is None:
            raise cherrypy.HTTPError(400, "Run suggest_text_columns first")

        spell = SpellChecker()

        def clean_entry(text):
            if pd.isna(text):
                return text
            words = str(text).strip().split()
            cleaned_words = []
            for word in words:
                cleaned_word = spell.correction(word) or word
                cleaned_words.append(cleaned_word)
            return " ".join(cleaned_words)

        cleaned_df = self.filtered_text_df.copy()
        for col in self.text_columns_to_clean:
            cleaned_df[f"{col}_cleaned"] = cleaned_df[col].apply(clean_entry)

        # Merge cleaned columns into original dataset
        for col in self.text_columns_to_clean:
            self.data[f"{col}_cleaned"] = cleaned_df[f"{col}_cleaned"]

        self.cleaned_text_df = self.data

        return {
            "columns_cleaned": [f"{col}_cleaned" for col in self.text_columns_to_clean],
            "cleaned_preview": self.cleaned_text_df.head(5).fillna("").to_dict(orient="records")
        }

    @cherrypy.expose
    def download_cleaned(self):
        if self.cleaned_text_df is None:
            raise cherrypy.HTTPError(400, "No cleaned data available")

        cherrypy.response.headers['Content-Type'] = 'text/csv'
        cherrypy.response.headers['Content-Disposition'] = 'attachment; filename="cleaned_data.csv"'
        return self.cleaned_text_df.to_csv(index=False).encode("utf-8")

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    config = {
        "/": {
            "tools.sessions.on": True,
            "tools.response_headers.on": True,
            "tools.response_headers.headers": [("Content-Type", "application/json")]
        },
        "/download_cleaned": {
            "tools.response_headers.on": True,
            "tools.response_headers.headers": [("Content-Type", "text/csv")]
        }
    }

    cherrypy.quickstart(CleanlifyAPI(), "/", config)