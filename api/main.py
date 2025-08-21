from fastapi import FastAPI, HTTPException, Query
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from .db import get_snowflake_connection, comments_collection
from .models import Comment
import numpy as np
from functools import lru_cache
from cachetools import TTLCache

# ---------------------------------------------------------------------
# Cache configurations
# ---------------------------------------------------------------------
# TTL (seconds) — adjust based on acceptable staleness
COVID_SUMMARY_CACHE = TTLCache(maxsize=100, ttl=300)   # 5 min
CLUSTER_CACHE = TTLCache(maxsize=10, ttl=300)          # 5 min
COMMENTS_CACHE = TTLCache(maxsize=500, ttl=60)         # 1 min
HISTORICAL_CACHE = TTLCache(maxsize=100, ttl=3600)     # 1 hour

app = FastAPI(title="COVID-19 Data API")


# ---------------------------------------------------------------------
# Helper function for Snowflake queries
# ---------------------------------------------------------------------
def fetch_from_snowflake(query: str, params=None):
    conn = None
    try:
        conn = get_snowflake_connection()
        return pd.read_sql(query, conn, params=params)
    finally:
        if conn:
            conn.close()


# ---------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------
@app.get("/covid-data/")
def get_covid_data(country: str):
    """Fetch COVID-19 summary data for a given country (cached)."""
    cache_key = country.lower().strip()
    if cache_key in COVID_SUMMARY_CACHE:
        return COVID_SUMMARY_CACHE[cache_key]

    try:
        query = """
            SELECT COUNTRY_REGION, SUM(CASES) AS TOTAL_CASES, SUM(DEATHS) AS TOTAL_DEATHS
            FROM ECDC_GLOBAL
            WHERE LOWER(COUNTRY_REGION) = LOWER(%(country)s)
            GROUP BY COUNTRY_REGION
        """
        df = fetch_from_snowflake(query, {"country": country})
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No COVID data found for country '{country}'.")
        result = df.to_dict(orient='records')[0]
        COVID_SUMMARY_CACHE[cache_key] = result
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching COVID data: {e}")


@app.get("/cluster-countries/")
def cluster_countries(k: int = Query(3, ge=2, le=10)):
    """Cluster countries based on COVID cases and deaths (cached per k)."""
    cache_key = f"k={k}"
    if cache_key in CLUSTER_CACHE:
        return CLUSTER_CACHE[cache_key]

    try:
        query = """
            SELECT COUNTRY_REGION, SUM(CASES) AS TOTAL_CASES, SUM(DEATHS) AS TOTAL_DEATHS
            FROM ECDC_GLOBAL
            GROUP BY COUNTRY_REGION
            HAVING SUM(CASES) > 0
        """
        df = fetch_from_snowflake(query)
        if df.empty:
            raise HTTPException(status_code=404, detail="No COVID data available for clustering.")

        # Log transform to reduce skew
        features = df[["TOTAL_CASES", "TOTAL_DEATHS"]].applymap(lambda x: np.log1p(x)).values
        scaled_features = StandardScaler().fit_transform(features)

        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)

        df["cluster"] = clusters
        result = {"clusters": df[["COUNTRY_REGION", "cluster"]].to_dict(orient="records"), "k": k}
        CLUSTER_CACHE[cache_key] = result
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing clustering: {e}")


@app.get("/comments/")
def get_comments(country_key: str):
    """Retrieve comments for a given normalized country key (cached)."""
    normalized_key = country_key.lower().strip()
    if normalized_key in COMMENTS_CACHE:
        return COMMENTS_CACHE[normalized_key]

    try:
        comments = list(comments_collection.find({"country_key": normalized_key}))
        for c in comments:
            c["_id"] = str(c["_id"])
        result = {"country_key": normalized_key, "comments": comments, "count": len(comments)}
        COMMENTS_CACHE[normalized_key] = result
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving comments: {e}")


@app.post("/comments/")
def add_comment(comment: Comment):
    """Add a new comment and clear related cache."""
    try:
        doc = comment.dict()
        doc['country_key'] = doc['country_key'].lower().strip()
        doc['timestamp'] = datetime.utcnow()
        result = comments_collection.insert_one(doc)

        # Invalidate cache for that country
        COMMENTS_CACHE.pop(doc['country_key'], None)

        return {"message": "Comment added successfully", "inserted_id": str(result.inserted_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding comment: {e}")


@app.get("/historical-data/")
def get_historical_data(country: str):
    """Fetch historical COVID-19 case data for a given country ordered by date (cached)."""
    cache_key = country.lower().strip()
    if cache_key in HISTORICAL_CACHE:
        return HISTORICAL_CACHE[cache_key]

    try:
        query = """
            SELECT DATE, CASES
            FROM ECDC_GLOBAL
            WHERE LOWER(COUNTRY_REGION) = LOWER(%(country)s)
            ORDER BY DATE
        """
        df = fetch_from_snowflake(query, {"country": country})
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for country '{country}'.")
        result = {"country": country, "historical_data": df.to_dict(orient='records'), "count": len(df)}
        HISTORICAL_CACHE[cache_key] = result
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching historical data: {e}")
