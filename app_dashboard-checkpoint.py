# D:\CAPSTONE_FINAL\app_dashboard.py
# Inventory + Sentiment Dashboard + Recommendations tab (complete & robust)
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import numpy as np
import joblib
import lightgbm as lgb

# --- CONFIG ---
from pathlib import Path

from pathlib import Path

# robust base path for local dev and Streamlit Cloud
# prefer: project root where the app file is executed
BASE = Path.cwd()                # <--- use this
# If you prefer relative to the script file and _file_ is available, fallback:
# try:
#     BASE = Path(_file_).resolve().parent
# except NameError:
#     BASE = Path.cwd()

DATA_PATH = BASE / "final_merged_inventory_sentiment.csv"
RERANK_PATH = BASE / "reranker_small.csv"   # or your filename
MODEL_PATH = BASE / "lightgbm_recommender.txt"
DEFAULT_RERANK = BASE / "data" / "reranker_train.csv"  # <- original (big) file

# choose the small sample if present, otherwise use the original path
RERANK_PATH = SAMPLE_RERANK if SAMPLE_RERANK.exists() else DEFAULT_RERANK# Model path can be either a joblib .pkl with {"model":..., "features":...} or a LightGBM text model.
MODEL_PATH = BASE / "models" / "reranker_lgbm.pkl"   # primary
LEGACY_MODEL_PATH = BASE / "data" / "lightgbm_recommender.txt"  # fallback (older path)

st.set_page_config(page_title="Inventory & Sentiment Dashboard", layout="wide")
st.title("📊 Inventory & Sentiment Analytics Dashboard")
st.markdown("Combines sales forecasts, sentiment analysis, inventory metrics and personalized recommendations.")

# --- DATA LOAD ---
@st.cache_data(show_spinner=False)
def load_main_data(path=DATA_PATH):
    df = pd.read_csv(path)
    # safe numeric conversions
    for c in ["avg_pos_prob","avg_rating","predicted_sales_next_28_days","initial_inventory","inventory_score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    # create combined inventory health metric
    df.loc[:, "inventory_health"] = df["inventory_score"] * df["avg_pos_prob"] * (df["avg_rating"] / 5.0)
    return df

@st.cache_data(show_spinner=False)
def load_rerank_data(path=RERANK_PATH):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df

df = load_main_data()
rerank_df = load_rerank_data()

st.success(f"Loaded main dataset with {len(df)} items.")

# --- FILTERS ---
category = st.selectbox("📂 Select Category", ["All"] + sorted(df["category"].unique().tolist()))
df_view = df.copy()
if category != "All":
    df_view = df_view[df_view["category"] == category]

st.sidebar.header("Filter Options")
min_sent = st.sidebar.slider("Minimum Avg Sentiment", 0.0, 1.0, 0.5, 0.05)
min_score = st.sidebar.slider("Minimum Inventory Score", 0.0, 1.0, 0.0, 0.05)
df_view = df_view[(df_view["avg_pos_prob"] >= min_sent) & (df_view["inventory_score"] >= min_score)]

# --- KPIs ---
col1, col2, col3 = st.columns(3)
col1.metric("📦 Total Items", len(df_view))
col2.metric("⭐ Avg Rating", f"{df_view['avg_rating'].mean():.2f}")
col3.metric("📈 Avg Predicted Sales (next 28 days)", f"{df_view['predicted_sales_next_28_days'].mean():.1f}")

# --- TABS: Charts & Alerts ---
tab1, tab2, tab3, tab4 = st.tabs(["Top Performers", "Sentiment vs Inventory", "Correlation Matrix", "Restock & Overstock Alerts"])

with tab1:
    st.subheader("🏆 Top 10 Products by Sentiment & Sales")
    top = df_view.sort_values(["avg_pos_prob", "predicted_sales_next_28_days"], ascending=False).head(10)
    fig = px.bar(top, x="title", y="avg_pos_prob", color="predicted_sales_next_28_days",
                 hover_data=["avg_rating", "initial_inventory", "inventory_score"], title="Top Sentiment & Sales Items")
    fig.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("💬 Sentiment vs Inventory Health")
    fig2 = px.scatter(df_view, x="avg_pos_prob", y="inventory_score", size="predicted_sales_next_28_days",
                      color="avg_rating", hover_name="title", title="Sentiment vs Inventory Score")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("📊 Feature Correlations")
    corr_cols = [c for c in ["avg_pos_prob","avg_rating","predicted_sales_next_28_days","inventory_score"] if c in df_view.columns]
    if len(corr_cols) >= 2:
        corr = df_view[corr_cols].corr()
        fig3 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Correlation Heatmap")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation matrix.")

with tab4:
    # --- REPLACEMENT RESTOCK / OVERSTOCK LOGIC (paste here) ---
    st.subheader("🚨 AI-Driven Inventory Alerts (improved)")

    # Parameters (tweakable)
    lookahead_days = 28
    lead_time_days = 14               # expected supplier lead time
    reorder_point_days = 14           # target days-of-stock threshold to trigger reorder
    overstock_days = 90               # if days_of_stock > this -> candidate overstock
    high_sales_quantile = 0.75
    low_sentiment_quantile = 0.25
    min_dos = 0.01                    # avoid divide-by-zero

    # Compute simple derived metrics
    df = df.copy()
    df["pred_next_per_day"] = df["predicted_sales_next_28_days"] / float(lookahead_days)
    df["pred_next_per_day"] = df["pred_next_per_day"].replace(0, 1e-6)  # avoid zero
    df["days_of_stock"] = df["initial_inventory"] / (df["pred_next_per_day"].clip(lower=min_dos))
    df["safety_stock"] = df["pred_next_per_day"] * lead_time_days * 0.5   # 50% of lead-time demand as simple safety stock

    # thresholds from data
    high_sales_threshold = df["predicted_sales_next_28_days"].quantile(high_sales_quantile)
    low_sentiment_threshold = df["avg_pos_prob"].quantile(low_sentiment_quantile)

    # Restock criteria:
    # - days_of_stock below reorder point
    # - predicted demand high (above quantile)
    # - positive sentiment not terrible (avoid restocking poor items)
    restock = df[
        (df["days_of_stock"] <= reorder_point_days) &
        (df["predicted_sales_next_28_days"] >= high_sales_threshold) &
        (df["avg_pos_prob"] >= 0.4)
    ].copy()

    # Priority score: how many units short for lead_time, weighted by sentiment & inventory score & margin if present
    # (positive -> higher priority)
    restock["short_units"] = (df["pred_next_per_day"] * lead_time_days) - df["initial_inventory"]
    restock["short_units"] = restock["short_units"].clip(lower=0.0)
    # margin factor if price/cost available, else 1.0
    if "price" in restock.columns and "cost" in restock.columns:
        restock["margin_factor"] = (restock["price"] - restock["cost"]).clip(lower=0.0) / (restock["price"].clip(lower=1.0))
    else:
        restock["margin_factor"] = 1.0
    restock["shortage_score"] = restock["short_units"] * restock["avg_pos_prob"] * (1.0 + restock["inventory_score"]) * restock["margin_factor"]
    restock = restock.sort_values("shortage_score", ascending=False)

    # Overstock criteria:
    # - very high days_of_stock OR (high inventory + low sentiment)
    overstock = df[
        (df["days_of_stock"] >= overstock_days) |
        ((df["initial_inventory"] > df["predicted_sales_next_28_days"]) & (df["avg_pos_prob"] <= low_sentiment_threshold))
    ].copy()
    # risk score larger -> more urgent to act (discount / flash sale)
    overstock["overstock_risk"] = (overstock["initial_inventory"] - (overstock["pred_next_per_day"] * lookahead_days)).clip(lower=0.0) * (1.0 - overstock["avg_pos_prob"]) * (1.0 + (1.0 - overstock["inventory_score"]))
    overstock = overstock.sort_values("overstock_risk", ascending=False)

    # Display summary and tables (unchanged UI calls)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🩸 Restock Needed (priority sorted)")
        st.metric("Count", len(restock))
        st.dataframe(
            restock[["item_id","title","avg_pos_prob","predicted_sales_next_28_days","initial_inventory","days_of_stock","short_units","shortage_score"]].head(50),
            use_container_width=True
        )
        if len(restock) > 0:
            fig4 = px.bar(
                restock.head(10),
                x="title",
                y="shortage_score",
                color="avg_pos_prob",
                hover_data=["short_units","days_of_stock","predicted_sales_next_28_days"],
                title="Top Restock Candidates (by shortage_score)",
            )
            st.plotly_chart(fig4, use_container_width=True)

    with col2:
        st.markdown("### 🧊 Overstock Risk (priority sorted)")
        st.metric("Count", len(overstock))
        st.dataframe(
            overstock[["item_id","title","avg_pos_prob","predicted_sales_next_28_days","initial_inventory","days_of_stock","overstock_risk"]].head(50),
            use_container_width=True
        )
        if len(overstock) > 0:
            fig5 = px.bar(
                overstock.head(10),
                x="title",
                y="overstock_risk",
                color="avg_pos_prob",
                hover_data=["days_of_stock","initial_inventory"],
                title="Top Overstock Risks",
            )
            st.plotly_chart(fig5, use_container_width=True)

    # Helpful short text explaining logic
    st.markdown(
        """
        *Notes:*  
        - days_of_stock = initial_inventory / (predicted daily demand).  
        - Restock is prioritized by shortage_score = short_units × sentiment × (1+inventory_score) × margin.  
        - Overstock risk uses inventory gap × (1 - sentiment) to prioritize promotions/discounts.
        """
    )
    # --- END replacement block ---
# --- Recommendations: load model + rerank data ---
st.header("🔎 Personalized Recommendations (Reranker)")

@st.cache_resource(ttl=600)
def load_recommender(model_path=MODEL_PATH, legacy_path=LEGACY_MODEL_PATH, rerank_path=RERANK_PATH):
    """
    Tries multiple model loading strategies:
    - joblib file containing {"model":..., "features":...}
    - lightgbm Booster loaded from text file
    Returns: (model_obj, feature_list, rerank_df)
    model_obj: object with predict_proba(X) or Booster with predict(X)
    feature_list: list of feature column names (may be guessed)
    rerank_df: DataFrame of user-item interactions / candidate features
    """
    model = None
    features = None

    # 1) try joblib (recommended)
    if model_path.exists():
        try:
            data = joblib.load(model_path)
            if isinstance(data, dict) and "model" in data and "features" in data:
                model = data["model"]
                features = data["features"]
            else:
                # maybe the joblib is just the model
                model = data
        except Exception as e:
            st.warning(f"Failed to load joblib model: {e}")

    # 2) fallback: try lightgbm text booster
    if model is None and legacy_path.exists():
        try:
            model = lgb.Booster(model_file=str(legacy_path))
            # feature names might be in the booster
            try:
                features = model.feature_name()
            except Exception:
                features = None
        except Exception as e:
            st.warning(f"Failed to load LightGBM booster: {e}")

    # 3) load rerank dataframe (candidates)
    rerank_df = pd.DataFrame()
    if rerank_path.exists():
        try:
            rerank_df = pd.read_csv(rerank_path)
        except Exception as e:
            st.warning(f"Failed to load rerank CSV: {e}")

    # attempt to infer features if not provided
    if features is None and not rerank_df.empty:
        # use common numeric columns (exclude ids/labels)
        possible = [c for c in rerank_df.columns if c not in ("user_id","item_id","category","label","ts")]
        # pick numeric-like first 12
        features = possible[:12]

    return model, features, rerank_df

model, feat_cols, rerank_df_loaded = load_recommender()

st.write(f"Model loaded: {'Yes' if model is not None else 'No'}  —  Features known: {len(feat_cols) if feat_cols else 0}")

# UI: choose user
user_sample_size = 200
if not rerank_df_loaded.empty:
    uniq_users = rerank_df_loaded["user_id"].dropna().unique()
    sample_users = list(pd.Series(uniq_users).sample(min(user_sample_size, len(uniq_users)), random_state=42))
else:
    sample_users = []

chosen_user = st.selectbox("Choose user (or paste id below)", options=["--paste-id--"] + sample_users)
pasted = st.text_input("Or paste user_id here (takes precedence):", value="")
if pasted.strip():
    user = pasted.strip()
else:
    user = None if chosen_user == "--paste-id--" else chosen_user

top_k = st.slider("Top K recommendations", 1, 20, 5)

# helper: safe feature builder using merged main df (catalog-like) or rerank_df_loaded
def build_candidate_features_from_df(cand_df, required_feats):
    # cand_df: DataFrame with many columns from merged dataset (final_merged_inventory_sentiment.csv)
    dfc = cand_df.copy()
    # ensure columns exist
    for c in required_feats:
        if c not in dfc.columns:
            dfc[c] = 0.0
    # derived features present in many flows:
    if "predicted_sales_next_28_days" in dfc.columns and "initial_inventory" in dfc.columns:
        dfc.loc[:, "sales_per_inv"] = dfc["predicted_sales_next_28_days"] / (dfc["initial_inventory"] + 1e-6)
    if "price" in dfc.columns and "avg_rating" in dfc.columns:
        dfc.loc[:, "price_per_rating"] = dfc["price"] / (dfc["avg_rating"].replace(0, np.nan).fillna(1.0))
    dfc[required_feats] = dfc[required_feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return dfc

# rerank function
def recommend_for_user_from_history(user_id, top_k=5):
    """
    If rerank_df_loaded contains user history and feature columns, use it.
    Otherwise fallback to catalog-wide ranking using df (main merged features).
    """
    # If no model -> empty result (UI will handle fallback)
    if model is None:
        return pd.DataFrame()

    # Case 1: user has history in rerank_df_loaded
    if (not rerank_df_loaded.empty) and ("user_id" in rerank_df_loaded.columns):
        user_candidates = rerank_df_loaded[rerank_df_loaded["user_id"] == user_id].copy()
        if user_candidates.shape[0] > 0:
            # Make sure features exist
            use_feats = [f for f in feat_cols if f in user_candidates.columns]
            if len(use_feats) == 0:
                # attempt to pick numeric columns
                use_feats = [c for c in user_candidates.columns if c not in ("user_id","item_id","category","label","ts")]
            # safe numeric fill
            user_candidates[use_feats] = user_candidates[use_feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            # two loader types:
            try:
                # joblib/sklearn model with predict_proba
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(user_candidates[use_feats])[:,1]
                else:
                    # LightGBM Booster expects numpy 2D (matrix) order of features as feature_name
                    if isinstance(model, lgb.Booster):
                        Xmat = user_candidates[use_feats].astype(float).values
                        probs = model.predict(Xmat)
                    else:
                        # fallback to .predict
                        probs = model.predict(user_candidates[use_feats])
                user_candidates.loc[:, "pred_score"] = probs
                ranked = user_candidates.sort_values("pred_score", ascending=False)
                display_cols = [c for c in ["item_id","pred_score","avg_rating","avg_pos_prob","inventory_score_inv","initial_inventory"] if c in ranked.columns]
                return ranked[display_cols].head(top_k).reset_index(drop=True)
            except Exception as e:
                st.warning(f"Model predict failed: {e}")
                return pd.DataFrame()
    # Case 2: fallback: rank across catalog-like df (main merged df)
    # Use df (final_merged_inventory_sentiment.csv) as candidate pool
    if not df.empty and feat_cols:
        # build candidate features from df, only keep column names that match feat_cols
        cand = df.copy()
        cand_feat = build_candidate_features_from_df(cand, feat_cols)
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(cand_feat[feat_cols])[:,1]
            elif isinstance(model, lgb.Booster):
                probs = model.predict(cand_feat[feat_cols].astype(float).values)
            else:
                probs = model.predict(cand_feat[feat_cols])
            cand.loc[:, "pred_score"] = probs
            ranked = cand.sort_values("pred_score", ascending=False)
            display_cols = [c for c in ["item_id","title","predicted_sales_next_28_days","avg_pos_prob","inventory_score","pred_score"] if c in ranked.columns]
            return ranked[display_cols].head(top_k).reset_index(drop=True)
        except Exception as e:
            st.warning(f"Catalog rerank failed: {e}")
            return pd.DataFrame()

    return pd.DataFrame()

# Show recommendations
if user:
    recs = recommend_for_user_from_history(user, top_k=top_k)
    if recs.empty:
        st.info("No personalized results available (missing model or no user history). Showing global top by sentiment.")
        global_top = (df.groupby(["item_id","title"], as_index=False)
                        .agg({"avg_pos_prob":"mean","avg_rating":"mean","predicted_sales_next_28_days":"mean"})
                        .sort_values("avg_pos_prob", ascending=False)
                        .head(top_k))
        st.dataframe(global_top.reset_index(drop=True), use_container_width=True)
    else:
        st.write(f"Top {top_k} recommendations for user {user}:")
        st.dataframe(recs, use_container_width=True)
else:
    st.info("Pick or paste a user id to see personalized recommendations.")

st.markdown("---")
st.caption("Notes: model-loading supports joblib (.pkl with dict {'model','features'}) or LightGBM booster text file. If no model is available, dashboard falls back to global heuristics.")

# ---- end of file ----
