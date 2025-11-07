# D:\CAPSTONE_FINAL\app_dashboard.py
# Inventory + Sentiment Dashboard + Recommendations tab (robust version)
# DEBUG WRAPPER — put this at the very top of app_dashboard.py
try:
    # actual imports and code follow (we'll re-import below)
    pass
except Exception as e:
    import streamlit as st, traceback
    st.set_page_config(page_title="Startup error", layout="wide")
    st.title("App failed at startup — full traceback")
    st.error(traceback.format_exc())
    raise
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib
import lightgbm as lgb

# ----------------- CONFIG / PATHS -----------------
# default explicit path for your machine (fallback)
import os
from pathlib import Path

# Prefer running-directory (works for streamlit and console). If it doesn't contain a data folder, fallback to DEFAULT_BASE.
DEFAULT_BASE = Path(r"D:\CAPSTONE_FINAL")
BASE = Path(os.getcwd())
if not (BASE / "data").exists():
    BASE = DEFAULT_BASEDATA_PATH = BASE / "data" / "final_merged_inventory_sentiment.csv"
RERANK_PATH = BASE / "data" / "reranker_train.csv"
MODEL_PATH = BASE / "models" / "reranker_lgbm.pkl"   # primary
LEGACY_MODEL_PATH = BASE / "data" / "lightgbm_recommender.txt"  # fallback (older path)

# Streamlit page
st.set_page_config(page_title="Inventory & Sentiment Dashboard", layout="wide")
st.title("📊 Inventory & Sentiment Analytics Dashboard")
st.markdown("Combines sales forecasts, sentiment analysis, inventory metrics and personalized recommendations.")

# ----------------- HELPER: check files -----------------
def check_required_files():
    missing = []
    # main dataset is required
    if not DATA_PATH.exists():
        missing.append(str(DATA_PATH))
    # show helpful message and stop if missing required file(s)
    if missing:
        st.error("Missing required data files. Please ensure these are present:\n\n" + "\n".join(missing))
        st.stop()

check_required_files()

# ----------------- DATA LOAD (cached) -----------------
@st.cache_data(show_spinner=False)
def load_main_data(path):
    # load with pandas; allow CSV or parquet if present
    if path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    # ensure key numeric columns exist and are numeric
    for c in ["avg_pos_prob","avg_rating","predicted_sales_next_28_days","initial_inventory","inventory_score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0
    # create combined inventory health metric (safe)
    df["inventory_health"] = df["inventory_score"] * df["avg_pos_prob"] * (df["avg_rating"] / 5.0 + 1e-9)
    return df

@st.cache_data(show_spinner=False)
def load_rerank_data(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

# show spinner while loading heavy data
with st.spinner("Loading main dataset (may take 10-60s for large files)..."):
    df = load_main_data(DATA_PATH)
rerank_df = load_rerank_data(RERANK_PATH)

st.success(f"Loaded main dataset with {len(df):,} items.")

# ----------------- SIMPLE UI FILTERS -----------------
category = st.selectbox("📂 Select Category", ["All"] + sorted(df["category"].dropna().unique().tolist()))
df_view = df.copy()
if category != "All":
    df_view = df_view[df_view["category"] == category]

st.sidebar.header("Filter Options")
min_sent = st.sidebar.slider("Minimum Avg Sentiment", 0.0, 1.0, 0.5, 0.05)
min_score = st.sidebar.slider("Minimum Inventory Score", 0.0, 1.0, 0.0, 0.05)
df_view = df_view[(df_view["avg_pos_prob"] >= min_sent) & (df_view["inventory_score"] >= min_score)]

# ----------------- KPIs -----------------
col1, col2, col3 = st.columns(3)
col1.metric("📦 Total Items", f"{len(df_view):,}")
col2.metric("⭐ Avg Rating", f"{df_view['avg_rating'].mean():.2f}")
col3.metric("📈 Avg Predicted Sales (next 28 days)", f"{df_view['predicted_sales_next_28_days'].mean():.1f}")

# ----------------- TABS: Charts & Alerts -----------------
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
    st.subheader("🚨 AI-Driven Inventory Alerts (improved)")

    # Parameters
    lookahead_days = 28
    lead_time_days = 14
    reorder_point_days = 14
    overstock_days = 90
    high_sales_quantile = 0.75
    low_sentiment_quantile = 0.25
    min_dos = 1e-6

    # compute derived metrics
    df_calc = df.copy()
    df_calc["pred_next_per_day"] = df_calc["predicted_sales_next_28_days"] / float(lookahead_days)
    df_calc["pred_next_per_day"] = df_calc["pred_next_per_day"].replace(0, min_dos)
    df_calc["days_of_stock"] = df_calc["initial_inventory"] / (df_calc["pred_next_per_day"].clip(lower=min_dos))
    df_calc["safety_stock"] = df_calc["pred_next_per_day"] * lead_time_days * 0.5

    # thresholds from data
    high_sales_threshold = df_calc["predicted_sales_next_28_days"].quantile(high_sales_quantile)
    low_sentiment_threshold = df_calc["avg_pos_prob"].quantile(low_sentiment_quantile)

    # Restock
    restock = df_calc[
        (df_calc["days_of_stock"] <= reorder_point_days) &
        (df_calc["predicted_sales_next_28_days"] >= high_sales_threshold) &
        (df_calc["avg_pos_prob"] >= 0.4)
    ].copy()

    restock["short_units"] = (df_calc["pred_next_per_day"] * lead_time_days) - df_calc["initial_inventory"]
    restock["short_units"] = restock["short_units"].clip(lower=0.0)

    if "price" in restock.columns and "cost" in restock.columns:
        restock["margin_factor"] = (restock["price"] - restock["cost"]).clip(lower=0.0) / restock["price"].clip(lower=1.0)
    else:
        restock["margin_factor"] = 1.0

    restock["shortage_score"] = restock["short_units"] * restock["avg_pos_prob"] * (1.0 + restock["inventory_score"]) * restock["margin_factor"]
    restock = restock.sort_values("shortage_score", ascending=False)

    # Overstock
    overstock = df_calc[
        (df_calc["days_of_stock"] >= overstock_days) |
        ((df_calc["initial_inventory"] > df_calc["predicted_sales_next_28_days"]) & (df_calc["avg_pos_prob"] <= low_sentiment_threshold))
    ].copy()

    overstock["overstock_risk"] = (overstock["initial_inventory"] - (overstock["pred_next_per_day"] * lookahead_days)).clip(lower=0.0) * (1.0 - overstock["avg_pos_prob"]) * (1.0 + (1.0 - overstock["inventory_score"]))
    overstock = overstock.sort_values("overstock_risk", ascending=False)

    # Display
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🩸 Restock Needed (priority sorted)")
        st.metric("Count", len(restock))
        st.dataframe(
            restock[["item_id","title","avg_pos_prob","predicted_sales_next_28_days","initial_inventory","days_of_stock","short_units","shortage_score"]].head(50),
            use_container_width=True
        )
        if len(restock) > 0:
            fig4 = px.bar(restock.head(10), x="title", y="shortage_score", color="avg_pos_prob",
                          hover_data=["short_units","days_of_stock","predicted_sales_next_28_days"],
                          title="Top Restock Candidates (by shortage_score)")
            st.plotly_chart(fig4, use_container_width=True)

    with col2:
        st.markdown("### 🧊 Overstock Risk (priority sorted)")
        st.metric("Count", len(overstock))
        st.dataframe(
            overstock[["item_id","title","avg_pos_prob","predicted_sales_next_28_days","initial_inventory","days_of_stock","overstock_risk"]].head(50),
            use_container_width=True
        )
        if len(overstock) > 0:
            fig5 = px.bar(overstock.head(10), x="title", y="overstock_risk", color="avg_pos_prob",
                          hover_data=["days_of_stock","initial_inventory"], title="Top Overstock Risks")
            st.plotly_chart(fig5, use_container_width=True)

    st.markdown(
        """
        Notes:  
        - days_of_stock = initial_inventory / (predicted daily demand).  
        - Restock is prioritized by shortage_score = short_units × sentiment × (1+inventory_score) × margin.  
        - Overstock risk uses inventory gap × (1 - sentiment) to prioritize promotions/discounts.
        """
    )

# ----------------- Recommendations: load model + rerank data -----------------
st.header("🔎 Personalized Recommendations (Reranker)")

@st.cache_resource(ttl=600)
def load_recommender(model_path=MODEL_PATH, legacy_path=LEGACY_MODEL_PATH, rerank_path=RERANK_PATH):
    model = None
    features = None

    # try joblib
    if model_path.exists():
        try:
            data = joblib.load(model_path)
            if isinstance(data, dict) and "model" in data and "features" in data:
                model = data["model"]
                features = data["features"]
            else:
                model = data
        except Exception as e:
            st.warning(f"Failed to load joblib model: {e}")

    # fallback: lightgbm text model
    if model is None and legacy_path.exists():
        try:
            model = lgb.Booster(model_file=str(legacy_path))
            try:
                features = model.feature_name()
            except Exception:
                features = None
        except Exception as e:
            st.warning(f"Failed to load LightGBM booster: {e}")

    # load rerank df if present
    rerank_df_local = pd.DataFrame()
    if rerank_path.exists():
        try:
            rerank_df_local = pd.read_csv(rerank_path)
        except Exception as e:
            st.warning(f"Failed to load rerank CSV: {e}")

    # infer features if missing
    if features is None and not rerank_df_local.empty:
        possible = [c for c in rerank_df_local.columns if c not in ("user_id","item_id","category","label","ts")]
        features = possible[:12]

    return model, features, rerank_df_local

model, feat_cols, rerank_df_loaded = load_recommender()

if model is not None:
    loaded_from = MODEL_PATH if MODEL_PATH.exists() else (LEGACY_MODEL_PATH if LEGACY_MODEL_PATH.exists() else "unknown")
    st.success(f"Model loaded from: {loaded_from} — features known: {len(feat_cols) if feat_cols else 0}")
else:
    st.warning("No model loaded — recommendations will fall back to global heuristics.")

# ----------------- Simple recommendation UI -----------------
user_sample_size = 200
if not rerank_df_loaded.empty and "user_id" in rerank_df_loaded.columns:
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

def build_candidate_features_from_df(cand_df, required_feats):
    dfc = cand_df.copy()
    for c in required_feats:
        if c not in dfc.columns:
            dfc[c] = 0.0
    if "predicted_sales_next_28_days" in dfc.columns and "initial_inventory" in dfc.columns:
        dfc.loc[:, "sales_per_inv"] = dfc["predicted_sales_next_28_days"] / (dfc["initial_inventory"] + 1e-6)
    if "price" in dfc.columns and "avg_rating" in dfc.columns:
        dfc.loc[:, "price_per_rating"] = dfc["price"] / (dfc["avg_rating"].replace(0, np.nan).fillna(1.0))
    dfc[required_feats] = dfc[required_feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return dfc

def recommend_for_user_from_history(user_id, top_k=5):
    if model is None:
        return pd.DataFrame()

    # case: user candidates in rerank df
    if (not rerank_df_loaded.empty) and ("user_id" in rerank_df_loaded.columns):
        user_candidates = rerank_df_loaded[rerank_df_loaded["user_id"] == user_id].copy()
        if user_candidates.shape[0] > 0:
            use_feats = [f for f in feat_cols if f in user_candidates.columns] if feat_cols else [c for c in user_candidates.columns if c not in ("user_id","item_id","category","label","ts")]
            if len(use_feats) == 0:
                use_feats = [c for c in user_candidates.columns if c not in ("user_id","item_id","category","label","ts")]
            user_candidates[use_feats] = user_candidates[use_feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            try:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(user_candidates[use_feats])[:,1]
                elif isinstance(model, lgb.Booster):
                    Xmat = user_candidates[use_feats].astype(float).values
                    probs = model.predict(Xmat)
                else:
                    probs = model.predict(user_candidates[use_feats])
                user_candidates.loc[:, "pred_score"] = probs
                ranked = user_candidates.sort_values("pred_score", ascending=False)
                display_cols = [c for c in ["item_id","pred_score","avg_rating","avg_pos_prob","inventory_score","initial_inventory"] if c in ranked.columns]
                return ranked[display_cols].head(top_k).reset_index(drop=True)
            except Exception as e:
                st.warning(f"Model predict failed: {e}")
                return pd.DataFrame()

    # fallback: rank catalog
    if not df.empty and feat_cols:
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

# show recommendations
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
# ----------------- end -----------------