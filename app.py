"""
CO2 Emission by Vehicles — Streamlit Web App
AIC354 Machine Learning Fundamentals | Assignment 3
Deploy: https://streamlit.io/cloud → connect this GitHub repo → set app.py as entry point
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CO2 Emission Predictor | AIC354",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0a0d14 0%, #0d1220 50%, #0a1628 100%);
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Custom metric cards */
.metric-card {
    background: rgba(17,24,39,0.8);
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: #38bdf8; }
.metric-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-label { font-size: 0.75rem; color: #64748b; margin-top: 4px; letter-spacing: 0.5px; text-transform: uppercase; }

/* Result box */
.result-box {
    background: rgba(17,24,39,0.9);
    border: 2px solid;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.result-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 4rem; font-weight: 800; line-height: 1;
}
.result-unit { font-size: 0.9rem; color: #94a3b8; margin-top: 4px; }

/* Section headers */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem; font-weight: 700;
    color: #f0f6ff; margin-bottom: 0.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2d45;
}
/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1220;
    border-right: 1px solid #1e2d45;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
VEHICLE_CLASSES = [
    'COMPACT', 'FULL-SIZE', 'MID-SIZE', 'MINICOMPACT',
    'MINIVAN', 'PICKUP TRUCK - SMALL', 'PICKUP TRUCK - STANDARD',
    'SPECIAL PURPOSE VEHICLE', 'STATION WAGON - MID-SIZE',
    'STATION WAGON - SMALL', 'SUV - SMALL', 'SUV - STANDARD',
    'SUBCOMPACT', 'TWO-SEATER', 'VAN - CARGO', 'VAN - PASSENGER',
]
TRANSMISSIONS = [
    'A4','A5','A6','A7','A8','A9','A10',
    'AM5','AM6','AM7','AM8','AM9',
    'AS4','AS5','AS6','AS7','AS8','AS9','AS10',
    'AV','AV1','AV6','AV7','AV8','AV10',
    'M5','M6','M7',
]
FUEL_TYPES = {
    'X': 'Regular Gasoline',
    'Z': 'Premium Gasoline',
    'D': 'Diesel',
    'E': 'Ethanol (E85)',
    'N': 'Natural Gas',
}

# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

model_bundle = load_model()

# ── Load Dataset for EDA ───────────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    data_path = os.path.join(os.path.dirname(__file__), 'dataset', 'CO2 Emissions_Canada.csv')
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

df = load_dataset()

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem;">
  <div style="font-size:3rem; margin-bottom:0.3rem;">🚗💨</div>
  <h1 style="font-family:'Space Grotesk',sans-serif; font-size:2.5rem; font-weight:800;
             background:linear-gradient(90deg,#38bdf8,#818cf8); -webkit-background-clip:text;
             -webkit-text-fill-color:transparent; margin:0; letter-spacing:-1px;">
    CO₂ Emission Predictor
  </h1>
  <p style="color:#64748b; font-size:1rem; margin-top:0.5rem;">
    AIC354 — Machine Learning Fundamentals | Assignment 3 &nbsp;·&nbsp; Powered by XGBoost + 4 ML Algorithms
  </p>
</div>
""", unsafe_allow_html=True)

# ── Hero Stats ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, val, label in [
    (c1, "7,385",  "Training Records"),
    (c2, "5",      "ML Algorithms"),
    (c3, "10-Fold","Cross-Validation"),
    (c4, "RMSE",   "Evaluation Metric"),
]:
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-val">{val}</div>
      <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔮 Predict CO₂", "📊 Dataset Explorer", "🤖 ML Comparison"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if model_bundle is None:
        st.warning("""
        ⚠️ **Model not found.**
        Run `notebooks/notebook2_full_dataset_10fold.ipynb` on Google Colab first,
        then download `best_model.pkl` and place it in the `models/` folder.
        """)
        st.info("💡 You can still explore the **Dataset Explorer** and **ML Comparison** tabs.")
    else:
        st.success(f"✅ Model loaded: **{model_bundle.get('model_name', 'Unknown')}**")

    st.markdown("<div class='section-header'>⚙️ Vehicle Specifications</div>", unsafe_allow_html=True)

    # ── Input Form ────────────────────────────────────────────────────────────
    with st.form("prediction_form"):
        row1 = st.columns(3)
        row2 = st.columns(3)
        row3 = st.columns(3)

        with row1[0]:
            engine_size = st.number_input("🔧 Engine Size (L)", min_value=0.5, max_value=8.0,
                                          value=2.0, step=0.1)
        with row1[1]:
            cylinders = st.selectbox("🔩 Cylinders",
                                     options=[3, 4, 5, 6, 8, 10, 12], index=1)
        with row1[2]:
            vehicle_class = st.selectbox("🚙 Vehicle Class", VEHICLE_CLASSES, index=0)

        with row2[0]:
            transmission = st.selectbox("⚡ Transmission", TRANSMISSIONS,
                                        index=TRANSMISSIONS.index('AS6'))
        with row2[1]:
            fuel_type_label = st.selectbox("⛽ Fuel Type", list(FUEL_TYPES.values()))
            fuel_type = [k for k, v in FUEL_TYPES.items() if v == fuel_type_label][0]
        with row2[2]:
            fuel_city = st.number_input("🏙️ City Fuel (L/100km)", min_value=1.0,
                                        max_value=30.0, value=10.0, step=0.1)

        with row3[0]:
            fuel_hwy = st.number_input("🛣️ Highway Fuel (L/100km)", min_value=1.0,
                                       max_value=25.0, value=7.5, step=0.1)
        with row3[1]:
            fuel_comb = st.number_input("🔄 Combined Fuel (L/100km)", min_value=1.0,
                                        max_value=28.0, value=9.0, step=0.1)
        with row3[2]:
            fuel_comb_mpg = st.number_input("📏 Combined Fuel (mpg)", min_value=5,
                                            max_value=100, value=31, step=1)

        submitted = st.form_submit_button("🔮 Predict CO₂ Emission",
                                          use_container_width=True, type="primary")

    # ── Prediction Logic ───────────────────────────────────────────────────────
    if submitted:
        if model_bundle is None:
            st.error("❌ Cannot predict — model file missing. See instructions above.")
        else:
            le_dict = model_bundle['le_dict']
            model   = model_bundle['model']

            def safe_encode(le, value):
                classes = list(le.classes_)
                return le.transform([value])[0] if value in classes else 0

            vc_enc = safe_encode(le_dict['Vehicle Class'], vehicle_class)
            tr_enc = safe_encode(le_dict['Transmission'],  transmission)
            ft_enc = safe_encode(le_dict['Fuel Type'],     fuel_type)

            X_input = np.array([[vc_enc, engine_size, cylinders, tr_enc, ft_enc,
                                  fuel_city, fuel_hwy, fuel_comb, fuel_comb_mpg]])
            prediction = max(0, float(model.predict(X_input)[0]))

            # Emission category
            if prediction < 120:
                cat, color, icon, tip = "Very Low", "#2ecc71", "🌿", \
                    "Excellent! This vehicle has minimal environmental impact."
            elif prediction < 170:
                cat, color, icon, tip = "Low", "#f1c40f", "✅", \
                    "Good choice. Below-average emissions for a typical vehicle."
            elif prediction < 230:
                cat, color, icon, tip = "Moderate", "#e67e22", "⚠️", \
                    "Average range. Consider fuel-efficient alternatives for daily commutes."
            elif prediction < 280:
                cat, color, icon, tip = "High", "#e74c3c", "🔴", \
                    "High emissions. Consider a smaller engine or hybrid option."
            else:
                cat, color, icon, tip = "Very High", "#c0392b", "⛔", \
                    "Very high emissions. An EV or hybrid would significantly reduce your footprint."

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>📊 Prediction Result</div>",
                        unsafe_allow_html=True)

            res_col, gauge_col = st.columns([1, 1])

            with res_col:
                st.markdown(f"""
                <div class="result-box" style="border-color:{color}; box-shadow: 0 0 40px {color}33;">
                  <div style="font-size:2.5rem; margin-bottom:0.5rem;">{icon}</div>
                  <div class="result-val" style="color:{color};">{prediction:.1f}</div>
                  <div class="result-unit">g/km CO₂ Emissions</div>
                  <div style="margin-top:1rem; font-size:1.1rem; font-weight:700;
                              color:{color}; letter-spacing:1px;">
                    {cat.upper()} EMISSIONS
                  </div>
                </div>
                """, unsafe_allow_html=True)

                st.info(f"**Model:** {model_bundle.get('model_name', 'Unknown')}")
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.04); border-left:4px solid {color};
                            padding:0.8rem 1rem; border-radius:8px; font-size:0.88rem; color:#cbd5e1;">
                  💡 {tip}
                </div>""", unsafe_allow_html=True)

            with gauge_col:
                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prediction,
                    title={"text": "CO₂ Emission (g/km)", "font": {"color": "#94a3b8", "size": 14}},
                    number={"suffix": " g/km", "font": {"color": color, "size": 28}},
                    delta={"reference": 206, "relative": False,
                           "decreasing": {"color": "#2ecc71"},
                           "increasing": {"color": "#e74c3c"}},
                    gauge={
                        "axis": {"range": [0, 400], "tickcolor": "#4b5d78",
                                 "tickfont": {"color": "#4b5d78"}},
                        "bar": {"color": color, "thickness": 0.25},
                        "bgcolor": "rgba(0,0,0,0)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0,   120], "color": "#2ecc7122"},
                            {"range": [120, 170], "color": "#f1c40f22"},
                            {"range": [170, 230], "color": "#e67e2222"},
                            {"range": [230, 280], "color": "#e74c3c22"},
                            {"range": [280, 400], "color": "#c0392b22"},
                        ],
                        "threshold": {
                            "line": {"color": color, "width": 3},
                            "thickness": 0.75,
                            "value": prediction,
                        },
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#f0f6ff"},
                    height=280,
                    margin=dict(l=20, r=20, t=40, b=0),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Emission scale reference
                scale_df = pd.DataFrame({
                    "Category": ["Very Low", "Low", "Moderate", "High", "Very High"],
                    "Max g/km": [120, 170, 230, 280, 400],
                    "Color": ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#c0392b"],
                })
                st.markdown("**Emission Scale Reference:**")
                for _, row_s in scale_df.iterrows():
                    marker = "◀ **YOU**" if row_s["Category"] == cat else ""
                    st.markdown(
                        f"<div style='display:flex;align-items:center;gap:8px;margin:3px 0;font-size:0.82rem;'>"
                        f"<span style='width:12px;height:12px;border-radius:50%;background:{row_s['Color']};display:inline-block;'></span>"
                        f"<span style='color:#94a3b8;'>{row_s['Category']} (&lt;{row_s['Max g/km']} g/km)</span>"
                        f"<span style='color:{row_s['Color']};font-weight:700;'>{marker}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if df is None:
        st.error("Dataset not found. Make sure `dataset/CO2 Emissions_Canada.csv` is in the repo.")
    else:
        st.markdown("<div class='section-header'>📂 Dataset Overview</div>",
                    unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Records",  f"{len(df):,}")
        m2.metric("Features",        str(df.shape[1] - 1))
        m3.metric("Avg CO₂ (g/km)", f"{df['CO2 Emissions(g/km)'].mean():.0f}")
        m4.metric("CO₂ Range",      f"{df['CO2 Emissions(g/km)'].min()}–{df['CO2 Emissions(g/km)'].max()}")

        st.dataframe(df.head(20), use_container_width=True, height=300)

        st.markdown("<div class='section-header'>📈 Visualizations</div>",
                    unsafe_allow_html=True)

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig1 = px.histogram(
                df, x='CO2 Emissions(g/km)', nbins=50,
                title="Distribution of CO₂ Emissions",
                color_discrete_sequence=['#38bdf8'],
                template='plotly_dark',
            )
            fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)',
                               font_color='#94a3b8',
                               title_font_color='#f0f6ff')
            st.plotly_chart(fig1, use_container_width=True)

        with chart_col2:
            fig2 = px.scatter(
                df, x='Engine Size(L)', y='CO2 Emissions(g/km)',
                color='Cylinders', size_max=6, opacity=0.5,
                title="Engine Size vs CO₂ Emissions",
                color_continuous_scale='Plasma',
                template='plotly_dark',
            )
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)',
                               font_color='#94a3b8',
                               title_font_color='#f0f6ff')
            st.plotly_chart(fig2, use_container_width=True)

        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            fuel_means = (df.groupby('Fuel Type')['CO2 Emissions(g/km)']
                          .mean().reset_index().sort_values('CO2 Emissions(g/km)', ascending=False))
            fuel_means['Fuel Name'] = fuel_means['Fuel Type'].map(FUEL_TYPES).fillna(fuel_means['Fuel Type'])
            fig3 = px.bar(
                fuel_means, x='Fuel Name', y='CO2 Emissions(g/km)',
                title="Mean CO₂ by Fuel Type",
                color='CO2 Emissions(g/km)', color_continuous_scale='RdYlGn_r',
                template='plotly_dark',
            )
            fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)',
                               font_color='#94a3b8',
                               title_font_color='#f0f6ff',
                               showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

        with chart_col4:
            cyl_means = (df.groupby('Cylinders')['CO2 Emissions(g/km)']
                         .mean().reset_index().sort_values('Cylinders'))
            fig4 = px.line(
                cyl_means, x='Cylinders', y='CO2 Emissions(g/km)',
                markers=True, title="Mean CO₂ by Number of Cylinders",
                color_discrete_sequence=['#818cf8'],
                template='plotly_dark',
            )
            fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)',
                               font_color='#94a3b8',
                               title_font_color='#f0f6ff')
            st.plotly_chart(fig4, use_container_width=True)

        # Correlation heatmap
        st.markdown("<div class='section-header'>🔥 Correlation Heatmap</div>",
                    unsafe_allow_html=True)
        num_df = df.select_dtypes(include=[np.number])
        corr   = num_df.corr()
        fig_heat = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale='RdBu_r',
            title="Feature Correlation Matrix",
            template='plotly_dark',
        )
        fig_heat.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)',
                               font_color='#94a3b8',
                               title_font_color='#f0f6ff',
                               height=450)
        st.plotly_chart(fig_heat, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>🤖 Machine Learning Algorithms Compared</div>",
                unsafe_allow_html=True)
    st.info("ℹ️ **Note:** All 5 algorithms below are intentionally different from those used in the Car Price Prediction System (Linear Regression, KNN, Decision Tree, Random Forest, Gradient Boosting).")

    # Algorithm info table
    algo_df = pd.DataFrame({
        "Rank":       ["🥇 1", "🥈 2", "🥉 3", "4", "5"],
        "Algorithm":  ["XGBoost Regressor", "Extra Trees Regressor",
                       "SVR (RBF Kernel)", "Ridge Regression", "Lasso Regression"],
        "Category":   ["Boosting", "Ensemble (Bagging)", "Kernel-Based",
                       "Linear (L2 Reg.)", "Linear (L1 Reg.)"],
        "Key Config":  [
            "n_est=200, lr=0.05, depth=6",
            "n_est=200, max_features=sqrt",
            "C=100, epsilon=0.1, gamma=scale",
            "alpha=1.0",
            "alpha=0.01, max_iter=10000",
        ],
        "Advantage":  [
            "Best accuracy, handles non-linearity",
            "Fast, robust, low overfitting",
            "Excellent on scaled data",
            "Stable baseline with regularization",
            "Feature selection via sparsity",
        ],
    })
    st.dataframe(algo_df, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-header'>📊 Evaluation Methodology</div>",
                unsafe_allow_html=True)

    ev_col1, ev_col2 = st.columns(2)
    with ev_col1:
        st.markdown("""
        **10-Fold Cross-Validation Process:**
        1. Dataset split into **10 equal folds**
        2. Each fold used once as **validation set**
        3. Remaining 9 folds used for **training**
        4. RMSE computed for each fold
        5. **Mean ± Std** reported as final metric
        """)
    with ev_col2:
        st.markdown("""
        **RMSE (Root Mean Squared Error):**
        ```
        RMSE = √( Σ(ŷᵢ - yᵢ)² / n )
        ```
        - **Lower RMSE = Better model**
        - Units: g/km (same as target)
        - Sensitive to large errors
        - Used for comparison across all 5 algorithms
        """)

    # Simulated RMSE chart (placeholder — actual values from notebook)
    st.markdown("<div class='section-header'>📈 Typical RMSE Comparison (Indicative)</div>",
                unsafe_allow_html=True)
    st.caption("ℹ️ Run Notebook 2 on Google Colab to get exact RMSE values for your dataset run.")

    sim_df = pd.DataFrame({
        "Model": ["XGBoost Regressor", "Extra Trees Regressor",
                  "SVR (RBF Kernel)", "Ridge Regression", "Lasso Regression"],
        "Typical RMSE (g/km)": [5.2, 6.1, 9.8, 18.4, 18.6],
        "Color": ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c"],
    })

    fig_rmse = go.Figure()
    fig_rmse.add_trace(go.Bar(
        y=sim_df["Model"],
        x=sim_df["Typical RMSE (g/km)"],
        orientation='h',
        marker_color=sim_df["Color"],
        text=[f"{v:.1f}" for v in sim_df["Typical RMSE (g/km)"]],
        textposition='outside',
    ))
    fig_rmse.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#94a3b8',
        xaxis_title="RMSE (g/km) — Lower is Better",
        yaxis=dict(autorange='reversed'),
        height=300,
        margin=dict(l=10, r=80, t=10, b=40),
    )
    st.plotly_chart(fig_rmse, use_container_width=True)

    st.markdown("<div class='section-header'>🚀 Deployment Info</div>",
                unsafe_allow_html=True)
    st.markdown("""
    | Item | Detail |
    |------|--------|
    | **Platform** | Streamlit Community Cloud (free) |
    | **Repo** | [github.com/Ummad-mk/CO2-emission-by-vehicles-prediction-ai](https://github.com/Ummad-mk/CO2-emission-by-vehicles-prediction-ai) |
    | **Entry Point** | `app.py` |
    | **Course** | AIC354 — Machine Learning Fundamentals |
    | **Assignment** | 3 — Comparing Multiple ML Algorithms |
    """)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#4b5d78; font-size:0.78rem; padding: 0.5rem 0 1rem;">
  AIC354 — Machine Learning Fundamentals | Assignment 3 &nbsp;·&nbsp;
  CO2 Emission Prediction System &nbsp;·&nbsp;
  Dataset: Government of Canada (7,385 vehicle records) &nbsp;·&nbsp;
  10-Fold Cross-Validation
</div>
""", unsafe_allow_html=True)
