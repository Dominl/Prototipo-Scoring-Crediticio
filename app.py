import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge, Circle

# ── Configuración de la página ────────────────────────────────
st.set_page_config(
    page_title="Scoring Crediticio",
    page_icon="💳",
    layout="centered"
)

# ── Cargar modelo y scaler ────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('modelo_lr.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# ── Funciones del indicador ───────────────────────────────────
def risk_label(prob):
    if prob < 0.33:
        return "BAJO", "#2ecc71"
    elif prob < 0.66:
        return "MEDIO", "#f1c40f"
    else:
        return "ALTO", "#e74c3c"

def draw_gauge(prob):
    fig, ax = plt.subplots(figsize=(7, 3.5), subplot_kw={"aspect": "equal"})
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.25, 1.2)
    ax.axis("off")
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')

    sections = [
        (180, 120, "#2ecc71"),
        (120, 60,  "#f1c40f"),
        (60,  0,   "#e74c3c"),
    ]
    for start, end, color in sections:
        wedge = Wedge((0,0), 1.0, end, start, width=0.28,
                      facecolor=color, alpha=0.92)
        ax.add_patch(wedge)

    angle = 180 - (prob * 180)
    xn = 0.76 * np.cos(np.deg2rad(angle))
    yn = 0.76 * np.sin(np.deg2rad(angle))
    ax.plot([0, xn], [0, yn], color='white', linewidth=3.5, zorder=5)
    ax.add_patch(Circle((0,0), 0.045, color='white', zorder=6))

    label, color = risk_label(prob)
    ax.text(0, 0.38, f"{prob*100:.1f}%", ha='center', va='center',
            fontsize=22, fontweight='bold', color='white')
    ax.text(0, 0.18, f"Riesgo {label}", ha='center', va='center',
            fontsize=14, color=color, fontweight='bold')

    ax.text(-0.85, -0.08, "BAJO",  ha='center', fontsize=10, color='#2ecc71', fontweight='bold')
    ax.text(0.0,  -0.22, "MEDIO", ha='center', fontsize=10, color='#f1c40f', fontweight='bold')
    ax.text(0.85, -0.08, "ALTO",  ha='center', fontsize=10, color='#e74c3c', fontweight='bold')

    plt.tight_layout(pad=0)
    return fig

# ── Parámetros de preprocesamiento ────────────────────────────
p01_vals = {
    'Edad': 0.0100,
    'Mto_ingreso_mensual': 0.5493,
    'Nro_prod_financieros_deuda': 0.1644,
    'Nro_creditos_hipotecarios': -0.8098,
    'Nro_dependiente': -0.3415
}
p99_vals = {
    'Edad': 1.0578,
    'Mto_ingreso_mensual': 0.8114,
    'Nro_prod_financieros_deuda': 1.2030,
    'Nro_creditos_hipotecarios': 1.3497,
    'Nro_dependiente': 0.5692
}
vars_simetricas = list(p01_vals.keys())
vars_clip_0_1 = [
    'Prct_uso_tc', 'Prct_deuda_vs_ingresos',
    'nro_retraso3meses', 'nro_retraso2meses', 'nro_retraso1mes'
]
feature_names = [
    'Prct_uso_tc', 'Edad', 'nro_retraso3meses', 'Prct_deuda_vs_ingresos',
    'Mto_ingreso_mensual', 'Nro_prod_financieros_deuda', 'nro_retraso2meses',
    'Nro_creditos_hipotecarios', 'nro_retraso1mes', 'Nro_dependiente'
]

# ── Interfaz ──────────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center; color:#4472C4; font-size:2rem;'>
        💳 Scoring Crediticio
    </h1>
    <p style='text-align:center; color:#94a3b8; margin-bottom:2rem;'>
        Ingrese los datos del cliente para evaluar su riesgo de default
    </p>
""", unsafe_allow_html=True)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 👤 Datos Personales")
    edad = st.number_input("Edad", min_value=18, max_value=100, value=35)
    nro_dependiente = st.number_input("Nro. Dependientes", min_value=0, max_value=20, value=1)

    st.markdown("#### 💰 Datos Financieros")
    mto_ingreso = st.number_input("Ingreso Mensual (S/)", min_value=0.0, value=3000.0, step=100.0)
    prct_uso_tc = st.slider("% Uso Tarjeta de Crédito", 0.0, 1.0, 0.3, 0.01)
    prct_deuda = st.slider("Ratio Deuda vs Ingresos", 0.0, 1.0, 0.2, 0.01)

with col2:
    st.markdown("#### 🏦 Historial Crediticio")
    nro_prod = st.number_input("Nro. Productos Financieros", min_value=0, max_value=20, value=2)
    nro_cred_hip = st.number_input("Nro. Créditos Hipotecarios", min_value=0, max_value=10, value=0)

    st.markdown("#### ⚠️ Historial de Retrasos")
    nro_retraso3 = st.number_input("Retrasos >3 meses (últ. 3 años)", min_value=0, max_value=20, value=0)
    nro_retraso2 = st.number_input("Retrasos >60 días (últ. 3 años)", min_value=0, max_value=20, value=0)
    nro_retraso1 = st.number_input("Retrasos >30 días (últ. 3 años)", min_value=0, max_value=20, value=0)

st.divider()

# ── Botón de predicción ───────────────────────────────────────
if st.button("🔍 Evaluar Riesgo Crediticio", use_container_width=True, type="primary"):

    user_data = {
        'Prct_uso_tc':              prct_uso_tc,
        'Edad':                     edad,
        'nro_retraso3meses':        nro_retraso3,
        'Prct_deuda_vs_ingresos':   prct_deuda,
        'Mto_ingreso_mensual':      mto_ingreso,
        'Nro_prod_financieros_deuda': nro_prod,
        'nro_retraso2meses':        nro_retraso2,
        'Nro_creditos_hipotecarios': nro_cred_hip,
        'nro_retraso1mes':          nro_retraso1,
        'Nro_dependiente':          nro_dependiente
    }

    # Preprocesamiento
    user_df = pd.DataFrame([user_data])
    user_df_log = user_df.apply(np.log1p)
    user_scaled = scaler.transform(user_df_log)
    processed = pd.DataFrame(user_scaled, columns=feature_names)

    for col in vars_simetricas:
        processed[col] = processed[col].clip(p01_vals[col], p99_vals[col])
    for col in vars_clip_0_1:
        processed[col] = processed[col].clip(0, 1)

    prob = model.predict_proba(processed.values)[:, 1][0]
    label, color = risk_label(prob)

    # Mostrar resultado
    st.markdown(f"""
        <div style='background:#1e293b; border-radius:12px; padding:1.5rem;
                    border-left: 5px solid {color}; margin-bottom:1rem;'>
            <h3 style='color:{color}; margin:0;'>Riesgo de Default: {label}</h3>
            <p style='color:#94a3b8; margin:0.5rem 0 0 0;'>
                Probabilidad estimada: <strong style='color:white;'>{prob*100:.1f}%</strong>
            </p>
        </div>
    """, unsafe_allow_html=True)

    fig = draw_gauge(prob)
    st.pyplot(fig)

    # Interpretación
    st.markdown("#### 📋 Interpretación")
    if label == "BAJO":
        st.success("✅ El cliente presenta bajo riesgo crediticio. Se recomienda **aprobar** el crédito.")
    elif label == "MEDIO":
        st.warning("⚠️ El cliente presenta riesgo moderado. Se recomienda **evaluar con condiciones** adicionales.")
    else:
        st.error("❌ El cliente presenta alto riesgo crediticio. Se recomienda **rechazar o condicionar** el crédito.")
