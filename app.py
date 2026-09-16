import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="AIMeCHA: Stochastic Cost & ROA Engine",
    page_icon="🏗️",
    layout="wide",
)

# App Header
st.title(
    "🏗️ AIMeCHA: Municipal Sewerage Infrastructure Stochastic Risk & ROA Engine"
)
st.markdown(
    "*Interactive decision support tool bridging Itô diffusion cost modeling"
    " and Real Options Analysis (ROA) for Indah Water Konsortium (IWK) / CADCS"
    " workflows.*"
)
st.markdown("---")

# Sidebar Controls for Inputs
st.sidebar.header("🎛️ Simulation Parameters")

C0 = st.sidebar.number_input(
    "Baseline CapEx ($C_0$ in RM)",
    min_value=500000.0,
    max_value=100000000.0,
    value=5000000.0,
    step=500000.0,
)
alpha_static = st.sidebar.slider(
    "Traditional Contingency Markup ($\alpha$)",
    min_value=0.01,
    max_value=0.30,
    value=0.10,
    step=0.01,
)
T = st.sidebar.slider(
    "Project Completion Horizon ($T$ in Years)",
    min_value=0.5,
    max_value=5.0,
    value=1.5,
    step=0.25,
)
mu = st.sidebar.slider(
    "Operational Cost Drift ($\mu$)",
    min_value=0.0,
    max_value=0.15,
    value=0.05,
    step=0.01,
)
sigma = st.sidebar.slider(
    "Subterranean Volatility ($\sigma$)",
    min_value=0.05,
    max_value=0.40,
    value=0.20,
    step=0.01,
)
r = st.sidebar.slider(
    "Risk-Free Rate ($r$)",
    min_value=0.01,
    max_value=0.08,
    value=0.035,
    step=0.005,
)
delta = st.sidebar.slider(
    "Service Yield / Cash-Flow Rate ($\delta$)",
    min_value=0.0,
    max_value=0.06,
    value=0.02,
    step=0.005,
)

# --- Core Calculations ---
C_ceiling = (1 + alpha_static) * C0

# 1. Budget Burst Probability Calculation
numerator_k = np.log(1 + alpha_static) - (mu - 0.5 * (sigma**2)) * T
denominator_k = sigma * np.sqrt(T)
k = numerator_k / denominator_k
p_burst = 1 - stats.norm.cdf(k)

# 2. Real Options Analysis (ROA) Critical Threshold Calculation
discriminant = ((r - delta) / (sigma**2) - 0.5) ** 2 + (2 * r) / (sigma**2)
beta = 0.5 - (r - delta) / (sigma**2) + np.sqrt(discriminant)
if beta > 1:
  C_star = (beta / (beta - 1)) * ((r - delta) / r) * C0
  alpha_roA = max(0, (C_star - C0) / C0)
else:
  alpha_roA = alpha_static  # fallback safety

# --- Layout: Key Metrics Dashboard ---
col1, col2, col3, col4 = st.columns(4)

with col1:
  st.metric(label="Baseline CapEx ($C_0$)", value=f"RM {C0:,.2f}")
with col2:
  st.metric(
      label="Static Risk Ceiling",
      value=f"RM {C_ceiling:,.2f}",
      delta=f"{alpha_static*100:.1f}% Buffer",
  )
with col3:
  st.metric(
      label="Budget Burst Probability ($\mathbb{P}_{\text{burst}}$)",
      value=f"{p_burst*100:.1f}%",
      delta="High Risk" if p_burst > 0.3 else "Manageable Risk",
      delta_color="inverse",
  )
with col4:
  st.metric(
      label="ROA Optimized Markup ($\alpha_{\text{ROA}}$)",
      value=f"{alpha_roA*100:.1f}%",
      delta=f"RM {C0*alpha_roA:,.2f}",
  )

st.markdown("---")

# --- Monte Carlo Simulation Visualization ---
st.subheader(
    "📊 Stochastic Cost Evolution Paths vs. Static & ROA Risk Ceilings"
)

steps = int(T * 12)  # Monthly steps
dt = T / steps


@st.cache_data
def run_monte_carlo(C0, mu, sigma, T, steps, num_paths=100):
  time_grid = np.linspace(0, T, steps + 1)
  paths = np.zeros((num_paths, steps + 1))
  paths[:, 0] = C0
  for t in range(1, steps + 1):
    z = np.random.standard_normal(num_paths)
    paths[:, t] = paths[:, t - 1] * np.exp(
        (mu - 0.5 * (sigma**2)) * dt + sigma * np.sqrt(dt) * z
    )
  return time_grid, paths


time_grid, sim_paths = run_monte_carlo(C0, mu, sigma, T, steps, num_paths=100)

fig, ax = plt.subplots(figsize=(10, 5))
for i in range(sim_paths.shape[0]):
  ax.plot(time_grid, sim_paths[i], color="#1f77b4", alpha=0.15, linewidth=1)

# Highlight mean path
mean_path = np.mean(sim_paths, axis=0)
ax.plot(
    time_grid,
    mean_path,
    color="#ff7f0e",
    linewidth=2.5,
    label="Expected Cost Path (Mean)",
)

# Horizontal Ceilings
ax.axhline(
    y=C_ceiling,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Static Risk Ceiling (10%): RM {C_ceiling:,.0f}",
)
ROA_ceiling_val = C0 * (1 + alpha_roA)
ax.axhline(
    y=ROA_ceiling_val,
    color="green",
    linestyle="-.",
    linewidth=2,
    label=f"ROA Optimized Ceiling: RM {ROA_ceiling_val:,.0f}",
)

ax.set_title(
    "Monte Carlo Simulation of Sewerage Infrastructure Cost Diffusion",
    fontsize=12,
    weight="bold",
)
ax.set_xlabel("Project Horizon (Years)")
ax.set_ylabel("Realized Cumulative Cost ($C_t$ in RM)")
ax.legend(loc="upper left")
ax.grid(True, linestyle=":", alpha=0.6)

st.pyplot(fig)

# --- Explanation Panel ---
with st.expander("💡 How to Interpret This Simulation for IWK & CADCS Workflows"):
  st.markdown(f"""
    * **Budget Burst Probability ({p_burst*100:.1f}%):** This represents the exact mathematical likelihood that your subterranean works will breach the traditional static ceiling before project handover. If this number is high, flat 10% contingency buffers will fail.
    * **ROA Optimized Markup ({alpha_roA*100:.1f}%):** Rather than guessing, Real Options Analysis suggests pricing the reserve at **{alpha_roA*100:.1f}%** (or **RM {C0*alpha_roA:,.2f}**). This accounts for subterranean shock volatility ($\sigma = {sigma}$) such as deep trenching anomalies and unmapped utility clashes.
    * **Deployment:** You can run this app locally by saving it as `app.py` and running `streamlit run app.py` in your terminal, or deploy it instantly for free on Streamlit Community Cloud.
    """)
# --- NEW: User Guide & Parameter Definitions Panel ---
with st.expander(
    "📖 Parameter Definitions, Units & Engineering Rationale (Click to Expand)"
):
  st.markdown("""
    This guide explains the parameters controlled via the sidebar sliders to help municipal planners and engineers interpret the simulation:

    * **Baseline CapEx ($C_0$)**
      * **Unit:** Malaysian Ringgit (RM)
      * **Rationale:** Represents the upfront baseline capital expenditure for a mid-scale municipal sewerage infrastructure project (gravity sewers, manholes, or local STP) approved under the CADCS framework.
    
    * **Traditional Contingency Markup ($\alpha$)**
      * **Unit:** Decimal ratio (`0.10` = 10%)
      * **Rationale:** Matches the legacy industry-standard flat contingency rule traditionally added to civil engineering Bills of Quantities (BQ). Used to benchmark the risk of a budget burst.
    
    * **Project Completion Horizon ($T$)**
      * **Unit:** Years (`1.50` = 18 months)
      * **Rationale:** Reflects the typical construction and commissioning timeline from initial groundbreaking to final regulatory handover to Indah Water Konsortium (IWK).
    
    * **Operational Cost Drift ($\mu$)**
      * **Unit:** Annualized rate (`0.05` = 5%/year)
      * **Rationale:** Captures standard baseline cost creep over time driven by macroeconomic material inflation, daily labor burn rates, and project overheads.
    
    * **Subterranean Volatility ($\sigma$)**
      * **Unit:** Annualized volatility coefficient (`0.20` = 20%)
      * **Rationale:** Quantifies severe, unpredictable geological risks unique to underground utility work—such as unmapped utility clashes, high groundwater tables, and hard rock anomalies during deep trenching.
    
    * **Risk-Free Rate ($r$)**
      * **Unit:** Annualized percentage rate (`0.04` = 4%)
      * **Rationale:** Benchmarked against national municipal bond yields, serving as the discount rate foundation for the Real Options Analysis (ROA) option-pricing engine.
    
    * **Service Yield / Cash-Flow Rate ($\delta$)**
      * **Unit:** Annualized percentage rate (`0.02` = 2%)
      * **Rationale:** Represents the rate at which the completed infrastructure asset begins generating operational service utility once connected to the national municipal grid.
    """)

# --- Explanation Panel ---
with st.expander("💡 How to Interpret This Simulation for IWK & CADCS Workflows"):
  st.markdown(f"""
    * **Budget Burst Probability ({p_burst*100:.1f}%):** This represents the exact mathematical likelihood that your subterranean works will breach the traditional static ceiling before project handover. If this number is high, flat 10% contingency buffers will fail.
    * **ROA Optimized Markup ({alpha_roA*100:.1f}%):** Rather than guessing, Real Options Analysis suggests pricing the reserve at **{alpha_roA*100:.1f}%** (or **RM {C0*alpha_roA:,.2f}**). This accounts for subterranean shock volatility ($\sigma = {sigma}$) such as deep trenching anomalies and unmapped utility clashes.
    * **Deployment:** You can run this app locally by saving it as `app.py` and running `streamlit run app.py` in your terminal, or deploy it instantly for free on Streamlit Community Cloud.
    """)
