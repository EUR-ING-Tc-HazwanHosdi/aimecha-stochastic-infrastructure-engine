# AIMeCHA: Stochastic Cost Modeling & Real Options Analysis for Municipal Sewerage Infrastructure

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.32%2B-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Based on the research paper by EUR ING Tc. Hazwan B. Hosdi** (Planning and Engineering Department - CADCS, Indah Water Konsortium (IWK) Sdn Bhd / WorldQuant University).

---

## 📖 Overview

Traditional municipal sewerage infrastructure development relies heavily on deterministic Bills of Quantities (BQ) and static capital expenditure (CapEx) estimations. Despite comprehensive upfront calculations, urban development projects routinely suffer from systemic budget overruns—commonly termed **"budget bursts"**—during project execution and operational handover.

**AIMeCHA** bridges financial engineering and civil infrastructure management by replacing arbitrary percentage buffers (such as flat 10% contingency rules) with:
1. **Itô Stochastic Differential Equations (SDE)** modeling cost evolution driven by expected operational drift and subterranean shock volatility.
2. **Real Options Analysis (ROA)** adapted from the Black-Scholes-Merton framework to price contingency tranches and manage project flexibilities.

---

## 📐 Mathematical Formulation

### 1. Baseline CapEx ($C_0$)
Defined deterministically via approved engineering quantities ($q_i$) and fixed unit rates ($p_i$):
$$C_0 = \sum_{i=1}^{n} q_i \cdot p_i$$

### 2. Stochastic Cost Evolution (Itô Diffusion Process)
Governs instantaneous cost evolution via geometric Brownian motion:
$$dC_t = \mu C_t \, dt + \sigma C_t \, dW_t$$
*(where $\mu$ is expected operational drift, $\sigma$ is subterranean shock volatility, and $W_t$ is a Wiener process).*

### 3. Probability of a Budget Burst ($\mathbb{P}_{\text{burst}}$)
Formulated using the standard normal cumulative distribution function ($\Phi$) against a static risk ceiling ($C_{\text{ceiling}} = (1 + \alpha)C_0$):
$$\mathbb{P}_{\text{burst}} = 1 - \Phi\left( \frac{\ln(1 + \alpha) - \left( \mu - \frac{1}{2}\sigma^2 \right) T}{\sigma \sqrt{T}} \right)$$

---

## 🚀 Quickstart & Installation

To run the interactive Streamlit application locally, follow these steps:

### Prerequisites
* Python 3.10 or higher
* pip package manager

### 1. Clone the Repository & Setup Environment
```bash
git clone [https://github.com/your-username/aimecha-sewerage-cost-model.git](https://github.com/your-username/aimecha-sewerage-cost-model.git)
cd aimecha-sewerage-cost-model

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
