
# Forecasting FTSE MIB Dynamics: Econometrics vs. Deep Learning


An empirical study evaluating short-term forecasting models for the Italian equity index (**FTSE MIB**), incorporating two exogenous global drivers: the **S&P 500** index and **Crude Oil (WTI)** prices.

This repository compares classical econometric time series frameworks (**VAR**, **VECM**) with deep learning architectures (**Vanilla RNN**, **LSTM**), analyzing the impacts of stationarity, cointegration, and memory dynamics on out-of-sample forecast accuracy.

---

## 📌 Economic Motivation

- **S&P 500 (Global Market Leadership & Spillover):** Serves as a leading proxy for international financial market sentiment, macroeconomic cycles, and global risk appetite driving European trading sessions.
- **Crude Oil / WTI (Sectoral Composition & Energy Sensitivity):** Acts as a direct fundamental proxy for cash flow expectations within the FTSE MIB, which is heavily weighted toward energy, industrial, and utility giants (e.g., Eni, Saipem, Tenaris).

---

## 🛠️ Methodology & Architecture

1. **Preprocessing & Unit Root Testing:**
   - Automated data retrieval via `yfinance` covering daily market data from 2015 to 2026.
   - Rigorous unit root and stationarity evaluation across log-prices and log-returns using **Augmented Dickey-Fuller (ADF)**, **Phillips-Perron (PP)**, and **KPSS** tests.
   - Empirical confirmation of $I(1)$ integration in log-prices and pure $I(0)$ stationarity in log-returns.

2. **Econometric Frameworks (VECM & VAR):**
   - Optimal lag order selection ($p$) based on information criteria (AIC/BIC).
   - **Johansen Cointegration Test** (Trace and Maximum Eigenvalue statistics) across bivariate and trivariate systems to evaluate long-run equilibrium relationships.
   - Estimation of a Vector Autoregression (**VAR**) model on standardized log-returns.

3. **Deep Learning Frameworks (PyTorch):**
   - Sliding window dataset construction (20-day temporal windows).
   - PyTorch implementations of **Vanilla RNN** and **LSTM** architectures to capture potential non-linear temporal interactions.
   - Strictly chronological data splitting (**70% Train / 15% Validation / 15% Test**) with **Early Stopping** based on validation loss to prevent overfitting.

4. **Model Evaluation & Hypothesis Testing:**
   - Out-of-sample performance measured via **Mean Squared Error (MSE)** on standardized log-returns against a **Naive Random Walk** baseline.
   - **Diebold-Mariano Test** to evaluate whether differences in predictive accuracy between linear econometrics and neural networks are statistically significant.

---

## 📊 Empirical Results

Testing on out-of-sample standardized log-returns confirms that modeling stationary return series cuts forecast error by approximately 50% relative to the Naive Random Walk benchmark. 

| Model | Input Data | Out-of-Sample Test MSE |
| :--- | :--- | :--- |
| **Naive Benchmark (Random Walk)** | Log-Returns | `0.994082` |
| **Vanilla RNN** | Log-Returns | `0.532573` |
| **LSTM Network** | Log-Returns | `0.529362` |
| **VAR(p) Model** | Log-Returns | **`0.490353`** |

### Key Takeaways:
- **Absence of Cointegration:** The Johansen test fails to reject the null hypothesis of no cointegration ($r=0$). Without a long-run equilibrium vector binding asset prices together, long-term price memory provides no structural forecasting power.
- **Gating Mechanism Convergence:** Because daily equity returns exhibit weak autocorrelation beyond immediate short lags ($1–3$ days), the long-term cell state and gating mechanisms of the LSTM offer no significant gain over the Vanilla RNN.
- **Parsimony Prevails (VAR Winner):** The linear **VAR** model achieves the lowest out-of-sample MSE. In high-frequency, noise-dominated financial returns, parsimonious linear models minimize overfitting risk compared to parameter-dense neural networks.

---

## 💻 Tech Stack

- **Language:** Python 3.x
- **Data Manipulation & Visualization:** `pandas`, `numpy`, `matplotlib`
- **Econometrics & Statistics:** `statsmodels`, `arch`, `scipy`
- **Deep Learning:** `torch` (PyTorch)
- **Financial Data:** `yfinance`

## 🔥 Push this bottom to open Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benedettalava/stock-market-forecasting-rnn-vecm/blob/main/price.ipynb)

