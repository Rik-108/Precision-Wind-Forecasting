# Precision Wind: AI-Driven Energy Forecasting ⚡

## 1. Executive Summary
This project deployed a **Machine Learning Forecasting System** that predicts wind turbine power output with **99.2% accuracy**. By shifting from static manufacturer theoretical charts to a dynamic "Physics-Aware" AI model, the forecasting error rate was reduced to **3.5%**, significantly minimizing financial risk in the Day-Ahead energy market.

## 2. The Business Problem: "The Cost of Uncertainty"
Wind farm operators must bid their energy volume 24 hours in advance. 
- **Under-Forecasting:** Leads to unsold energy dumped at rock-bottom prices (Lost Revenue).
- **Over-Forecasting:** Forces the operator to buy missing power from the spot market at premium rates (Direct Penalties).

**The Challenge:** Manufacturer "Theoretical Power Curves" fail to account for real-world turbulence and mechanical inertia, leading to error rates as high as **48%** during low-wind periods.

## 3. The Solution: "Physics-Aware" Machine Learning
I developed a **Random Forest Regressor** that combines meteorological data with turbine operational physics. 

**Key Drivers:**
* **Inertia Modeling:** Analyzes turbine output from the previous 10 minutes to understand momentum.
* **Physics Integration:** Engineered features representing the **Cube of Wind Speed ($v^3$)** to align the AI with aerodynamic laws.
* **Probabilistic Risk:** Developed "Confidence Zones" using Quantile Regression to provide traders with a 95% safety net for bidding.

## 4. Key Performance Indicators (KPIs)
| Metric | Value | Business Meaning |
| :--- | :--- | :--- |
| **Accuracy ($R^2$)** | **99.25%** | Explains nearly all variance in power generation. |
| **Real Error Rate (WMAPE)** | **3.54%** | Within the industry "safe zone" (<5%). |
| **Reliability (Coverage)** | **88%** | Actual output stays within predicted "Safety Bounds" 88% of the time. |

## 5. Dashboard: Asset Operations Center
The project includes a **Streamlit Dashboard** for executive stakeholders.
* **Live Metrics:** Real-time monitoring of MAE, WMAPE, and Revenue Risk.
* **The "Blue Zone":** Visualizes quantile thresholds for anomaly detection (identifying blade icing or gearbox degradation).

## 6. How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the Dashboard: `streamlit run app/dashboard.py`
