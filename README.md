# HELIOSGRID: Solar Forecasting & Grid Telemetry Platform



Industrial-grade machine learning application designed to predict 24-hour ahead solar power generation, integrate real-time weather telemetry, and detect equipment operational anomalies under clear-sky conditions.



![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)

![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)

![License](https://img.shields.io/badge/License-MIT-green)



---



##  Executive Summary



HeliosGrid is an enterprise-ready solar generation forecasting and microgrid management platform. Built using **XGBoost time-series regression**, it combines cyclic temporal encodings with dynamic autoregressive lag parameters to deliver reliable 24-hour generation estimates. 



The platform continuously syncs with the **Open-Meteo REST API** for live cloud cover metrics, features a **Contextual Anomaly Engine** to isolate genuine inverter faults from weather disturbances, and provides **TreeSHAP model explainability** for operational transparency.



---



## Key Features



- **XGBoost Time-Series Engine:** Predicts solar power output for a 24-hour rolling horizon using cyclic sine/cosine hour encodings, month/day parameters, and autoregressive lags (`lag_1h`, `lag_24h`).

- **Live Weather Telemetry Sync:** Integrates real-time cloud cover metrics via Open-Meteo REST API to scale expected solar irradiance dynamically across global site locations.

- **Contextual Anomaly Detection Engine:** Eliminates false positives by evaluating real-time operational drops against daylight windows, cloud cover index, and model baselines to flag hardware/inverter failures.

- **Explainable AI (TreeSHAP):** Calculates global and local feature importance via SHAP values, quantifying exactly how time factors, cloud cover, and historical lags influence predictions.

- **Command Center Telemetry UI:** Designed with a high-contrast dark telemetry theme featuring custom CSS, glassmorphism KPI cards, and interactive Plotly telemetry charts.



---



## System Architecture





[ Global Coordinates ] ---> [ Open-Meteo Live API ] ---> [ Dynamic Irradiance Factor ]

                                                                   |

                                                                   v

[ Feature Matrix (Lags, Sin/Cos) ] ---> [ XGBoost Regression Model ] ---> [ Raw Generation Forecast ]

                                                                                   |

                                                                                   v

[ Daylight & Weather Rules ] <---------------------------------------- [ Anomaly Detection Engine ]

             |                                                                     |

             v                                                                     v

[ Interactive Plotly Telemetry ] <--- [ Streamlit Command Dashboard ] <--- [ Critical Fault Alerts ]



**** Repository Structure ****



heliosgrid-solar-forecasting/

├── models/

│   └── solar_xgboost_artifacts.pkl   # Serialized model, features, and lag metadata

├── app.py                            # Streamlit dark-theme dashboard application

├── process.ipynb                     # Notebook for data cleaning, feature engineering & model training

├── .gitignore                        # Git exclusion rules

├── requirements.txt                  # Python dependency list

└── README.md                         # Project documentation



 **** Tech Stack ****

Core Language: Python 3.10+



Machine Learning: XGBoost, Scikit-learn, SHAP (TreeExplainer)



Data Engineering: Pandas, NumPy



Interactive Visualizations: Plotly



Web Application Framework: Streamlit (Custom Dark Theme CSS)



Telemetry Data Integration: Open-Meteo REST API







**** Getting Started ****

1. Clone the Repository

git clone [https://github.com/prinse0204/heliosgrid-solar-forecasting.git](https://github.com/prinse0204/heliosgrid-solar-forecasting.git)

cd heliosgrid-solar-forecasting



2. Set Up Virtual Environment

# Windows

python -m venv venv

venv\Scripts\activate



# macOS/Linux

python3 -m venv venv

source venv/bin/activate



3. Install Dependencies

Create a requirements.txt file (if not present) and install required libraries:

pip install -r requirements.txt



4. Launch Application

streamlit run app.py





**** Anomaly Engine Logic (False Positive Mitigation) ****

To prevent false alarms, the anomaly engine evaluates power drops using contextual validation across three parameters:

Filter,Logic / Rule,Outcome

Temporal Filter,Hour is outside 10:00 AM – 3:00 PM,Engine bypasses alert (0 output expected at night/dusk)

Weather Filter,Live Cloud Cover >70%,Reduction classified as normal weather disturbance

Hardware Drop,Generation drops >60% under saaf skies (<70% clouds),CRITICAL FAULT ALERT raised for potential inverter failure







**** License ****

Distributed under the MIT License. See LICENSE for more information.

