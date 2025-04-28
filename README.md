# 📈 Portfolio Optimization Toolkit

A lightweight and modular Python toolkit for portfolio optimization, backtesting, and data visualization. Built with Streamlit for a simple and interactive user interface.

---

## 🚀 Features

- **Portfolio Optimization**: 
  - Maximize Sharpe Ratio
  - Minimize Portfolio Volatility
  - Risk Parity Optimization
- **Backtesting**:
  - Simulate and compare portfolio strategies against benchmarks
- **Data Handling**:
  - Load historical asset prices from local CSV files
- **Visualization**:
  - Interactive performance plots and risk-return charts
- **Streamlit App**:
  - Web-based UI for easy experimentation

---

## 📂 Project Structure
       
      Portfolio Optimization Toolkit/
      ├── app/
      │   └── streamlit_app.py
      ├── portfolio_optimizer/
      │   ├── __init__.py
      │   └── optimizer.py
      ├── src/
      │   ├── backtest.py
      │   ├── data_loader.py
      │   ├── plotter.py
      │   └── utils.py
      ├── requirements.txt
      ├── README.md
      └── LICENSE

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/portfolio-optimization-toolkit.git
   cd portfolio-optimization-toolkit
2. Install required packages:

   ```bash
   pip install -r requirements.txt
   Run the Streamlit app:
   streamlit run app/streamlit_app.py

---

## 📊 Technologies Used
Python 3.10+

Streamlit for the web app

pandas, NumPy for data manipulation

matplotlib, seaborn for visualization

scipy.optimize for portfolio optimization

---

## ✨ Future Improvements
Add more robust backtesting framework (transaction costs, slippage)

Integrate live data from APIs (Yahoo Finance, Alpha Vantage)

Add machine learning-based portfolio selection models

Extend to multi-period optimization

---

## 📜 License
This project is licensed under the MIT License.

---

## 👨‍💻 Authors
Diego Urdaneta — @Diegotistical
