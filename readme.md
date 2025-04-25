# 📊 TradeLens: Integrated Stock Analysis Platform

**Built by:**  
- **Abhinav Shresth** – [abhinav8@bu.edu](mailto:abhinav8@bu.edu)  
- **Parampal Singh Sidhu** – [parampal@bu.edu](mailto:parampal@bu.edu)  
- **Sanjal Atul Desai** – [sanjal@bu.edu](mailto:sanjal@bu.edu)  

---

TradeLens simplifies stock analysis by unifying three essential capabilities—**fundamental screening**, **sentiment analytics**, and **strategy backtesting**—into one seamless, no-code platform.

Traditional tools often require users to navigate across multiple systems, slowing down decision-making and adding complexity. TradeLens solves this by offering an integrated dashboard where users can explore, analyze, and test investment ideas—all in one place.

---

## 🚀 Key Features

### 📋 Stock Screening & Filtering
Screen and evaluate stocks based on key financial metrics such as:
- Beta
- Debt-to-Equity Ratio
- Price-to-Earnings and more

### 🔍 Analytical Tools
- Perform real-time **sentiment analysis** using FinBERT on stock-related news
- Analyze portfolio **diversification**
- Leverage **KMeans clustering** and other ML models for advanced insights

### 📈 Backtesting
- Define custom **entry and exit strategies**
- Simulate performance over historical data
- No coding required—designed for intuitive use

---

## 🧭 Getting Started

To run the app locally:

1. **Clone the repository**
   ```bash
   git clone https://github.com/sanjalD/TradeLens/tree/main
   cd <your-repo-folder>
   ```

2. **Add your OpenAI API key**  
   Create a file named `.streamlit/secrets.toml` and add:
   ```toml
   OPENAI_API_KEY = "your-openai-api-key"
   ```

3. **Download the financial ratio dataset**  
   Save the file in your root project directory:  
   [Download CSV](https://drive.google.com/uc?export=download&id=1fjltpId8rHjZjvkpcrmHTY48lYOuPQnc)

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

---

## 📌 Note

This project is intended for educational and prototype purposes. It aims to simplify investment analysis workflows for users who may not have access to institutional-grade tools.
