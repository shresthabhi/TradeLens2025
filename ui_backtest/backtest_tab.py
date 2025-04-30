import streamlit as st
from .input_ui import show_backtest_input
# from .report_ui import show_backtest_report
from .report_ui2 import show_backtest_report

def backtest_tab():
    tab_input, tab_report = st.tabs(["🔧 Strategy Setup", "📊 Backtest Report"])

    with tab_input:
        st.warning("""
⚠️ **Important Notice**  
This app makes live API calls to a financial data library. Since it's hosted on Streamlit Cloud, these calls may occasionally **fail due to rate limits or API restrictions**.

Additionally, FinBERT sentiment analysis **won't work on Streamlit Cloud** because it requires the `torch` library, which is difficult to install in this environment.

👉 To run this app with full functionality, including sentiment analysis, **please follow the setup instructions in the [README](https://github.com/sanjalD/TradeLens/tree/main) and host it locally.**
""")
        show_backtest_input()

    with tab_report:
        st.warning("""
⚠️ **Important Notice**  
This app makes live API calls to a financial data library. Since it's hosted on Streamlit Cloud, these calls may occasionally **fail due to rate limits or API restrictions**.

Additionally, FinBERT sentiment analysis **won't work on Streamlit Cloud** because it requires the `torch` library, which is difficult to install in this environment.

👉 To run this app with full functionality, including sentiment analysis, **please follow the setup instructions in the [README](https://github.com/sanjalD/TradeLens/tree/main) and host it locally.**
""")
        show_backtest_report()