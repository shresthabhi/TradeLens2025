import streamlit as st
from .input_ui import show_backtest_input
# from .report_ui import show_backtest_report
from .report_ui2 import show_backtest_report

def backtest_tab():
    tab_input, tab_report = st.tabs(["🔧 Strategy Setup", "📊 Backtest Report"])

    with tab_input:
        show_backtest_input()

    with tab_report:
        show_backtest_report()