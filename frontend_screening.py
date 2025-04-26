import streamlit as st
import pandas as pd

def screening_tab(df):
    df.columns = df.columns.str.lower()
    df['public_date'] = pd.to_datetime(df['public_date'], errors='coerce')

    st.title("📊 Stock Screener")

    st.sidebar.subheader("Filter Stocks")
    # min_date = df['public_date'].min()
    # max_date = df['public_date'].max()
    min_date = pd.Timestamp("2018-01-01")
    max_date = pd.Timestamp("2024-12-31")
    date_range = st.sidebar.date_input("Select Public Date Range:", [min_date, max_date])

    if len(date_range) == 2:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1])
        # start_date = pd.Timestamp("2024-06-01")
        # end_date = pd.Timestamp("2024-12-31")
        df = df[(df['public_date'] >= start_date) & (df['public_date'] <= end_date)]

    readable_labels = {
        "dpr": "Dividend Payout Ratio",
        "peg_trailing": "Trailing P/E to Growth (PEG) ratio",
        "bm": "Book/Market",
        "capei": "Shillers Cyclically Adjusted P/E Ratio",
        "divyield": "Dividend Yield",
        "evm": "Enterprise Value Multiple",
        "pcf": "Price/Cash flow",
        "pe_exi": "P/E (Diluted, Excl. EI)",
        "pe_inc": "P/E (Diluted, Incl. EI)",
        "pe_op_basic": "Price/Operating Earnings (Basic, Excl. EI)",
        "pe_op_dil": "Price/Operating Earnings (Diluted, Excl. EI)",
        "ps": "Price/Sales",
        "ptb": "Price/Book",
        "efftax": "Effective Tax Rate",
        "gprof": "Gross Profit/Total Assets",
        "aftret_eq": "After-tax Return on Average Common Equity",
        "aftret_equity": "After-tax Return on Total Stockholders Equity",
        "aftret_invcapx": "After-tax Return on Invested Capital",
        "gpm": "Gross Profit Margin",
        "npm": "Net Profit Margin",
        "opmad": "Operating Profit Margin After Depreciation",
        "opmbd": "Operating Profit Margin Before Depreciation",
        "pretret_earnat": "Pre-tax Return on Total Earning Assets",
        "pretret_noa": "Pre-tax return on Net Operating Assets",
        "ptpm": "Pre-tax Profit Margin",
        "roa": "Return on Assets",
        "roce": "Return on Capital Employed",
        "roe": "Return on Equity",
        "capital_ratio": "Capitalization Ratio",
        "equity_invcap": "Common Equity/Invested Capital",
        "debt_invcap": "Long-term Debt/Invested Capital",
        "totdebt_invcap": "Total Debt/Invested Capital",
        "invt_act": "Inventory/Current Assets",
        "rect_act": "Receivables/Current Assets",
        "fcf_ocf": "Free Cash Flow/Operating Cash Flow",
        "ocf_lct": "Operating CF/Current Liabilities",
        "cash_debt": "Cash Flow/Total Debt",
        "cash_lt": "Cash Balance/Total Liabilities",
        "cfm": "Cash Flow Margin",
        "short_debt": "Short-Term Debt/Total Debt",
        "profit_lct": "Profit Before Depreciation/Current Liabilities",
        "curr_debt": "Current Liabilities/Total Liabilities",
        "debt_ebitda": "Total Debt/EBITDA",
        "dltt_be": "Long-term Debt/Book Equity",
        "int_debt": "Interest/Average Long-term Debt",
        "int_totdebt": "Interest/Average Total Debt",
        "lt_debt": "Long-term Debt/Total Liabilities",
        "lt_ppent": "Total Liabilities/Total Tangible Assets",
        "de_ratio": "Total Debt/Equity",
        "debt_assets": "Total Debt/Total Assets",
        # "debt_at": "Total Debt/Total Assets",
        "debt_capital": "Total Debt/Capital",
        "intcov": "After-tax Interest Coverage",
        "intcov_ratio": "Interest Coverage Ratio",
        "cash_conversion": "Cash Conversion Cycle (Days)",
        "cash_ratio": "Cash Ratio",
        "curr_ratio": "Current Ratio",
        "quick_ratio": "Quick Ratio (Acid Test)",
        "at_turn": "Asset Turnover",
        "inv_turn": "Inventory Turnover",
        "pay_turn": "Payables Turnover",
        "rect_turn": "Receivables Turnover",
        "sale_equity": "Sales/Stockholders Equity",
        "sale_invcap": "Sales/Invested Capital",
        "sale_nwc": "Sales/Working Capital",
        "accrual": "Accruals/Average Assets",
        "rd_sale": "Research and Development/Sales",
        "adv_sale": "Advertising Expenses/Sales",
        "staff_sale": "Labor Expenses/Sales"
    }

    filter_categories = {
        "Valuation Ratios": ["dpr", "peg_trailing", "bm", "capei", "divyield", "evm", "pcf", "pe_exi", "pe_inc", "pe_op_basic", "pe_op_dil", "ps", "ptb"],
        "Profitability Ratios": ["efftax", "gprof", "aftret_eq", "aftret_equity", "aftret_invcapx", "gpm", "npm", "opmad", "opmbd", "pretret_earnat", "pretret_noa", "ptpm", "roa", "roce", "roe"],
        "Capital Ratios": ["capital_ratio", "equity_invcap", "debt_invcap", "totdebt_invcap"],
        "Financial Soundness": ["invt_act", "rect_act", "fcf_ocf", "ocf_lct", "cash_debt", "cash_lt", "cfm", "short_debt", "profit_lct", "curr_debt", "debt_ebitda", "dltt_be", "int_debt", "int_totdebt", "lt_debt", "lt_ppent"],
        "Solvency Ratios": ["de_ratio", "debt_assets", "debt_capital", "intcov", "intcov_ratio"],
        "Liquidity Ratios": ["cash_conversion", "cash_ratio", "curr_ratio", "quick_ratio"],
        "Efficiency Ratios": ["at_turn", "inv_turn", "pay_turn", "rect_turn", "sale_equity", "sale_invcap", "sale_nwc"],
        "Other Ratios": ["accrual", "rd_sale", "adv_sale", "staff_sale"]
    }

    important_columns = ["gvkey", "ticker", "public_date", "ptb", "roa", "roe", "de_ratio", "curr_ratio", "at_turn"]
    display_labels = {col: readable_labels.get(col, col) for col in important_columns}

    st.sidebar.markdown("### ➕ Add More Filters")
    selected_filters = {}
    for group, columns in filter_categories.items():
        with st.sidebar.expander(f"📂 {group}", expanded=False):
            for col in columns:
                if col in df.columns:
                        label = readable_labels[col] if col in readable_labels else col
                        use_filter = st.checkbox(label, key=f"use_{col}")
                        if use_filter:
                            # min_val, max_val = float(df[col].min(skipna=True)), float(df[col].max(skipna=True))
                            percentile = 0.05
                            min_val, max_val = float(df[col].quantile(percentile)), float(df[col].quantile(1-percentile))
                            selected_range = st.slider(label, min_val, max_val, (min_val, max_val), key=f"slider_{col}")
                            selected_filters[col] = selected_range

    filtered_df = df.copy()
    for col, (min_val, max_val) in selected_filters.items():
        filtered_df = filtered_df[filtered_df[col].between(min_val, max_val)]

    latest_data = filtered_df.sort_values("public_date").groupby("permno").tail(1)

    additional_cols = list(selected_filters.keys())
    final_columns = important_columns + [col for col in additional_cols if col not in important_columns]
    all_labels = {**display_labels, **{col: readable_labels.get(col, col) for col in additional_cols}}
    display_df = latest_data[final_columns].rename(columns=all_labels)

    
    st.subheader("📄 Filtered Results")
    st.download_button(
        "📥 Download Filtered Data",
        latest_data[final_columns].to_csv(index=False),
        "filtered_data.csv",
        "text/csv"
    )
    styled_df = display_df.style.format(precision=2)
    st.dataframe(styled_df, use_container_width=True, height=500, hide_index=True)

    st.markdown("---")
    st.subheader("📊 Summary Metrics")

    summary_columns = [col for col in important_columns if col not in ['gvkey', 'ticker', 'public_date']] + [col for col in additional_cols if col not in ['gvkey', 'ticker', 'public_date']]
    if summary_columns:
        for i in range(0, len(summary_columns), 2):
            cols = st.columns(2)
            for idx, col in enumerate(summary_columns[i:i+2]):
                if col in latest_data:
                    label = readable_labels.get(col, col)
                    col_data = latest_data[col].dropna()
                    if not col_data.empty:
                        min_val = col_data.min()
                        max_val = col_data.max()
                        median_val = col_data.median()
                        summary_html = f"""
                        <div style='font-size:18px; font-weight:600;'>{label}</div>
                        <div style='font-size:14px;'>
                            <b>Median:</b> {median_val:.2f}<br>
                            <b>Min:</b> {min_val:.2f} &nbsp;&nbsp; <b>Max:</b> {max_val:.2f}
                        </div>
                        """
                        cols[idx].markdown(summary_html, unsafe_allow_html=True)
    else:
        st.info("No metrics to summarize.")

    st.markdown("---")
