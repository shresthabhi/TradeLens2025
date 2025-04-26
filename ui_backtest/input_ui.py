import streamlit as st
import pandas as pd
from datetime import date
from .run_backtest import run_backtest_from_ui

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
        "debt_at": "Total Debt/Total Assets",
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


# Define default values
DEFAULT_START_DATE = date(2018, 1, 1)
DEFAULT_END_DATE = date(2020, 12, 1)
DEFAULT_ENTRY_CONDITION = ('pe_inc', '>', 30.0)
DEFAULT_EXIT_CONDITION = ('pe_inc', '>', 50.0)
DEFAULT_OPERATORS = [">", "<", ">=", "<=", "==", "!="]
METRIC_OPTIONS = list(readable_labels.keys())


def show_backtest_input():

    st.title("📊 Fundamental Backtester")

    st.header("1. Upload Tickers")
    ticker_file = st.file_uploader("Upload a CSV with tickers", type=["csv"])

    st.header("2. Select Date Range")
    col1, col2 = st.columns(2)
    with col1:
        # Use default start date
        start_date = st.date_input("Start Date", value=DEFAULT_START_DATE)
    with col2:
        # Use default end date
        end_date = st.date_input("End Date", value=DEFAULT_END_DATE)

    # --- Entry Conditions ---
    st.header("3. Entry Condition")
    entry_condition_type = st.selectbox("Join type for multiple conditions", ["AND", "OR"], key="entry_type")

    # Initialize session state with defaults IF they don't exist
    if 'entry_condition_count' not in st.session_state:
        st.session_state.entry_condition_count = 1
    if 'entry_conditions' not in st.session_state:
        st.session_state.entry_conditions = [DEFAULT_ENTRY_CONDITION]

    # Use a temporary list to store widget values for this run
    entry_condition_rows = []
    rerun_entry = False

    for i in range(st.session_state.entry_condition_count):
        # Retrieve current condition for this index from session state
        current_metric, current_op, current_val = st.session_state.entry_conditions[i] if i < len(st.session_state.entry_conditions) else (METRIC_OPTIONS[0], DEFAULT_OPERATORS[0], 0.0)

        # Calculate default indices for selectboxes
        try:
            metric_index = METRIC_OPTIONS.index(current_metric)
        except ValueError:
            metric_index = 0 # Default to first metric if current_metric is somehow invalid

        try:
            op_index = DEFAULT_OPERATORS.index(current_op)
        except ValueError:
            op_index = 0 # Default to first operator

        c1, c2, c3, c4 = st.columns([3, 2, 3, 1])
        with c1:
            # Set default index for metric
            metric = st.selectbox(f"Metric #{i+1}", METRIC_OPTIONS, index=metric_index, format_func=lambda x: readable_labels[x], key=f"entry_metric_{i}")
        with c2:
            # Set default index for operator
            comparator = st.selectbox("Operator", DEFAULT_OPERATORS, index=op_index, key=f"entry_op_{i}")
        with c3:
            # Set default value
            value = st.number_input("Value", value=float(current_val), key=f"entry_val_{i}", format="%f")
        with c4:
            remove_button = st.button("❌", key=f"entry_remove_{i}")
            if remove_button and st.session_state.entry_condition_count > 1: # Prevent removing the last condition
                 # Mark for removal processing after the loop
                 entry_condition_rows.append({'remove': True, 'index': i})
                 rerun_entry = True
                 continue # Skip appending this row if marked for removal
            elif remove_button:
                 st.warning("Cannot remove the last condition.")


        # Store current widget values (if not removed)
        entry_condition_rows.append({
            'metric': metric,
            'comparator': comparator,
            'value': value,
            'remove': False,
            'index': i
        })

    # Process updates to session state AFTER the loop
    new_entry_conditions = []
    new_count = 0
    for row in entry_condition_rows:
        if not row['remove']:
            new_entry_conditions.append((row['metric'], row['comparator'], row['value']))
            new_count += 1

    # Only update state if changes occurred or count differs
    if st.session_state.entry_conditions != new_entry_conditions or st.session_state.entry_condition_count != new_count:
         st.session_state.entry_conditions = new_entry_conditions
         st.session_state.entry_condition_count = new_count


    if st.button("Add Entry Condition"):
        st.session_state.entry_condition_count += 1
        # Add a generic default for the new condition
        st.session_state.entry_conditions.append((METRIC_OPTIONS[0], DEFAULT_OPERATORS[0], 0.0))
        rerun_entry = True 

    if rerun_entry:
        st.rerun()


    # --- Exit Conditions (apply similar logic) ---
    st.header("4. Exit Condition")
    exit_condition_type = st.selectbox("Join type for multiple conditions", ["AND", "OR"], key="exit_type")

    # Initialize session state with defaults IF they don't exist
    if 'exit_condition_count' not in st.session_state:
        st.session_state.exit_condition_count = 1
    if 'exit_conditions' not in st.session_state:
        st.session_state.exit_conditions = [DEFAULT_EXIT_CONDITION]

    exit_condition_rows = []
    rerun_exit = False

    for i in range(st.session_state.exit_condition_count):
        current_metric, current_op, current_val = st.session_state.exit_conditions[i] if i < len(st.session_state.exit_conditions) else (METRIC_OPTIONS[0], DEFAULT_OPERATORS[0], 0.0)

        try:
            metric_index = METRIC_OPTIONS.index(current_metric)
        except ValueError:
            metric_index = 0
        try:
            op_index = DEFAULT_OPERATORS.index(current_op)
        except ValueError:
            op_index = 0

        c1, c2, c3, c4 = st.columns([3, 2, 3, 1])
        with c1:
            metric = st.selectbox(f"Metric #{i+1}", METRIC_OPTIONS, index=metric_index, format_func=lambda x: readable_labels[x], key=f"exit_metric_{i}")
        with c2:
            comparator = st.selectbox("Operator", DEFAULT_OPERATORS, index=op_index, key=f"exit_op_{i}")
        with c3:
            value = st.number_input("Value", value=float(current_val), key=f"exit_val_{i}", format="%f")
        with c4:
            remove_button = st.button("❌", key=f"exit_remove_{i}")
            if remove_button and st.session_state.exit_condition_count > 1:
                exit_condition_rows.append({'remove': True, 'index': i})
                rerun_exit = True
                continue
            elif remove_button:
                 st.warning("Cannot remove the last condition.")

        exit_condition_rows.append({
            'metric': metric,
            'comparator': comparator,
            'value': value,
            'remove': False,
            'index': i
        })

    new_exit_conditions = []
    new_exit_count = 0
    for row in exit_condition_rows:
        if not row['remove']:
            new_exit_conditions.append((row['metric'], row['comparator'], row['value']))
            new_exit_count += 1

    if st.session_state.exit_conditions != new_exit_conditions or st.session_state.exit_condition_count != new_exit_count:
        st.session_state.exit_conditions = new_exit_conditions
        st.session_state.exit_condition_count = new_exit_count

    if st.button("Add Exit Condition"):
        st.session_state.exit_condition_count += 1
        st.session_state.exit_conditions.append((METRIC_OPTIONS[0], DEFAULT_OPERATORS[0], 0.0))
        rerun_exit = True

    if rerun_exit:
        st.rerun()


    # --- Rebalancing Strategy ---
    st.header("5. Rebalancing Strategy")
    rebalance_mode = st.selectbox("Rebalance Mode", ["event", "periodic"])
    rebalance_strategy = st.selectbox("Rebalance Strategy", ["sharpe", "min_volatility"])
    rebalance_frequency = None
    if rebalance_mode == "periodic":
        rebalance_frequency = st.number_input("Rebalance Frequency (days)", min_value=1, value=30)

    st.header("6. Holding Period")
    holding_period = st.number_input("Value", min_value=15, value = 90)


    # --- Run Backtest Button ---
    if st.button("Run Backtest"):
        # Get conditions directly from session state as they are now managed correctly
        entry_conditions_list = st.session_state.get('entry_conditions', [])
        exit_conditions_list = st.session_state.get('exit_conditions', [])

        # Build individual condition strings safely
        entry_condition_strs = []
        for m, op, v in entry_conditions_list:
            condition_str = f"(f.get('{m}') is not None and f.get('{m}') {op} {float(v)})"
            entry_condition_strs.append(condition_str)

        exit_condition_strs = []
        for m, op, v in exit_conditions_list:
            condition_str = f"(f.get('{m}') is not None and f.get('{m}') {op} {float(v)})"
            exit_condition_strs.append(condition_str)

        # Join the conditions with AND/OR
        if entry_condition_strs:
            entry_str = f"({ ' {} '.format(entry_condition_type.lower()).join(entry_condition_strs) })"
        else:
            entry_str = "True" # Default entry: allow all

        if exit_condition_strs:
            exit_str = f"({ ' {} '.format(exit_condition_type.lower()).join(exit_condition_strs) })"
        else:
            exit_str = "False" # Default exit: never exit

        print("ENTRY CONDITION STRING:", entry_str)
        print("EXIT CONDITION STRING:", exit_str)

        try:
            entry_fn = eval(f"lambda f: {entry_str}")
            exit_fn = eval(f"lambda f: {exit_str}")

            # Store functions in session state FIRST
            st.session_state.entry_lambda = entry_fn
            st.session_state.exit_lambda = exit_fn
            # st.success("Lambda functions created.")
            # st.code(f"Entry Lambda:\nlambda f: {entry_str}")
            # st.code(f"Exit Lambda:\nlambda f: {exit_str}")

            # Now run the backtest if ticker file is provided
            if ticker_file is not None:
                ticker_df = pd.read_csv(ticker_file)
                tickers_list = ticker_df['ticker'].unique().tolist() # Ensure it's a list

                st.info("Running backtest... Please wait.") # Provide feedback

                nav_df, trade_log_df = run_backtest_from_ui(
                    tickers=tickers_list,
                    start_date=start_date, # Use the value from date_input
                    end_date=end_date,     # Use the value from date_input
                    entry_lambda=entry_fn,
                    exit_lambda=exit_fn,
                    rebalance_mode=rebalance_mode,
                    rebalance_strategy=rebalance_strategy,
                    rebalance_frequency=rebalance_frequency,
                    holding_period = holding_period
                )

                st.session_state.nav_df = nav_df
                st.session_state.trade_log_df = trade_log_df
                st.success("✅ Backtest complete. Results stored.")
                # Optionally display results summary here
                # st.dataframe(nav_df.head())
                # st.dataframe(trade_log_df.head())
            else:
                st.warning("Please upload a ticker CSV file to run the backtest.")


        except SyntaxError as e:
            st.error(f"Syntax Error creating strategy functions: {e}")
            st.error(f"Please check the conditions syntax.")
            st.error(f"Generated Entry String: {entry_str}")
            st.error(f"Generated Exit String: {exit_str}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error(f"Generated Entry String: {entry_str}")
            st.error(f"Generated Exit String: {exit_str}")
