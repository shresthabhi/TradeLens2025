import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import ollama


# Step 1: Upload filtered data
def clustering_tab():
    tab_input, tab_report = st.tabs(["🔧 K-Means Setup", "📊 Cluster Report"])

    with tab_input:
        uploaded_file = st.file_uploader("Upload filtered dataset (CSV from screener)", type=["csv"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully.")

            exclude_cols = ['gvkey', 'ticker', 'public_date']
            numeric_cols = [col for col in df.select_dtypes(include=['float64', 'int64']).columns if col not in exclude_cols]

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
            label_map = {readable_labels.get(col, col): col for col in numeric_cols}
            display_names = list(label_map.keys())
            selected_labels = st.multiselect("Select metrics for clustering:", options=display_names, default=display_names[:5])
            selected_cols = [label_map[label] for label in selected_labels]

            if selected_cols:
                multi_k_mode = st.checkbox("Compare multiple K values?", value=False)
                if not multi_k_mode:
                    n_clusters = st.slider("Number of clusters (k):", min_value=2, max_value=10, value=3)

                if st.button("Run K-Means Clustering"):
                    st.query_params["tab"] = "report"
                    from sklearn.manifold import TSNE
                    st.session_state["multi_k"] = multi_k_mode
                    X = df[selected_cols].dropna()
                    ids = df.loc[X.index]
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    if not multi_k_mode:
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        cluster_labels = kmeans.fit_predict(X_scaled)

                        tsne = TSNE(n_components=2, random_state=42)
                        tsne_result = tsne.fit_transform(X_scaled)

                        tsne_df = pd.DataFrame(tsne_result, columns=["TSNE1", "TSNE2"])
                        tsne_df["Cluster"] = cluster_labels
                        tsne_df["Company"] = ids.get("ticker", ids.index)

                        st.session_state["cluster_result"] = df.loc[X.index].assign(Cluster=cluster_labels)
                        st.session_state["tsne_df"] = tsne_df
                    else:
                        from sklearn.metrics import silhouette_score
                        sil_scores = {}
                        for k in range(2, 11):
                            km = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
                            sil_scores[k] = silhouette_score(X_scaled, km.labels_)
                        st.session_state["sil_scores"] = sil_scores
                        st.session_state["X"] = X
                        st.session_state["X_scaled"] = X_scaled
                        st.session_state["df_ids"] = ids
                        st.session_state["selected_cols"] = selected_cols

    with tab_report:
        if st.session_state.get("multi_k"):
            selected_k = st.selectbox("Select K to view clustering result:", list(st.session_state["sil_scores"].keys()), index=0)
            kmeans = KMeans(n_clusters=selected_k, random_state=42).fit(st.session_state["X_scaled"])
            labels = kmeans.labels_
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42)
            tsne_result = tsne.fit_transform(st.session_state["X_scaled"])

            tsne_df = pd.DataFrame(tsne_result, columns=["TSNE1", "TSNE2"])
            tsne_df["Cluster"] = labels
            tsne_df["Company"] = st.session_state["df_ids"].get("ticker", st.session_state["df_ids"].index)
            cluster_counts = pd.Series(labels).value_counts().sort_index()
            from sklearn.metrics import silhouette_score
            score = silhouette_score(st.session_state["X_scaled"], labels)
            st.markdown(f"<h3 style='margin-top: 0;'>Silhouette Score: <span style='color:#1f77b4'>{score:.2f}</span></h3>", unsafe_allow_html=True)

            from plotly.subplots import make_subplots
            import plotly.graph_objects as go

            fig = make_subplots(rows=1, cols=2, subplot_titles=("t-SNE Cluster Plot", "Cluster Distribution"))
            for cluster in tsne_df["Cluster"].unique():
                subset = tsne_df[tsne_df["Cluster"] == cluster]
                fig.add_trace(go.Scatter(x=subset["TSNE1"], y=subset["TSNE2"], mode="markers", name=f"Cluster {cluster}", text=subset["Company"], marker=dict(size=8)), row=1, col=1)
            fig.add_trace(go.Bar(x=cluster_counts.index.astype(str), y=cluster_counts.values, name="Cluster Count"), row=1, col=2)
            fig.update_layout(height=500, width=1100, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

            # Static silhouette plot
            sil_df = pd.DataFrame.from_dict(st.session_state["sil_scores"], orient="index", columns=["Silhouette"])
            sil_df.reset_index(inplace=True)
            sil_df.rename(columns={"index": "K"}, inplace=True)
            fig_sil = px.line(sil_df, x="K", y="Silhouette", markers=True, title="Silhouette Score vs. K")
            st.plotly_chart(fig_sil, use_container_width=True)

        elif "tsne_df" in st.session_state:
            st.subheader("📍 Cluster Report")
            st.markdown("The silhouette score is a measure of how well-separated and cohesive each cluster is. A score close to 1 indicates well-defined clusters, while a score close to 0 suggests overlapping clusters.")

            from sklearn.metrics import silhouette_score
            X = st.session_state["cluster_result"][selected_cols].dropna()
            X_scaled = StandardScaler().fit_transform(X)
            score = silhouette_score(X_scaled, st.session_state["cluster_result"]["Cluster"])
            st.markdown(f"<h3 style='margin-top: 0;'>Silhouette Score: <span style='color:#1f77b4'>{score:.2f}</span></h3>", unsafe_allow_html=True)
            X = st.session_state["cluster_result"][selected_cols].dropna()
            X_scaled = StandardScaler().fit_transform(X)
            score = silhouette_score(X_scaled, st.session_state["cluster_result"]["Cluster"])

            
            from plotly.subplots import make_subplots

            # t-SNE scatter plot and bar chart side-by-side
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go

            fig = make_subplots(rows=1, cols=2, subplot_titles=("t-SNE Cluster Plot", "Cluster Distribution"))

            cluster_data = st.session_state["tsne_df"]
            for cluster in cluster_data["Cluster"].unique():
                cluster_subset = cluster_data[cluster_data["Cluster"] == cluster]
                fig.add_trace(
                    go.Scatter(
                        x=cluster_subset["TSNE1"],
                        y=cluster_subset["TSNE2"],
                        mode="markers",
                        name=f"Cluster {cluster}",
                        text=cluster_subset["Company"],
                        marker=dict(size=8)
                    ),
                    row=1, col=1
                )

            cluster_counts = st.session_state["cluster_result"]["Cluster"].value_counts().sort_index()
            fig.add_trace(
                go.Bar(
                    x=cluster_counts.index.astype(str),
                    y=cluster_counts.values,
                    name="Cluster Count"
                ),
                row=1, col=2
            )

            fig.update_layout(height=500, width=1100, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "📥 Download Clustered Data",
                st.session_state["cluster_result"].to_csv(index=False),
                "clustered_output.csv",
                "text/csv"
            )

            # AI Summary button and OpenAI integration
            st.markdown("---")
            if st.button("🧠 Generate AI Cluster Summary"):

                st.subheader("🔍 Interpreting Clusters Using AI")
                kmeans = KMeans(n_clusters=st.session_state["cluster_result"]["Cluster"].nunique(), random_state=42)
                cluster_data = st.session_state["cluster_result"][selected_cols].dropna()
                kmeans.fit(StandardScaler().fit_transform(cluster_data))
                centers = pd.DataFrame(kmeans.cluster_centers_, columns=selected_cols)

                readable = {
                    col: readable_labels.get(col, col) for col in centers.columns
                }

                prompt = """
You are a financial analyst. The following are the cluster centers from a K-Means clustering analysis on company financial ratios. 
Each cluster center represents the average financial profile of companies in that cluster.

Please summarize the key traits and likely interpretation of each cluster.

Order of data provided for each cluser : 
"""             
                for col in centers.columns:
                    prompt += readable_labels.get(col, col) + ", "
                prompt +="""
Cluster values are :
"""
                for index, row in centers.iterrows():
                    prompt += f"Cluster {index} {row}"
                    prompt += """
"""

                prompt += """Provide concise insights in bullet points for each cluster.
Also format the output in markdown tabular format, one column for cluster number, another for key traits, and last for descriotion.
Format with proper highlighting of text, like bold, italics and use some standard emojis for high low.
Don't use html tags"""

                response = ollama.generate(
                    model='mistral',
                    prompt=prompt
                )
                st.markdown(response['response'])