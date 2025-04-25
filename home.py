import streamlit as st

def home():
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.8rem;
            font-weight: 700;
            color: #4A90E2;
            margin-bottom: 0.5rem;
        }
        .sub-title {
            font-size: 1.2rem;
            color: #333;
            max-width: 900px;
        }
        .creators {
            font-size: 1.1rem;
            color: #444;
            margin-top: 2rem;
        }
        .creators a {
            color: #4A90E2;
            text-decoration: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-title">📊 TradeLens: Integrated Stock Analysis Platform</div>', unsafe_allow_html=True)

    st.markdown(
        '''
        <div class="sub-title">
            Advanced investment tools—such as fundamental screeners, sentiment analytics, and strategy backtesting—are often spread across multiple platforms, making the analysis process time-consuming and fragmented.
            <br><br>
            <strong>TradeLens</strong> consolidates these capabilities into one streamlined, no-code platform, enabling efficient and data-driven investment workflows.
            <br><br>
            <strong>Key Features:</strong>
            <ul>
                <li><strong>📋 Stock Screening & Filtering:</strong> Evaluate stocks based on financial ratios like Beta, Debt-to-Equity, and more.</li>
                <li><strong>🔍 Analytical Tools:</strong> Perform sentiment analysis using FinBERT, assess portfolio diversification, and leverage machine learning models like KMeans for deeper insights.</li>
                <li><strong>📈 Backtesting:</strong> Simulate your investment strategies with custom entry and exit conditions over historical stock data—no coding required.</li>
            </ul>
        </div>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(" ")
    st.success("Use the tabs on the left to explore and get started.")

    st.markdown(
        '''
        🔧 Built by:<br><br>
        • <strong>Abhinav Shresth</strong> – <a href="mailto:abhinav8@bu.edu">abhinav8@bu.edu</a><br>
        • <strong>Parampal Singh Sidhu</strong> – <a href="mailto:parampal@bu.edu">parampal@bu.edu</a><br>
        • <strong>Sanjal Atul Desai</strong> – <a href="mailto:sanjal@bu.edu">sanjal@bu.edu</a>
        ''',
        unsafe_allow_html=True
    )
