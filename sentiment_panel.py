import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from gnews import GNews
from newspaper import Article
import requests


@st.cache_resource
def load_finbert():
    tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def resolve_real_url(google_news_link):
    try:
        response = requests.get(google_news_link, allow_redirects=True, timeout=100)
        print(f"new url : {response.url}")
        return response.url
    except Exception as e:
        print("Failed to resolve:", e)
        return None

def get_news(ticker):
    google_news = GNews(language='en', max_results=10)
    raw_news = google_news.get_news(f"{ticker} stock")

    for item in raw_news:
        gnews_link = item.get('url')
        real_url = resolve_real_url(gnews_link)
        item['real_url'] = real_url

    return raw_news

def sentiment_analysis_panel():
    st.header("🧾 Company News + Sentiment Analysis")
    st.warning("""
⚠️ **Important Notice**  
This app makes live API calls to a financial data library. Since it's hosted on Streamlit Cloud, these calls may occasionally **fail due to rate limits or API restrictions**.

Additionally, FinBERT sentiment analysis **won't work on Streamlit Cloud** because it requires the `torch` library, which is difficult to install in this environment.

👉 To run this app with full functionality, including sentiment analysis, **please follow the setup instructions in the [README](https://github.com/sanjalD/TradeLens/tree/main) and host it locally.**
""")
    ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA, NVDA)", value="AAPL")

    if st.button("Fetch & Analyze News"):
        news_items = get_news(ticker.upper())

        if not news_items:
            st.warning("No news found or invalid ticker.")
        else:
            with st.spinner("Loading FinBERT and analyzing news..."):
                finbert = load_finbert()
                results = []

                for item in news_items:
                    title = item.get("title")
                    content = item.get("content")
                    link = item.get("real_url", "N/A")

                    article = Article(link)
                    article.download()
                    article.parse()

                    content = article.text

                    content = f"{title} {content}"  if content is not None and content != "" else title

                    if not title:
                        continue  # skip if no title

                    try:
                        # sentiment = finbert(title)[0]
                        sentiment = finbert(content)[0]
                        results.append({
                            "Title": f"[{title}]({link})",
                            "Sentiment": sentiment["label"],
                            "Confidence": round(sentiment["score"], 2)
                        })
                    except Exception as e:
                        st.error(f"Failed to analyze: {title}\nError: {e}")

            if results:
                results_df = pd.DataFrame(results)
                st.subheader(f"📰 News Sentiment for {ticker.upper()}")
                st.markdown(results_df.to_markdown(index=False), unsafe_allow_html=True)

                fig = px.histogram(
                    results_df,
                    x="Sentiment",
                    title="Sentiment Distribution",
                    category_orders={"Sentiment": ["Positive", "Neutral", "Negative"]},
                    color="Sentiment",
                    color_discrete_map = {
                        "Positive": "#1e7c1e",  # richer green
                        "Neutral": "#2c4f9e",   # deeper blue
                        "Negative": "#a32020"   # saturated red
                    }
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid news articles to analyze.")

