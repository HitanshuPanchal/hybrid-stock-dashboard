# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import re
import unicodedata
import easyocr
import os
import tensorflow as tf
import plotly.graph_objs as go

from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(page_title="Hybrid Stock Prediction Dashboard", layout="wide")
st.title("Hybrid Stock Prediction Dashboard")


# ============================================================
# TEXT CLEANING FUNCTION (FOR OCR + SENTIMENT)
# ============================================================
def clean_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^a-zA-Z0-9.,!? ]+", "", text)
    text = re.sub(r"\b[a-zA-Z]\b", "", text)
    return text


# ============================================================
# OCR READER (CACHED)
# ============================================================
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()


# ============================================================
# MODEL BUILDERS
# ============================================================
def build_lstm_model(input_shape):
    return tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1)
    ])

def build_sentiment_model(vocab_size, max_len):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])


# ============================================================
# LOAD NSE COMPANY LIST
# ============================================================
CSV_URL = "https://raw.githubusercontent.com/imanojkumar/NSE-India-All-Stocks-Tickers-Data/main/Ticker_List_NSE_India.csv"

@st.cache_data
def load_nse_companies():
    df = pd.read_csv(CSV_URL)
    df = df[["SYMBOL", "NAME OF COMPANY"]]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

nse_df = load_nse_companies()

# Create label for dropdown
nse_df["display"] = nse_df["NAME OF COMPANY"] + " (" + nse_df["SYMBOL"] + ")"


# ============================================================
# SIDEBAR : STOCK SELECTION
# ============================================================
st.sidebar.header("📌 Stock Data")

# Default stock selection
if "RELIANCE" in nse_df["SYMBOL"].values:
    default_index = int(nse_df.index[nse_df["SYMBOL"] == "RELIANCE"][0])
else:
    default_index = 0

selected_company = st.sidebar.selectbox(
    "Search Company",
    options=nse_df["display"].tolist(),
    index=default_index
)

# Extract stock symbol
selected_symbol = nse_df.loc[
    nse_df["display"] == selected_company, "SYMBOL"
].iloc[0]

ticker = selected_symbol + ".NS"

# Fetch full company name from Yahoo Finance
try:
    stock_info = yf.Ticker(ticker).info
    full_name = stock_info.get("longName", selected_company)
except:
    full_name = selected_company


# ============================================================
# DATE RANGE INPUT
# ============================================================
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input(
    "End Date",
    pd.to_datetime("today") - pd.Timedelta(days=1)
)


# ============================================================
# SIDEBAR : SENTIMENT ANALYSIS INPUT
# ============================================================
st.sidebar.header("📰 Sentiment Analysis")

if "news_text" not in st.session_state:
    st.session_state["news_text"] = ""

uploaded_image = st.sidebar.file_uploader(
    "Upload News Image", type=["png", "jpg", "jpeg"]
)

# OCR extraction from uploaded image
if uploaded_image:
    extracted = reader.readtext(uploaded_image.read(), detail=0)
    st.session_state["news_text"] = clean_text(" ".join(extracted))
    st.sidebar.success("OCR text loaded!")

news_input = st.sidebar.text_area("Enter News Text:", key="news_text")
sentiment_ready = news_input.strip() != ""


# ============================================================
# LOAD STOCK PRICE DATA
# ============================================================
df = yf.download(ticker, start=start_date, end=end_date)

if df.empty:
    st.error("No stock data found.")
    st.stop()

df_desc = df.sort_index(ascending=False)

st.subheader(f"Historical Stock Data — {full_name}")
st.dataframe(df_desc.head(30))


# ============================================================
# TECHNICAL INDICATORS
# ============================================================
df["MA20"] = df["Close"].rolling(20).mean()

delta = df["Close"].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
df["RSI"] = 100 - (100 / (1 + avg_gain / avg_loss))

EMA12 = df["Close"].ewm(span=12).mean()
EMA26 = df["Close"].ewm(span=26).mean()
df["MACD"] = EMA12 - EMA26
df["Signal"] = df["MACD"].ewm(span=9).mean()

df.dropna(inplace=True)


# ============================================================
# LSTM DATA PREPARATION
# ============================================================
features = df[['Open','High','Low','Close','Volume','MA20','RSI','MACD','Signal']]
scaler = MinMaxScaler()
scaled = scaler.fit_transform(features)

SEQ_LEN = 60
X, y = [], []

for i in range(SEQ_LEN, len(scaled)):
    X.append(scaled[i-SEQ_LEN:i])
    y.append(scaled[i, 3])

X, y = np.array(X), np.array(y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_test = y[split:]


# ============================================================
# LOAD LSTM MODEL (WEIGHTS ONLY)
# ============================================================
lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
lstm_model.build((None, X_train.shape[1], X_train.shape[2]))
lstm_model.load_weights("lstm_weights.weights.h5")


# ============================================================
# LSTM PREDICTIONS
# ============================================================
pred_scaled = lstm_model.predict(X_test, verbose=0)

close_scaler = MinMaxScaler()
close_scaler.min_, close_scaler.scale_ = scaler.min_[3:4], scaler.scale_[3:4]

predictions = close_scaler.inverse_transform(pred_scaled)
actual = close_scaler.inverse_transform(y_test.reshape(-1, 1))


# ============================================================
# ACTUAL VS PREDICTED PLOT
# ============================================================
st.subheader("Actual vs Predicted")

test_dates = df.index[SEQ_LEN + split:]

fig = go.Figure()
fig.add_trace(go.Scatter(x=test_dates, y=actual.flatten(), name="Actual"))
fig.add_trace(go.Scatter(x=test_dates, y=predictions.flatten(), name="Predicted"))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price (₹)",
    hovermode="x unified"
)

st.plotly_chart(fig, width='stretch')


# ============================================================
# LOAD SENTIMENT MODEL
# ============================================================
with open("config.pkl", "rb") as f:
    config = pickle.load(f)

with open("tokenizer.json", "r") as f:
    tokenizer = tokenizer_from_json(f.read())

sentiment_model = build_sentiment_model(config["vocab_size"], config["max_len"])
sentiment_model.build((None, config["max_len"]))
sentiment_model.load_weights("sentiment_weights.weights.h5")


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================
st.subheader("Sentiment Analysis Result")

if not sentiment_ready:
    st.caption("Add news text or upload a news image to enable sentiment analysis.")

prob = 0.5

if sentiment_ready:
    seq = tokenizer.texts_to_sequences([news_input])
    padded = pad_sequences(seq, maxlen=config["max_len"], padding="post")
    prob = sentiment_model.predict(padded, verbose=0)[0][0]

    score = prob * 100

    if prob > 0.6:
        label, color = "Positive", "green"
    elif prob < 0.4:
        label, color = "Negative", "red"
    else:
        label, color = "Neutral", "orange"

    st.markdown(f"<h3 style='color:{color}'>{label}</h3>", unsafe_allow_html=True)
    st.write(f"Sentiment Score: {score:.2f}%")


# ============================================================
# HYBRID TRADING SIGNAL
# ============================================================
price_change = (predictions[-1][0] - actual[-1][0]) / actual[-1][0] * 100
combined = 0.7 * price_change + 0.3 * prob

if combined > 0.2:
    signal, sig_color = "BUY", "green"
elif combined < -0.2:
    signal, sig_color = "SELL", "red"
else:
    signal, sig_color = "HOLD", "orange"

st.subheader("Hybrid Trading Signal")
st.markdown(
    f"<h2 style='color:{sig_color}; text-align:center'>{signal}</h2>",
    unsafe_allow_html=True
)


# ============================================================
# 7-DAY PRICE FORECAST
# ============================================================
future = []
last_seq = scaled[-SEQ_LEN:].copy()

for _ in range(7):
    pred_scaled = lstm_model.predict(
        last_seq.reshape(1, SEQ_LEN, features.shape[1]),
        verbose=0
    )

    pred_price = close_scaler.inverse_transform(pred_scaled)[0][0]
    future.append(pred_price)

    new_row = last_seq[-1].copy()
    new_row[3] = pred_scaled[0][0]
    last_seq = np.vstack([last_seq[1:], new_row])

forecast_dates = pd.date_range(
    start=df.index[-1] + pd.Timedelta(days=1),
    periods=7,
    freq="B"
)

forecast_df = pd.DataFrame({
    "Date": forecast_dates.date,
    "Predicted Price (₹)": np.round(future, 2)
})

st.subheader("7-Day Price Forecast Table")
st.dataframe(forecast_df, width='stretch')


# ============================================================
# MODEL PERFORMANCE METRICS
# ============================================================
actual = np.array(actual)
predictions = np.array(predictions)

rmse = np.sqrt(mean_squared_error(actual, predictions))
r2 = r2_score(actual, predictions)

mask = ~np.isnan(actual) & ~np.isnan(predictions)
actual_clean = actual[mask]
pred_clean = predictions[mask]

if len(actual_clean) > 1:
    actual_diff = np.diff(actual_clean)
    pred_diff = np.diff(pred_clean)

    valid_mask = (actual_diff != 0) & (pred_diff != 0)

    if np.any(valid_mask):
        directional_accuracy = np.mean(
            np.sign(actual_diff[valid_mask]) ==
            np.sign(pred_diff[valid_mask])
        )
    else:
        directional_accuracy = 0.0
else:
    directional_accuracy = 0.0

if directional_accuracy >= 0.60:
    da_status = "Strong"
elif directional_accuracy >= 0.55:
    da_status = "Good"
else:
    da_status = "Needs Improvement"


# ============================================================
# METRICS UI
# ============================================================
st.subheader("Model Performance Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("RMSE (₹)", f"{rmse:.2f}")

with col2:
    st.metric("R² Score", f"{r2:.4f}")

with col3:
    st.metric("Directional Accuracy", f"{directional_accuracy * 100:.2f}%")

st.caption(
    "RMSE measures average prediction error in ₹. "
    "R² indicates how well the model explains price variance. "
    "Directional Accuracy shows how often the model correctly "
    "predicts the direction of price movement."
)

st.success(f"Directional Accuracy Assessment: **{da_status}**")

st.markdown("---")

st.info(
    "⚠️ **Disclaimer:** This project is developed for academic purposes only. "
    "Stock market prices are highly volatile and influenced by external factors. "
    "Predictions should not be used for real-world trading or investment decisions."
)
