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
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import mean_squared_error, r2_score

# PAGE CONFIG

st.set_page_config(page_title="Hybrid Stock Prediction Dashboard", layout="wide")
st.title("Hybrid Stock Prediction Dashboard")

# CLEAN TEXT

def clean_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^a-zA-Z0-9.,!? ]+", "", text)
    text = re.sub(r"\b[a-zA-Z]\b", "", text)
    return text


# OCR (CACHED)
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

reader = load_ocr()

# MODEL BUILDERS

def build_lstm_model(input_shape):
    return tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1)
    ])

def build_sentiment_model(vocab_size, max_len):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64, input_length=max_len),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])


# SIDEBAR INPUTS

st.sidebar.header("📌 Stock Data")
stock_name = st.sidebar.text_input("Enter Stock Name", "RELIANCE")
ticker = stock_name.upper().strip() + ".NS"
try:
    stock_info = yf.Ticker(ticker).info
    full_name = stock_info.get("longName", stock_name.upper())
except:
    full_name = stock_name.upper()
    
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input(
    "End Date",
    pd.to_datetime("today") - pd.Timedelta(days=1)
)

st.sidebar.header("📰 Sentiment Analysis")
if "news_text" not in st.session_state:
    st.session_state["news_text"] = ""

uploaded_image = st.sidebar.file_uploader("Upload News Image", type=["png", "jpg", "jpeg"])
if uploaded_image:
    extracted = reader.readtext(uploaded_image.read(), detail=0)
    st.session_state["news_text"] = clean_text(" ".join(extracted))
    st.sidebar.success("OCR text loaded!")

news_input = st.sidebar.text_area("Enter News Text:", key="news_text")
sentiment_ready = news_input.strip() != ""

# LOAD STOCK DATA

df = yf.download(ticker, start=start_date, end=end_date)

if df.empty:
    st.error("No stock data found.")
    st.stop()

# Sort by Date index in descending order
df_desc = df.sort_index(ascending=False)



st.subheader(f"Historical Stock Data — {full_name}")
st.dataframe(df_desc.head(30))

# TECHNICAL INDICATORS

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

# LSTM DATA PREP

features = df[['Open','High','Low','Close','Volume','MA20','RSI','MACD','Signal']]
scaler = MinMaxScaler()
scaled = scaler.fit_transform(features)

SEQ_LEN = 60
X, y = [], []
for i in range(SEQ_LEN, len(scaled)):
    X.append(scaled[i-SEQ_LEN:i])
    y.append(scaled[i,3])

X, y = np.array(X), np.array(y)
split = int(len(X)*0.8)
X_train, X_test = X[:split], X[split:]
y_test = y[split:]

# LOAD LSTM WEIGHTS

if not os.path.exists("lstm_weights.weights.h5"):
    st.error("Missing lstm_weights.weights.h5")
    st.stop()

lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
lstm_model.load_weights("lstm_weights.weights.h5")

# LSTM PREDICTION

pred_scaled = lstm_model.predict(X_test, verbose=0)

close_scaler = MinMaxScaler()
close_scaler.min_, close_scaler.scale_ = scaler.min_[3:4], scaler.scale_[3:4]
predictions = close_scaler.inverse_transform(pred_scaled)
actual = close_scaler.inverse_transform(y_test.reshape(-1,1))

# PLOT ACTUAL VS PREDICTED
st.subheader("Actual vs Predicted")

test_dates = df.index[SEQ_LEN + split:]  # dates aligned with X_test

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=test_dates,
    y=actual.flatten(),
    name="Actual"
))
fig.add_trace(go.Scatter(
    x=test_dates,
    y=predictions.flatten(),
    name="Predicted"
))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price (₹)",
    hovermode="x unified"
)

st.plotly_chart(fig, width='stretch')

# LOAD SENTIMENT MODEL (WEIGHTS ONLY)

with open("config.pkl","rb") as f:
    config = pickle.load(f)

with open("tokenizer.json", "r") as f:
    tokenizer = tokenizer_from_json(f.read())

if not os.path.exists("sentiment_weights.weights.h5"):
    st.error("Missing sentiment_weights.weights.h5")
    st.stop()

sentiment_model = build_sentiment_model(config["vocab_size"], config["max_len"])
sentiment_model.load_weights("sentiment_weights.weights.h5")

# SENTIMENT ANALYSIS

st.subheader("Sentiment Analysis Result")
if not sentiment_ready:
    st.caption("Add news text or upload a news image to enable sentiment analysis.")

prob = 0.5  # default neutral
if sentiment_ready:
    seq = tokenizer.texts_to_sequences([news_input])
    padded = pad_sequences(seq, maxlen=config["max_len"], padding="post")
    prob = sentiment_model.predict(padded, verbose=0)[0][0]
    score = prob * 100

    if prob > 0.6:
        label, color = "Positive 😊", "green"
    elif prob < 0.4:
        label, color = "Negative 😟", "red"
    else:
        label, color = "Neutral 😐", "orange"

    st.markdown(f"<h3 style='color:{color}'>{label}</h3>", unsafe_allow_html=True)
    st.write(f"Sentiment Score: {score:.2f}%")

# HYBRID SIGNAL

price_change = (predictions[-1][0] - actual[-1][0]) / actual[-1][0] * 100
combined = 0.7 * price_change + 0.3 * prob
if combined > 0.2:
    signal, sig_color = "BUY", "green"
elif combined < -0.2:
    signal, sig_color = "SELL", "red"
else:
    signal, sig_color = "HOLD", "orange"

st.subheader("Hybrid Trading Signal")
st.markdown(f"<h2 style='color:{sig_color}; text-align:center'>{signal}</h2>", unsafe_allow_html=True)

# 7-DAY FORECAST
st.subheader("7-Day Forecast")

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

# Plot forecast
st.line_chart(future)

# Create forecast table
forecast_dates = pd.date_range(
    start=df.index[-1] + pd.Timedelta(days=1),
    periods=7,
    freq="B"   # Business days (skips weekends)
)

forecast_df = pd.DataFrame({
    "Date": forecast_dates.date,
    "Predicted Price (₹)": np.round(future, 2)
})

st.subheader("7-Day Price Forecast Table")
st.dataframe(forecast_df, use_container_width=True)

# MODEL PERFORMANCE METRICS

rmse = np.sqrt(mean_squared_error(actual, predictions))
r2 = r2_score(actual, predictions)

st.subheader("Model Performance Metrics")

col1, col2 = st.columns(2)

with col1:
    st.metric(label="RMSE", value=f"{rmse:.2f}")

with col2:
    st.metric(label="R² Score", value=f"{r2:.4f}")
st.caption(
    "RMSE measures average price error in ₹. R² indicates how well the model explains price variance."
)

st.markdown("---")
st.info(
    "⚠️**Disclaimer:** This project is developed for academic purposes only, stock market prices are highly volatile and influenced by external factors. Predictions should not be used for real-world trading or investment decisions."
)



