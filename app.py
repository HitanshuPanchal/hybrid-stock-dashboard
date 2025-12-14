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

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Hybrid Stock Prediction Dashboard",
    layout="wide"
)

st.title("Hybrid Stock Prediction Dashboard (LSTM + ANN + OCR)")

# ------------------------------------------------------------
# CLEAN TEXT FUNCTION
# ------------------------------------------------------------
def clean_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^a-zA-Z0-9.,!? ]+", "", text)
    text = re.sub(r"\b[a-zA-Z]\b", "", text)
    text = re.sub(" +", " ", text)
    return text

# ------------------------------------------------------------
# OCR (CACHED)
# ------------------------------------------------------------
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

reader = load_ocr()

# ------------------------------------------------------------
# BUILD LSTM ARCHITECTURE (IMPORTANT)
# ------------------------------------------------------------
def build_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(
            64, return_sequences=True, input_shape=input_shape
        ),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1)
    ])
    return model

# ------------------------------------------------------------
# SIDEBAR INPUT
# ------------------------------------------------------------
st.sidebar.header("📌 Stock & Date Selection")

ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))

# ------------------------------------------------------------
# SENTIMENT INPUT
# ------------------------------------------------------------
st.sidebar.header("📰 Sentiment Analysis (ANN)")

if "news_text" not in st.session_state:
    st.session_state["news_text"] = ""

uploaded_image = st.sidebar.file_uploader(
    "Upload News Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_image:
    st.sidebar.image(uploaded_image, width=250)
    extracted = reader.readtext(uploaded_image.read(), detail=0)
    cleaned = clean_text(" ".join(extracted))

    if cleaned.strip():
        st.session_state["news_text"] = cleaned
        st.sidebar.success("OCR text loaded!")
    else:
        st.sidebar.warning("No readable text found.")

news_input = st.sidebar.text_area(
    "Enter News Text:",
    key="news_text"
)

sentiment_ready = news_input.strip() != ""

# ------------------------------------------------------------
# LOAD STOCK DATA
# ------------------------------------------------------------
df = yf.download(ticker, start=start_date, end=end_date)

if df.empty:
    st.error("No stock data found. Please check ticker or date range.")
    st.stop()

st.subheader("📊 Historical Stock Data")
st.dataframe(df.tail(50))

# ------------------------------------------------------------
# TECHNICAL INDICATORS
# ------------------------------------------------------------
df["MA20"] = df["Close"].rolling(20).mean()

delta = df["Close"].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

df["RSI"] = 100 - (100 / (1 + (avg_gain / avg_loss)))

EMA12 = df["Close"].ewm(span=12, adjust=False).mean()
EMA26 = df["Close"].ewm(span=26, adjust=False).mean()

df["MACD"] = EMA12 - EMA26
df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

df.dropna(inplace=True)

# ------------------------------------------------------------
# LSTM FEATURE PREPARATION
# ------------------------------------------------------------
features = df[
    ['Open', 'High', 'Low', 'Close', 'Volume',
     'MA20', 'RSI', 'MACD', 'Signal']
]

data = features.values
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

SEQ_LEN = 60

X, y = [], []
for i in range(SEQ_LEN, len(scaled)):
    X.append(scaled[i-SEQ_LEN:i])
    y.append(scaled[i, 3])  # Close price

X, y = np.array(X), np.array(y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ------------------------------------------------------------
# LOAD LSTM WEIGHTS (FINAL FIX)
# ------------------------------------------------------------
WEIGHTS_PATH = "lstm_weights.weights.h5"

if not os.path.exists(WEIGHTS_PATH):
    st.error("LSTM weights file not found. Upload lstm_weights.weights.h5")
    st.stop()

lstm_model = build_lstm_model(
    input_shape=(X_train.shape[1], X_train.shape[2])
)

lstm_model.load_weights(WEIGHTS_PATH)

# ------------------------------------------------------------
# LSTM PREDICTION
# ------------------------------------------------------------
pred_scaled = lstm_model.predict(X_test, verbose=0)

close_scaler = MinMaxScaler()
close_scaler.min_ = scaler.min_[3:4]
close_scaler.scale_ = scaler.scale_[3:4]

predictions = close_scaler.inverse_transform(pred_scaled)
actual = close_scaler.inverse_transform(y_test.reshape(-1, 1))

# ------------------------------------------------------------
# PLOT ACTUAL VS PREDICTED
# ------------------------------------------------------------
st.subheader("📈 Actual vs Predicted Close Price")

fig = go.Figure()
fig.add_trace(go.Scatter(y=actual.flatten(), mode="lines", name="Actual"))
fig.add_trace(go.Scatter(y=predictions.flatten(), mode="lines", name="Predicted"))

fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# LOAD SENTIMENT MODEL
# ------------------------------------------------------------
sentiment_model = tf.keras.models.load_model(
    "sentiment_model.h5",
    compile=False
)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("config.pkl", "rb") as f:
    config = pickle.load(f)

max_len = config["max_len"]

# ------------------------------------------------------------
# SENTIMENT ANALYSIS
# ------------------------------------------------------------
st.subheader("🧠 Sentiment Analysis Result")

if sentiment_ready:
    seq = tokenizer.texts_to_sequences([news_input])
    padded = pad_sequences(seq, maxlen=max_len, padding="post")

    prob = sentiment_model.predict(padded, verbose=0)[0][0]
    score = prob * 100

    if prob > 0.6:
        label, color = "Positive 😊", "green"
    elif prob < 0.4:
        label, color = "Negative 😟", "red"
    else:
        label, color = "Neutral 😐", "orange"

    st.markdown(
        f"<h3 style='color:{color}'>{label}</h3>",
        unsafe_allow_html=True
    )
    st.write(f"Sentiment Score: {score:.2f}%")
else:
    st.warning("Enter text or upload an image for sentiment analysis.")

# ------------------------------------------------------------
# HYBRID SIGNAL
# ------------------------------------------------------------
st.subheader("📌 Hybrid Trading Signal")

last_actual = actual[-1][0]
last_pred = predictions[-1][0]

price_change = (last_pred - last_actual) / last_actual * 100
combined_score = (0.7 * price_change) + (0.3 * (score / 100))

if combined_score > 0.2:
    signal, sig_color = "BUY", "green"
elif combined_score < -0.2:
    signal, sig_color = "SELL", "red"
else:
    signal, sig_color = "HOLD", "orange"

st.markdown(
    f"<h2 style='color:{sig_color}; text-align:center'>{signal}</h2>",
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# CONFIDENCE SCORE
# ------------------------------------------------------------
errors = np.abs(predictions - actual) / actual
confidence = max(0, min((1 - np.mean(errors)) * 100, 100))

st.subheader("🎯 Model Confidence Score")
st.progress(int(confidence))
st.write(f"Confidence: {confidence:.2f}%")

# ------------------------------------------------------------
# 7-DAY FORECAST
# ------------------------------------------------------------
st.subheader("🔮 7-Day Forecast")

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

st.line_chart(future)

vol = np.std(future)
mean = np.mean(future)
forecast_conf = max(0, min((1 - (vol / mean)) * 100, 100))

st.subheader("🔎 Forecast Confidence")
st.progress(int(forecast_conf))
st.write(f"Forecast Confidence: {forecast_conf:.2f}%")
