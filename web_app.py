import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf

st.title("Stock Price Predictor App")

stock = st.text_input("Enter the Stock ID", "GOOG")


from datetime import datetime
end = datetime.now()
start = datetime(end.year-20,end.month,end.day)

google_data = yf.download(stock, start, end)
model = load_model("Latest_stock_price_model.keras")
st.subheader("Stock Data")
st.write(google_data)

splitting_len = int(len(google_data)*0.7)
x_test = pd.DataFrame(google_data.Close[splitting_len:])

def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'orange')
    plt.plot(full_data.Close, 'blue')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

st.subheader('Oringinal Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'],google_data))

st.subheader('Oringinal Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_200_days'],google_data))

st.subheader('Oringinal Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data))

st.subheader('Oringinal Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,1,google_data['MA_for_250_days']))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test['Open'])

x_test = []
y_test = []

for i in range(100, len(scaled_data)):
    x_test.append(scaled_data[i-100:i])
    y_test.append(scaled_data[i])

x_test, y_test = np.array(x_test), np.array(y_test)

predictions = model.predict(x_test)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_test)

ploting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.rershape(-1),
        'predictions': inv_pre.reshape(1)

    }, 
       index= google_data.index[splitting_len+100:]

)

st.subheader("oringinal value and predicted value")
st.write(ploting_data)


st.subheader("oringinal close price  and predicted close price")
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([google_data.Open[:splitting_len+100],ploting_data], axis=0))
plt.legend(["Data- not used","Original test data", "predicted test data"])
st.pyplot(fig)