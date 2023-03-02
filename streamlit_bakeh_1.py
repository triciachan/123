# numerical and statistical utilities
import numpy as np

# visualization requirements
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as ex

# data utitilies
import yfinance as yf
import pandas as pd
import pandas_ta
import datetime as dt

# streamlit artiface 
import streamlit as st

# utility for technical indexes
import talib

# Backtest
from backtesting import Backtest, Strategy

# Observe recent changes
start = dt.datetime(2019, 1, 1).strftime('%Y-%m-%d')
end =  dt.date.today()
#end='2023-2-23'

# set ticker's symbol in yahoo stock
ticker='2330.TW'

# Downloading data

df = yf.download(ticker, start = start, end = end)

# More technic indexes

df['stoch_k'] = pandas_ta.stochrsi(close=df['Adj Close'],length=20).iloc[:,0]
df['stoch_d'] = pandas_ta.stochrsi(close=df['Adj Close'],length=20).iloc[:,1]
df['bb_lower'] = pandas_ta.bbands(close=df['Adj Close'],length=20).iloc[:,0]
df['bb_upper'] = pandas_ta.bbands(close=df['Adj Close'],length=20).iloc[:,2]
df['forward_1d'] = df['Adj Close'].pct_change(1).shift(-1)

#fig=ex.line(df, x=df.index, y=['Adj Close','bb_lower','bb_upper'], title='台積電 (2330.TW) Adj Close with Bollinger Bands')
#fig.update_layout(title_text=f'台積電 (2330.TW) Adj Close with Bollinger Bands', title_x=0.5);

# Plot!
#st.plotly_chart(fig, use_container_width=True)




# compute simple moving average using talib
sma1 = talib.SMA(df['Adj Close'], timeperiod=20)
sma2 = talib.SMA(df['Adj Close'], timeperiod=60)


class Strategy(Strategy):
    
    n1 = 20
    n2 = 60
    
    def init(self):
        super().init()
        
        # Precompute the two moving averages
        close = pd.Series(self.data.Close)
        sma1 = talib.SMA(close, timeperiod=self.n1)
        sma2 = talib.SMA(close, timeperiod=self.n2)

        # Precompute signal
        signal_long = (sma1 > sma2) & (sma1.shift() < sma2.shift())
        signal_short = (sma1 < sma2) & (sma1.shift() > sma2.shift())

        # combine signal
        signal = signal_long
        signal[signal_short] = -1
        
        # plot sma
        self.I(lambda x: sma1, 'sma1')
        self.I(lambda x: sma2, 'sma2')

        # set signal to trade
        self.signal = self.I(lambda x: signal, 'signal')

    def next(self):
        super().next()

        entry_size = self.signal[-1]

        if entry_size > 0:
            self.buy()
        elif entry_size < 0:
          for trade in self.trades:
              trade.close()

bt = Backtest(df, Strategy)
result1 = bt.run()

st.bokeh_chart(bt.plot(), use_container_width=True)


