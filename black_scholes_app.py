

import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price

def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price

st.title("📈 Black-Scholes Option Pricing Calculator")

# User inputs
S = st.number_input("Stock Price (S)", value=100.0)
K = st.number_input("Strike Price (K)", value=100.0)
T = st.number_input("Time to Expiration (T in years)", value=1.0)
r = st.number_input("Risk-Free Rate (r)", value=0.05)
sigma = st.number_input("Volatility (σ)", value=0.2)
option_type = st.selectbox("Option Type", ("call", "put"))

# Calculate option price
price = black_scholes(S, K, T, r, sigma, option_type)

st.markdown(f"### 💰 {option_type.capitalize()} Option Price: **${price:.2f}**")

if st.checkbox("📊 Show Option Price vs Stock Price Chart"):
    stock_prices = np.linspace(0.5 * K, 1.5 * K, 100)
    option_prices = [black_scholes(s, K, T, r, sigma, option_type) for s in stock_prices]

    fig, ax = plt.subplots()
    ax.plot(stock_prices, option_prices, label=f"{option_type.capitalize()} Option Value")
    ax.set_xlabel("Stock Price")
    ax.set_ylabel("Option Price")
    ax.set_title("Option Price vs Stock Price")
    ax.legend()
    st.pyplot(fig)
