#price: Perform risk analysis calculations for a portfolio of option contracts and fixed-income securities 
#○ For option contracts, calculate the theoretical price using the Black-Scholes model. 
#○ For fixed-income securities, calculate the duration and convexity. 
#○ If time permits, calculate Value at Risk (VaR) for the options and bonds portfolios.

import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, date
import warnings


def load_and_clean_data(bonds_file, options_file):
    # Load data
    bonds_data = pd.read_csv(bonds_file)
    options_data = pd.read_csv(options_file)


    # Handle missing values and data types
    bonds_data = bonds_data.dropna()
    bonds_data['Maturity Date'] = pd.to_datetime(bonds_data['Maturity Date']).dt.date

    options_data = options_data.dropna()
    options_data['Expiration Date'] = pd.to_datetime(options_data['Expiration Date']).dt.date

    return bonds_data, options_data


def black_scholes(S, K, T, r, sigma, option_type):
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'Call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def calculate_option_prices(options_data):
    r = 0.05  # Assume risk-free rate of 5%
    for idx, row in options_data.iterrows():
        S = 100  # Assume underlying asset price of 100
        K = row['Strike Price']
        T = (row['Expiration Date'] - date.today()).days / 365
        sigma = row['Implied Volatility']
        option_type = row['Contract']
        theoretical_price = black_scholes(S, K, T, r, sigma, option_type)
        options_data.at[idx, 'Theoretical Price'] = theoretical_price
    return options_data

def calculate_duration(face_value, coupon_rate, maturity_date, yield_to_maturity):
    periods = (maturity_date - date.today()).days / 365
    discount_factor = (1 + yield_to_maturity) ** -periods
    duration = sum([(t * coupon_rate * discount_factor) / (1 + yield_to_maturity) ** t for t in range(1, int(periods) + 1)]) + \
               (periods * face_value * discount_factor) / (1 + yield_to_maturity) ** periods
    return duration

def calculate_convexity(face_value, coupon_rate, maturity_date, yield_to_maturity):
    periods = (maturity_date - date.today()).days / 365
    discount_factor = (1 + yield_to_maturity) ** -periods
    convexity = sum([(t * (t + 1) * coupon_rate * discount_factor) / (1 + yield_to_maturity) ** (t + 2) for t in range(1, int(periods) + 1)]) + \
                (periods * (periods + 1) * face_value * discount_factor) / (1 + yield_to_maturity) ** (periods + 2)
    return convexity

def calculate_duration_and_convexity(bonds_data):
    # Calculate duration and convexity for bonds
    bonds_data['Duration'] = bonds_data.apply(lambda row: calculate_duration(row['Face Value'], row['Coupon Rate'], row['Maturity Date'], row['Yield to Maturity']), axis=1)
    bonds_data['Convexity'] = bonds_data.apply(lambda row: calculate_convexity(row['Face Value'], row['Coupon Rate'], row['Maturity Date'], row['Yield to Maturity']), axis=1)

    return bonds_data

def calculate_var(options_data, bonds_data, confidence_level=0.05):
    # Calculate VaR for options portfolio
    options_data['Theoretical Price'] = options_data['Theoretical Price'].astype(float)
    options_portfolio_value = options_data['Theoretical Price'].sum()
    options_portfolio_stddev = options_data['Theoretical Price'].std()
    options_portfolio_var = norm.ppf(confidence_level) * options_portfolio_stddev

    # Calculate VaR for bonds portfolio
    bonds_portfolio_value = bonds_data['Face Value'].sum()
    bonds_portfolio_stddev = bonds_data['Face Value'].std()
    bonds_portfolio_var = norm.ppf(confidence_level) * bonds_portfolio_stddev

    return options_portfolio_var, bonds_portfolio_var



#calculate
bonds_data, options_data = load_and_clean_data('bonds_data.csv', 'options_data.csv')
options_data = calculate_option_prices(options_data)
bonds_data = calculate_duration_and_convexity(bonds_data)
options_var, bonds_var = calculate_var(options_data, bonds_data)



