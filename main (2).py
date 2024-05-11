

from Black-Scholes import calculate_duration_and_convexity, calculate_convexity, calculate_duration, 	        calculate_option_prices, black_scholes, load_and_clean_data

def main(options_data, bonds_data, confidence_level=0.05):
    options_data['Theoretical Price'] = options_data['Theoretical Price'].astype(float)
    options_portfolio_value = options_data['Theoretical Price'].sum()
    options_portfolio_stddev = options_data['Theoretical Price'].std()
    options_portfolio_var = norm.ppf(confidence_level) * options_portfolio_stddev


if __name__ == "__main__":
    main()



