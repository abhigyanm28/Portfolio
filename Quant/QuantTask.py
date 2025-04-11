import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

# Load historical data
file_path = 'clean_ticker_data23.xlsx'
data = pd.read_excel(file_path, sheet_name='clean_ticker_data2', index_col='Date')

# Calculate daily returns
daily_returns = data.pct_change().dropna()

# Parameter values from screenshot
parameters = pd.DataFrame({
    'Ticker': ['AAPL', 'ADBE', 'COST', 'GOOGL', 'HD', 'JNJ', 'MSFT', 'PG', 'UNH', 'V'],
    'Sector': ['Technology', 'Technology', 'Consumer Defensive', 'Communication Services', 
               'Consumer Cyclical', 'Healthcare', 'Technology', 'Consumer Defensive', 
               'Healthcare', 'Financial'],
    'PE': [29.95, 22.99, 53.5, 18.1, 23.73, 26.43, 28.98, 26.09, 33.79, 31.99],
    'PFCF': [28.79, 16.19, 58.72, 24.55, 21.55, 18.61, 38.2, 22.97, 23.2, 29.99],
    'Short Float': [0.86, 1.93, 1.55, 1.24, 0.96, 1.03, 0.88, 0.77, 1.12, 1.58],
    'Beta': [1.28, 1.54, 0.99, 1.01, 1.05, 0.41, 0.98, 0.4, 0.52, 0.95],
    'Volatility': [2.96, 2.96, 2.98, 2.77, 2.47, 1.75, 2.18, 1.99, 2.27, 2.47],
    'RSI': [25.18, 24.83, 40.64, 28.03, 39.65, 38.75, 29.81, 40.9, 58.4, 30.43],
    'Volume': [125402855, 5859627, 5156578, 62026676, 7515034, 16593950, 48680290, 
               13446145, 9912509, 13191967]
})

# Add ESG risk scores
esg_risk = {
    "MSFT": 13.6, "HD": 12.6, "AAPL": 18.8, "UNH": 16.6,
    "ADBE": 14.4, "GOOGL": 24.9, "COST": 29.1, "JNJ": 19.9,
    "V": 14.9, "PG": 24.6
}
parameters['ESG Risk'] = parameters['Ticker'].map(esg_risk)

# Calculate normalized and derived metrics
parameters["Normalized Volatility"] = 1 / parameters["Volatility"]
parameters["Normalized Beta"] = 1 / (1 + abs(parameters["Beta"] - 1))
parameters["Value Score"] = (1/parameters["PE"] + 1/parameters["PFCF"])/2
parameters["RSI Score"] = 1 - abs(parameters["RSI"] - 45)/45  # Ideal RSI near 45
parameters["ESG Score"] = 1 - parameters["ESG Risk"]/30  # Normalize ESG risk

# Create a multi-factor composite score
parameters["Composite Score"] = (
    0.20 * parameters["Normalized Volatility"] + 
    0.25 * parameters["Value Score"] + 
    0.10 * parameters["Normalized Beta"] + 
    0.15 * parameters["RSI Score"] +
    0.30 * parameters["ESG Score"]
)

# Define portfolio optimization function
def portfolio_optimization(returns_data, param_data, risk_free_rate=0.03):
    # Ensure we're working with the tickers in the right order
    tickers = param_data['Ticker'].tolist()
    
    # Extract returns for the assets in our parameter list
    returns_subset = returns_data[tickers]
    
    n_assets = len(tickers)  # Number of assets
    mean_returns = returns_subset.mean() * 252  # Annualized returns
    cov_matrix = returns_subset.cov() * 252  # Annualized covariance
    
    # Initial weights based on composite scores
    initial_weights = param_data["Composite Score"].values / param_data["Composite Score"].sum()
    
    # Objective function: maximize Sharpe ratio
    def objective(weights):
        portfolio_return = np.sum(weights * mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio
    
    # Constraints: weights sum to 1, individual weight limits
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: np.mean(x) * 10 - np.max(x)},  # Diversity constraint
    ]
    
    # Bounds: min and max weights for diversification
    bounds = tuple((0.02, 0.50) for _ in range(n_assets))
    
    # Optimization
    result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints, method='SLSQP')
    
    # Calculate optimized portfolio metrics
    opt_weights = result['x']
    portfolio_return = np.sum(opt_weights * mean_returns)
    portfolio_volatility = np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix, opt_weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    return {
        'weights': opt_weights,
        'return': portfolio_return,
        'volatility': portfolio_volatility,
        'sharpe_ratio': sharpe_ratio,
        'tickers': tickers
    }

# Backtest function with transaction costs
def backtest_portfolio(returns_data, weights, tickers, initial_investment=10000, rebalance_period=63, transaction_cost=0.001):
    """
    Backtest portfolio performance
    
    Parameters:
    returns_data: DataFrame of historical returns
    weights: Array of portfolio weights
    tickers: List of ticker symbols in the same order as weights
    initial_investment: Starting investment amount (default $10,000)
    rebalance_period: Trading days between rebalancing (63 days = quarterly)
    transaction_cost: Cost per trade as a percentage (default 0.1%)
    """
    # Extract returns for the assets in our ticker list
    returns_subset = returns_data[tickers]
    
    portfolio_values = []
    current_weights = weights.copy()
    current_value = initial_investment
    
    for i in range(len(returns_subset)):
        if i > 0:
            # Get returns for the current day
            daily_return = np.sum(returns_subset.iloc[i] * current_weights)
            
            # Update portfolio value
            current_value *= (1 + daily_return)
            
            # Update weights due to price changes
            new_weights = current_weights * (1 + returns_subset.iloc[i].values)
            current_weights = new_weights / np.sum(new_weights)
            
            # Rebalance if needed
            if i % rebalance_period == 0:
                # Calculate transaction costs for rebalancing
                # Cost is based on the absolute change in weights times the transaction cost rate
                weight_changes = np.abs(current_weights - weights)
                total_cost = np.sum(weight_changes) * transaction_cost * current_value
                current_value -= total_cost
                
                # Reset to target weights
                current_weights = weights.copy()
        
        portfolio_values.append(current_value)
    
    return pd.Series(portfolio_values, index=returns_subset.index)

# Run optimization
optimization_result = portfolio_optimization(daily_returns, parameters)

# Create a DataFrame to display the allocation results
allocation = pd.DataFrame({
    'Ticker': parameters['Ticker'],
    'Sector': parameters['Sector'],
    'Weight': optimization_result['weights'] * 100  # Convert to percentage
})

# Calculate individual asset contribution to return
tickers = optimization_result['tickers']
asset_returns = daily_returns[tickers].mean() * 252
allocation['Contribution'] = optimization_result['weights'] * asset_returns.values
allocation = allocation.sort_values('Weight', ascending=False)

# Display portfolio metrics

print(f"\nWeight Allocation:")
print(allocation[['Ticker', 'Sector', 'Weight']].round(2))

# Function to create visualizations
def create_visualization(metric='Weight'):
    plt.figure(figsize=(12, 8))
    
    # Create a dropdown-like visualization with multiple subplots
    if metric == 'All':
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Weight allocation
        sns.barplot(x='Ticker', y='Weight', data=allocation, ax=axes[0], palette='viridis')
        axes[0].set_title('Portfolio Weight Allocation (%)')
        axes[0].set_ylabel('Weight (%)')
        axes[0].tick_params(axis='x', rotation=45)
        
        # ESG Risk scores
        esg_data = parameters.sort_values('ESG Risk')
        sns.barplot(x='Ticker', y='ESG Risk', data=esg_data, ax=axes[1], palette='RdYlGn_r')
        axes[1].set_title('ESG Risk Scores (Lower is Better)')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Volatility
        vol_data = parameters.sort_values('Volatility')
        sns.barplot(x='Ticker', y='Volatility', data=vol_data, ax=axes[2], palette='coolwarm')
        axes[2].set_title('Monthly Volatility (%)')
        axes[2].tick_params(axis='x', rotation=45)
        
        # Sector allocation
        sector_weights = allocation.groupby('Sector')['Weight'].sum().reset_index()
        sns.barplot(x='Sector', y='Weight', data=sector_weights, ax=axes[3], palette='Spectral')
        axes[3].set_title('Sector Weight Allocation (%)')
        axes[3].set_ylabel('Total Weight (%)')
        axes[3].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.suptitle('Portfolio Analysis Dashboard', fontsize=16, y=1.02)
        
    else:
        # Single metric visualization
        if metric == 'Weight':
            sns.barplot(x='Ticker', y='Weight', data=allocation, palette='viridis')
            plt.title('Portfolio Weight Allocation (%)')
            plt.ylabel('Weight (%)')
        elif metric == 'ESG Risk':
            esg_data = parameters.sort_values('ESG Risk')
            sns.barplot(x='Ticker', y='ESG Risk', data=esg_data, palette='RdYlGn_r')
            plt.title('ESG Risk Scores (Lower is Better)')
        elif metric == 'Volatility':
            sns.barplot(x='Ticker', y='Volatility', data=parameters, palette='coolwarm')
            plt.title('Monthly Volatility (%)')
        elif metric == 'Sector':
            sector_weights = allocation.groupby('Sector')['Weight'].sum().reset_index()
            sns.barplot(x='Sector', y='Weight', data=sector_weights, palette='Spectral')
            plt.title('Sector Weight Allocation (%)')
            plt.ylabel('Total Weight (%)')
            
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    plt.show()

# ESG impact visualization
def visualize_esg_impact():
    """Create a visualization showing the ESG impact of our portfolio"""
    # Calculate ESG weighted average
    weighted_esg = np.sum(optimization_result['weights'] * parameters['ESG Risk'].values)
    
    # Create comparison with market averages
    esg_comparison = pd.DataFrame({
        'Portfolio Type': ['Our Portfolio', 'Average S&P 500', 'High ESG Focus', 'No ESG Focus'],
        'ESG Risk Score': [weighted_esg, 22.2, 15.5, 28.7]  # Example values
    })
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Portfolio Type', y='ESG Risk Score', data=esg_comparison, palette='RdYlGn_r')
    plt.title('ESG Risk Comparison (Lower is Better)')
    plt.ylabel('ESG Risk Score')
    plt.axhline(y=20, color='gray', linestyle='--', alpha=0.8)  # Benchmark line
    plt.text(3.5, 20.5, 'Industry Average', fontsize=10)
    plt.tight_layout()
    plt.show()
    
    # Contribution to ESG score by sector
    esg_by_sector = pd.DataFrame({
        'Ticker': parameters['Ticker'],
        'Sector': parameters['Sector'],
        'ESG Risk': parameters['ESG Risk'],
        'Weight': optimization_result['weights'] * 100,
        'Contribution': optimization_result['weights'] * parameters['ESG Risk']
    })
    
    sector_esg = esg_by_sector.groupby('Sector')['Contribution'].sum().reset_index()
    sector_esg['Percentage'] = sector_esg['Contribution'] / sector_esg['Contribution'].sum() * 100
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Sector', y='Percentage', data=sector_esg, palette='viridis')
    plt.title('Sector Contribution to Portfolio ESG Risk')
    plt.ylabel('Contribution (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Create visualizations
create_visualization('All')
visualize_esg_impact()

# Run backtest
backtest_results = backtest_portfolio(
    daily_returns, 
    optimization_result['weights'], 
    optimization_result['tickers']
)

# Calculate performance metrics
total_return = (backtest_results.iloc[-1] / backtest_results.iloc[0]) - 1
annual_return = (1 + total_return) ** (252 / len(backtest_results)) - 1
volatility = backtest_results.pct_change().std() * np.sqrt(252)
sharpe_ratio = (annual_return - 0.03) / volatility
max_drawdown = (backtest_results / backtest_results.expanding().max() - 1).min()

# Display backtest results
print("\nBacktest Results:")
print(f"Total Return: {total_return:.4f} ({total_return*100:.2f}%)")
print(f"Annualized Return: {annual_return:.4f} ({annual_return*100:.2f}%)")
print(f"Annualized Volatility: {volatility:.4f} ({volatility*100:.2f}%)")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Maximum Drawdown: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)")

# Plot portfolio performance
plt.figure(figsize=(12, 6))
backtest_results.plot()
plt.title('Portfolio Backtest Performance')
plt.ylabel('Portfolio Value ($)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Compare with equal-weighted portfolio
equal_weights = np.ones(len(optimization_result['tickers'])) / len(optimization_result['tickers'])
equal_weighted_backtest = backtest_portfolio(
    daily_returns, 
    equal_weights, 
    optimization_result['tickers']
)

# Plot comparison
plt.figure(figsize=(12, 6))
backtest_results.plot(label='Optimized Portfolio')
equal_weighted_backtest.plot(label='Equal-Weighted Portfolio')
plt.title('Portfolio Comparison: Optimized vs Equal-Weighted')
plt.ylabel('Portfolio Value ($)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
