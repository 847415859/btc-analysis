from backtest_engine import DataLoader, BacktestEngine, AdvancedStrategy
import pandas as pd

def test_run():
    print("Running test backtest...")
    loader = DataLoader()
    # Fetch a very short period for testing
    start_str = "2024-01-01"
    end_str = "2024-02-01"
    
    print("Fetching Klines...")
    df_kline = loader.fetch_klines("BTCUSDT", "1h", start_str, end_str)
    print(f"Klines: {len(df_kline)}")
    
    print("Fetching Funding...")
    df_funding = loader.fetch_funding_rates("BTCUSDT", start_str, end_str)
    print(f"Funding: {len(df_funding)}")
    
    print("Merging...")
    df = loader.merge_data(df_kline, df_funding)
    print(f"Merged: {len(df)} rows")
    
    engine = BacktestEngine(initial_capital=10000)
    strategy = AdvancedStrategy()
    
    metrics = engine.run(df, strategy)
    print("Metrics:", metrics)

if __name__ == "__main__":
    test_run()
