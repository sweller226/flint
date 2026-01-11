import os
import sys
import pandas as pd
from solana_utils.log import log_trade_to_solana


current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


def process_and_audit(actions_path, forecast_path):
    # Load CSVs
    if not os.path.exists(actions_path) or not os.path.exists(forecast_path):
        print(f"❌ Error: Files not found at specified paths.")
        return

    actions_df = pd.read_csv(actions_path)
    forecast_df = pd.read_csv(forecast_path)
    
    # Filter for executed trades 
    trades = actions_df[actions_df['executed_trade'] == True].copy()
    
    if trades.empty:
        print("No executed trades found in the action logs.")
        return

    print(f"Found {len(trades)} executed trades. Syncing to Solana...")

    for _, trade in trades.iterrows():
        timestamp = trade['ts_event']
        # Find the matching timestamp in the forecast file 
        price_info = forecast_df[forecast_df['ts_event'] == timestamp]
        
        audit_packet = {
            "ts": timestamp,
            "action": trade['action'],
            "price": float(trade['price']),
            "pnl_net": float(trade['pnl_net']),
            "trade_count": int(trade['trade_count']),
            "contract": price_info['contract_month'].values[0] if not price_info.empty else "N/A"
        }
        
        try:
            signature = log_trade_to_solana(audit_packet)
            
            # Construct the Explorer Link
            explorer_link = f"https://explorer.solana.com/tx/{signature}?cluster=devnet"
            
            print(f"✅ Logged Trade {audit_packet['trade_count']} | Action: {audit_packet['action']}")
            print(f"🔗 View Entry: {explorer_link}") # This returns the clickable link
            
        except Exception as e:
            print(f"❌ Failed to log trade at {timestamp}: {e}")

if __name__ == "__main__":
    # Corrected paths based on your latest file direction 
    FORECAST_FILE = "/Users/elliottchan/flint/services/backend/model/forecaster/artifacts/forecasts/forecast_H_2025-09-15T07-00-00Z.csv"
    ACTIONS_FILE = "/Users/elliottchan/flint/services/backend/model/run_agent/artifacts/agent_trades/actions_H_2025-09-15T07-01-00Z.csv"
    
    process_and_audit(ACTIONS_FILE, FORECAST_FILE)