import requests
import json
import sys

def check_backend():
    try:
        resp = requests.get("http://localhost:8000/", timeout=10)
        print(f"Root status: {resp.status_code}")
        print(f"Root response: {resp.json()}")
        
        # Check Candles
        print("\nChecking candles endpoint...")
        resp = requests.get("http://localhost:8000/api/candles?limit=5", timeout=5)
        print(f"Candles status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"Candle count: {data.get('count')}")
            if data.get('candles'):
                print(f"First candle: {data['candles'][0]}")
            else:
                print("No candles returned")
        else:
            print(f"Candles error: {resp.text}")

    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_backend()
