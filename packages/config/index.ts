// Shared Constants for Flint
// This file is used by both the Desktop app and the Web app (and potentially others)

export const API_BASE_URL = "http://localhost:8000";
export const WS_BASE_URL = "ws://localhost:8000/ws";

export const BRAND_COLORS = {
  background: "#001219",
  panel: "#10141C",
  buy: "#94D2BD",
  sell: "#FF6B6B",
  warning: "#E9D8A6",
  gradients: {
    primary: "linear-gradient(to right, #00FFA3, #03E1FF, #DC1FFF)"
  }
};

export const TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "D"];

export const SOLANA_DEVNET_RPC = "https://api.devnet.solana.com";
export const FLINT_PROGRAM_ID = "FlintDemoProgramID1111111111111111111111111";
