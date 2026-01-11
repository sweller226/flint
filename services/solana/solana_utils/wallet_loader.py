import os
import base58
from dotenv import load_dotenv
from solders.keypair import Keypair

load_dotenv()

def get_audit_wallet():
    private_key_str = os.getenv("SOLANA_PRIVATE_KEY")
    if not private_key_str:
        raise ValueError("Missing SOLANA_PRIVATE_KEY in .env")
    secret_key = base58.b58decode(private_key_str)
    return Keypair.from_bytes(secret_key)