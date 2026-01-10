import os
import json
import base58
from dotenv import load_dotenv
from solana.rpc.api import Client
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.instruction import Instruction
from solders.message import MessageV0
from solders.transaction import VersionedTransaction

from services.backend.app.wallet_loader import get_audit_wallet

# Load environment variables from .env
load_dotenv()

# Configuration
RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.devnet.solana.com")
MEMO_PROGRAM_ID = Pubkey.from_string("MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr")

def get_audit_wallet():
    """Step 1: Verify Wallet Loading"""
    private_key_str = os.getenv("SOLANA_PRIVATE_KEY")
    if not private_key_str:
        raise ValueError("SOLANA_PRIVATE_KEY not found in .env file")
    
    secret_key = base58.b58decode(private_key_str)
    return Keypair.from_bytes(secret_key)

def test_audit_log():
    """Step 2: Verify On-Chain Recording"""
    client = Client(RPC_URL)
    wallet = get_audit_wallet()
    
    print(f"Testing with Wallet Address: {wallet.pubkey()}")
    
    # Check balance first to ensure you have gas money
    balance = client.get_balance(wallet.pubkey()).value
    print(f"Current Balance: {balance / 10**9} SOL")
    
    if balance == 0:
        print("❌ Error: Wallet has 0 SOL. Run 'solana airdrop 2' first.")
        return

    # Sample trade data from the Flint Execution Plan
    sample_trade = {
        "type": "SHORT",
        "entry": 9321.5,
        "reason": "ICT: Bounce off buy-side liquidity",
        "timestamp": "2025-01-10T14:51:00"
    }
    
    memo_data = json.dumps(sample_trade).encode("utf-8")
    
    # Build Instruction
    memo_ix = Instruction(
        program_id=MEMO_PROGRAM_ID,
        data=memo_data,
        accounts=[]
    )
    
    # Build and Sign Transaction
    recent_blockhash = client.get_latest_blockhash().value.blockhash
    message = MessageV0.try_compile(
        payer=wallet.pubkey(),
        instructions=[memo_ix],
        address_lookup_table_accounts=[],
        recent_blockhash=recent_blockhash
    )
    transaction = VersionedTransaction(message, [wallet])
    
    # Send
    print("Sending trade record to Solana...")
    response = client.send_transaction(transaction)
    tx_sig = response.value
    
    print(f"✅ Success! Transaction Signature: {tx_sig}")
    print(f"View on Explorer: https://explorer.solana.com/tx/{tx_sig}?cluster=devnet")

if __name__ == "__main__":
    test_audit_log()