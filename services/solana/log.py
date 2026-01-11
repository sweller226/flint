import os
import json
from solana.rpc.api import Client
from solders.pubkey import Pubkey
from solders.instruction import Instruction
from solders.message import MessageV0
from solders.transaction import VersionedTransaction
from .wallet_loader import get_audit_wallet

# The fixed Program ID for Solana's native Memo service
MEMO_PROGRAM_ID = Pubkey.from_string("MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr")

def log_trade_to_solana(trade_details: dict):
    client = Client(os.getenv("SOLANA_RPC_URL"))
    wallet = get_audit_wallet()
    
    # 1. Prepare data (Serialize trade dict to bytes)
    memo_data = json.dumps(trade_details).encode("utf-8")
    
    # 2. Create the Memo Instruction
    memo_ix = Instruction(
        program_id=MEMO_PROGRAM_ID,
        data=memo_data,
        accounts=[] # Memo program doesn't require specific accounts
    )
    
    # 3. Build the Transaction
    recent_blockhash = client.get_latest_blockhash().value.blockhash
    message = MessageV0.try_compile(
        payer=wallet.pubkey(),
        instructions=[memo_ix],
        address_lookup_table_accounts=[],
        recent_blockhash=recent_blockhash
    )
    
    transaction = VersionedTransaction(message, [wallet])
    
    # 4. Send and return the Signature
    response = client.send_transaction(transaction)
    return response.value  # This is the Tx Hash (e.g., "5HzW...")