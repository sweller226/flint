import os
import json
from solana.rpc.api import Client
from solders.pubkey import Pubkey
from solders.instruction import Instruction
from solders.message import MessageV0
from solders.transaction import VersionedTransaction
from .wallet_loader import get_audit_wallet

MEMO_PROGRAM_ID = Pubkey.from_string("MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr")

def log_trade_to_solana(trade_details: dict):
    # Use Devnet by default if URL isn't in env
    rpc_url = os.getenv("SOLANA_RPC_URL", "https://api.devnet.solana.com")
    client = Client(rpc_url)
    wallet = get_audit_wallet()
    
    memo_data = json.dumps(trade_details).encode("utf-8")
    
    memo_ix = Instruction(
        program_id=MEMO_PROGRAM_ID,
        data=memo_data,
        accounts=[]
    )
    
    recent_blockhash = client.get_latest_blockhash().value.blockhash
    message = MessageV0.try_compile(
        payer=wallet.pubkey(),
        instructions=[memo_ix],
        address_lookup_table_accounts=[],
        recent_blockhash=recent_blockhash
    )
    
    transaction = VersionedTransaction(message, [wallet])
    response = client.send_transaction(transaction)
    return response.value