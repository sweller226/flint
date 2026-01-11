import React from 'react';
import { motion } from 'framer-motion';

interface AccountState {
    balance: number;
    equity: number;
    pnl_unrealized: number;
    pnl_realized: number;
    position_size: number;
    entry_price: number;
}

interface PortfolioWidgetProps {
    account: AccountState | null;
}

export const PortfolioWidget: React.FC<PortfolioWidgetProps> = ({ account }) => {
    if (!account) return (
        <div className="p-4 rounded-xl bg-white/5 border border-white/10 animate-pulse">
            <div className="h-4 bg-white/10 rounded w-1/2 mb-2"></div>
            <div className="h-8 bg-white/10 rounded w-3/4"></div>
        </div>
    );

    const isProfit = account.pnl_unrealized >= 0;
    const pnlColor = isProfit ? 'text-emerald-400' : 'text-rose-400';
    const pnlSign = isProfit ? '+' : '';

    return (
        <div className="p-4 rounded-xl bg-white/5 border border-white/10 backdrop-blur-sm mb-4">
            <div className="flex justify-between items-start mb-2">
                <span className="text-xs uppercase tracking-wider text-white/50 font-medium">Demo Portfolio</span>
                <span className={`text-xs px-2 py-0.5 rounded-full ${account.position_size !== 0 ? 'bg-emerald-500/20 text-emerald-400' : 'bg-white/10 text-white/50'}`}>
                    {account.position_size !== 0 ? `LONG ${account.position_size} ES` : 'FLAT'}
                </span>
            </div>

            <div className="flex flex-col gap-1">
                <span className="text-3xl font-light text-white tracking-tight">
                    ${account.equity.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </span>

                <div className="flex items-center gap-2 text-sm">
                    <span className="text-white/40">PnL (Open)</span>
                    <span className={`font-mono ${pnlColor}`}>
                        {pnlSign}{account.pnl_unrealized.toLocaleString('en-US', { style: 'currency', currency: 'USD' })}
                    </span>
                </div>
                <div className="flex items-center gap-2 text-xs mt-1">
                    <span className="text-white/40">Cash</span>
                    <span className="font-mono text-white/70">
                        ${account.balance.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </span>
                </div>
            </div>
        </div>
    );
};
