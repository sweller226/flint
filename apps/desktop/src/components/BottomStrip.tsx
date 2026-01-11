import React from "react";

import { useSimulation } from "../context/SimulationContext";

export const BottomStrip = () => {
    const { portfolio } = useSimulation();

    // Use Stats from Context
    const winRate = portfolio.winRate || 68; // Mock default if 0
    const trades = portfolio.totalTrades || 42;
    const maxDD = portfolio.maxDrawdown || 0;
    const roi = portfolio.roi || 0;

    return (
        <footer className="h-12 bg-flint-panel border-t border-flint-border px-4 py-2 text-[11px] flex items-center justify-between shrink-0">
            <div className="flex flex-col sm:flex-row gap-1 sm:gap-4 items-center">
                <div className="flex items-center gap-2">
                    <span className="text-slate-400 font-medium">Session PnL</span>
                    <span className={`font-black tracking-wide ${portfolio.equity >= 100000 ? 'text-flint-positive' : 'text-flint-negative'}`}>
                        ${(portfolio.equity - 100000).toLocaleString('en-US', { minimumFractionDigits: 2 })}
                    </span>
                    <span className={`text-[10px] px-1.5 py-0.5 rounded ${roi >= 0 ? 'bg-flint-positive/20 text-flint-positive' : 'bg-flint-negative/20 text-flint-negative'}`}>
                        {roi.toFixed(1)}% Ann.
                    </span>
                </div>
                <div className="flex gap-4 text-slate-500 border-l border-slate-800 pl-4">
                    <span>Win rate <span className="text-slate-300 font-bold">{winRate}%</span></span>
                    <span>Max DD <span className="text-flint-negative font-bold">-${Math.abs(maxDD)}</span></span>
                    <span>Trades <span className="text-slate-300 font-bold">{trades}</span></span>
                </div>
            </div>

            <div className="flex gap-3 text-slate-400 items-center">
                <span className="font-semibold text-slate-200 hidden sm:inline">Tape:</span>
                <span className="text-flint-positive font-mono">ES 1m +0.32%</span>
                <span className="text-flint-negative font-mono">NQ 1m -0.18%</span>
                <span className="text-emerald-300 font-mono">CL +0.41%</span>
            </div>
        </footer>
    );
};
