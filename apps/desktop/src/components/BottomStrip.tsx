import React from "react";

export const BottomStrip = () => (
    <footer className="h-12 bg-flint-panel border-t border-flint-border px-4 py-2 text-[11px] flex items-center justify-between shrink-0">
        <div className="flex flex-col sm:flex-row gap-1 sm:gap-4 items-center">
            <div className="flex items-center gap-2">
                <span className="text-slate-400">Session PnL</span>
                <span className="text-flint-positive font-semibold">+ $1,240</span>
            </div>
            <div className="flex gap-3 text-slate-500 border-l border-slate-800 pl-3">
                <span>Win rate <span className="text-slate-300">61%</span></span>
                <span>Max DD <span className="text-flint-negative">-$420</span></span>
                <span>Trades <span className="text-slate-300">18</span></span>
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
