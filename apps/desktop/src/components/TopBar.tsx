import React from "react";

export const TopBar = () => (
    <header className="h-14 px-6 flex items-center justify-between bg-flint-panel border-b border-flint-border">
        <div className="flex items-center gap-4">
            <span className="text-[15px] font-black tracking-tight text-white uppercase italic">
                Flint
            </span>
            <div className="inline-flex items-center gap-2 rounded-xl bg-flint-bg px-3 py-1.5 text-[10px] text-flint-text-secondary border border-flint-border">
                <span className="h-2 w-2 rounded-full bg-flint-secondary animate-pulse" />
                <span className="font-bold text-flint-text-primary uppercase tracking-wider">ES Futures</span>
                <span className="opacity-30">|</span>
                <span className="uppercase tracking-widest font-semibold italic text-flint-accent">Replay: Jan 03, 2017</span>
            </div>
        </div>

        <div className="flex items-center gap-6">
            <div className="flex items-center gap-3">
                <span className="text-[10px] uppercase font-bold text-flint-text-secondary tracking-widest">Session PnL</span>
                <span className="text-flint-secondary font-black bg-flint-secondary/10 px-3 py-1 rounded-lg border border-flint-secondary/20 tabular-nums">+ $1,240.00</span>
            </div>
            <button className="h-9 w-9 rounded-xl bg-flint-bg border border-flint-border flex items-center justify-center text-[14px] hover:border-flint-primary/50 transition-all text-flint-text-secondary hover:text-white">
                🌙
            </button>
        </div>
    </header>
);
