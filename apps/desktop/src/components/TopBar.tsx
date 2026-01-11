import React from "react";

import { FlintLogo } from "./FlintLogo";

export const TopBar = () => (
    <header className="h-16 px-6 flex items-center justify-between bg-flint-panel border-b border-flint-border">
        <div className="flex items-center gap-4">
            <FlintLogo withText={true} size={28} />
            <div className="h-6 w-px bg-flint-border" />
            <div className="inline-flex items-center gap-2 rounded-lg bg-flint-subpanel px-3 py-1 text-[11px] text-flint-text-secondary border border-flint-border">
                <span className="h-2 w-2 rounded-full bg-flint-secondary animate-pulse" />
                <span className="font-bold text-flint-text-primary uppercase tracking-wider">ES Futures</span>
                <span className="opacity-30">|</span>
                <span className="uppercase tracking-widest font-semibold italic text-flint-accent">Replay: Jan 03, 2017</span>
            </div>
        </div>

        <div className="flex items-center gap-4">
            <span className="text-[10px] uppercase font-bold text-flint-text-muted tracking-widest">Session PnL</span>
            <span className="text-flint-green font-mono font-bold bg-flint-green/10 px-3 py-1.5 rounded-lg border border-flint-green/20 tabular-nums shadow-[0_0_10px_rgba(34,197,94,0.1)]">+ $1,240.00</span>
        </div>
        <button className="h-9 w-9 rounded-lg bg-flint-subpanel border border-flint-border flex items-center justify-center text-[14px] hover:border-flint-blue/50 hover:text-white transition-all">
            🌙
        </button>
    </header >
);
