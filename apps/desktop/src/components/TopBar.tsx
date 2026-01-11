import React from "react";

import { FlintLogo } from "./FlintLogo";

export const TopBar = () => (
    <header className="h-16 px-6 flex items-center justify-between bg-flint-panel border-b border-flint-border">
        <div className="flex items-center gap-4">
            <FlintLogo withText={true} size={28} />
            <div className="h-6 w-px bg-flint-border" />
            <div className="inline-flex items-center gap-2 rounded-lg bg-flint-subpanel px-3 py-1 text-[11px] text-flint-text-secondary border border-flint-border">
                <span className="font-bold text-flint-text-primary uppercase tracking-wider">ES Futures</span>
            </div>
        </div>
    </header >
);
