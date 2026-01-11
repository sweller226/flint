import React from "react";
import { useSimulation } from "../context/SimulationContext";

import { FlintLogo } from "./FlintLogo";

export const TopBar = () => {
    const { botEnabled, setBotEnabled } = useSimulation();

    const toggleBot = () => {
        setBotEnabled(!botEnabled);
    };

    return (
        <header className="h-16 px-6 flex items-center justify-between bg-flint-panel border-b border-flint-border">
            <div className="flex items-center gap-4">
                <FlintLogo withText={true} size={28} />
                <div className="h-6 w-px bg-flint-border" />
                <div className="inline-flex items-center gap-2 rounded-lg bg-flint-subpanel px-3 py-1 text-[11px] text-flint-text-secondary border border-flint-border">
                    <span className="font-bold text-flint-text-primary uppercase tracking-wider">ES Futures</span>
                </div>
            </div>

            {/* Bot Controls */}
            <div className="flex items-center gap-3">
                <span className="text-[10px] font-bold text-flint-text-muted uppercase tracking-wider">Auto-Pilot</span>
                <button
                    onClick={toggleBot}
                    className={`relative w-12 h-6 rounded-full transition-colors duration-200 ease-in-out border ${botEnabled ? "bg-flint-blue border-flint-blue" : "bg-flint-bg border-flint-border"}`}
                >
                    <div className={`absolute top-0.5 left-0.5 bg-white w-4 h-4 rounded-full shadow-md transform transition-transform duration-200 ${botEnabled ? "translate-x-6" : "translate-x-0"}`} />
                </button>
                <div className={`text-[10px] font-bold px-2 py-0.5 rounded transition-colors ${botEnabled ? "text-flint-blue bg-flint-blue/10" : "text-flint-text-muted bg-white/5"}`}>
                    {botEnabled ? "ACTIVE" : "OFF"}
                </div>
            </div>
        </header >
    );
};
