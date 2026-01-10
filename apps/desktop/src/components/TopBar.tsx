import React from "react";

export const TopBar: React.FC = () => {
    return (
        <header className="h-12 px-4 flex items-center justify-between bg-[#050B10] border-b border-[#10141C]">
            {/* Left: pair + timeframe */}
            <div className="flex items-center gap-4">
                <div className="flex items-center gap-2 bg-[#10141C] rounded-full px-3 py-1 hover:bg-[#1A202C] cursor-pointer transition-colors">
                    <span className="h-6 w-6 rounded-full bg-[#F4A261] flex items-center justify-center text-[10px] font-bold text-black">
                        ES
                    </span>
                    <span className="text-xs font-semibold">E‑Mini S&P 500</span>
                </div>
                <div className="flex items-center gap-1 text-[11px] text-[#9A9A9A]">
                    {["1m", "5m", "15m", "1h", "4h"].map(tf => (
                        <button
                            key={tf}
                            className={`px-2 py-1 rounded transition-colors ${tf === "1m"
                                    ? "bg-[#10141C] text-[#E0E0E0]"
                                    : "hover:bg-[#10141C] hover:text-[#E0E0E0]"
                                }`}
                        >
                            {tf}
                        </button>
                    ))}
                </div>
            </div>

            {/* Right: connection + layout */}
            <div className="flex items-center gap-4 text-[11px]">
                <div className="flex items-center gap-2 text-[#7FD36A]">
                    <span className="h-2 w-2 rounded-full bg-[#7FD36A] shadow-[0_0_8px_#7FD36A]" />
                    <span className="font-mono">CONNECTED</span>
                </div>
                <span className="text-[#9A9A9A] font-mono">ES_FUTURES_SIM_MODE</span>
            </div>
        </header>
    );
};
