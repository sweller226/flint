import React from "react";

export const BottomStrip: React.FC = () => {
    return (
        <div className="h-10 border-t border-[#10141C] bg-[#050B10] px-3 py-2 text-[11px] flex items-center justify-between">
            <div className="flex gap-6 items-center">
                <div className="flex gap-2">
                    <span className="text-[#9A9A9A]">PnL</span>
                    <span className="text-[#94D2BD] font-mono font-bold">+ $1,240.50</span>
                </div>
                <div className="h-3 w-px bg-[#10141C]" />
                <div className="flex gap-2 text-[#9A9A9A]">
                    <span>Win %</span>
                    <span className="text-[#E0E0E0] font-mono">62%</span>
                </div>
                <div className="h-3 w-px bg-[#10141C]" />
                <div className="flex gap-2 text-[#9A9A9A]">
                    <span>Max DD</span>
                    <span className="text-[#FF6B6B] font-mono">-$450.00</span>
                </div>
            </div>

            <div className="flex gap-4 text-[#9A9A9A] items-center">
                <span>Favorites:</span>
                <span className="text-[#E0E0E0] hover:text-white cursor-pointer px-1">ES 1m</span>
                <span className="text-[#FF6B6B] hover:text-[#ff8585] cursor-pointer px-1">NQ -0.42%</span>
                <span className="text-[#94D2BD] hover:text-[#b0ffe6] cursor-pointer px-1">CL +0.31%</span>
            </div>
        </div>
    );
};
