import React from "react";

export const OrderPanel: React.FC = () => {
    return (
        <div className="h-full flex flex-col">
            {/* Tabs: Order form / Order book / Market */}
            <div className="flex text-[11px] border-b border-[#10141C]">
                {["Order form", "Order book", "Market"].map((tab, idx) => (
                    <button
                        key={tab}
                        className={`flex-1 py-3 text-center transition-colors ${idx === 0
                                ? "border-b-2 border-[#00FFA3] text-[#E0E0E0] font-semibold"
                                : "text-[#9A9A9A] hover:bg-[#0D151D] hover:text-[#E0E0E0]"
                            }`}
                    >
                        {tab}
                    </button>
                ))}
            </div>

            {/* Buy / Sell toggle */}
            <div className="flex justify-center gap-2 p-3 text-[11px]">
                <button className="flex-1 py-1.5 rounded bg-[#123524] text-[#94D2BD] font-bold border border-transparent hover:border-[#94D2BD]/30 transition-all">
                    Buy
                </button>
                <button className="flex-1 py-1.5 rounded bg-[#200b11] text-[#FF6B6B] font-bold border border-transparent hover:border-[#FF6B6B]/30 transition-all">
                    Sell
                </button>
            </div>

            {/* Fields */}
            <div className="px-3 flex-1 overflow-auto text-[11px]">
                <div className="mb-3">
                    <div className="flex justify-between mb-1 text-[#9A9A9A]">
                        <span>Limit price</span>
                        <span>USD</span>
                    </div>
                    <div className="bg-[#001219] rounded px-3 py-2 border border-[#10141C] text-[#E0E0E0] font-mono">
                        4808.25
                    </div>
                </div>

                <div className="mb-3">
                    <div className="flex justify-between mb-1 text-[#9A9A9A]">
                        <span>Quantity</span>
                        <span>ES</span>
                    </div>
                    <div className="bg-[#001219] rounded px-3 py-2 border border-[#10141C] flex items-center justify-between">
                        <span className="font-mono text-[#E0E0E0]">1</span>
                        <span className="text-[#9A9A9A] text-[10px]">≈ $240,000 notional</span>
                    </div>
                </div>

                {/* Slider placeholder */}
                <div className="mb-4">
                    <div className="flex justify-between text-[#9A9A9A] mb-1">
                        <span>Risk per trade</span>
                        <span>1.0%</span>
                    </div>
                    <div className="h-1.5 rounded-full bg-[#10141C] overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-[#00FFA3] to-[#03E1FF] w-1/4 rounded-full" />
                    </div>
                </div>

                {/* Summary */}
                <div className="bg-[#001219] border border-[#10141C] rounded p-2 text-[#9A9A9A] mb-4 leading-relaxed">
                    Buy <span className="text-[#E0E0E0]">1 ES contract</span> at <span className="text-[#E0E0E0]">4808.25</span>, risk ~ $400, target 4820.50 (R:R 3.2).
                </div>
            </div>

            {/* Submit button */}
            <div className="p-3 mt-auto">
                <button className="w-full py-2.5 rounded bg-gradient-to-r from-[#00FFA3] to-[#03E1FF] text-[#001219] text-[12px] font-bold hover:shadow-[0_0_15px_rgba(0,255,163,0.3)] transition-all transform active:scale-[0.98]">
                    Place Sim Order
                </button>
            </div>
        </div>
    );
};
