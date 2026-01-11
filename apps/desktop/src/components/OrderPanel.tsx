import React, { useState } from "react";
import { useSimulation } from "../context/SimulationContext";

export const OrderPanel = () => {
    const { portfolio, placeOrder } = useSimulation();
    const [activeTab, setActiveTab] = useState("Order");
    const [side, setSide] = useState<"BUY" | "SELL">("BUY");
    const [price, setPrice] = useState("4808.25");
    const [quantity, setQuantity] = useState("1.00");
    const [risk, setRisk] = useState(1); // Percentage

    const handlePlaceOrder = () => {
        placeOrder(side, parseFloat(quantity));
    };

    return (
        <div className="h-full flex flex-col text-xs font-medium">
            {/* Tabs */}
            <div className="flex bg-flint-bg p-1 rounded-full mb-6">
                {["Order", "Book", "Depth"].map((tab) => (
                    <button
                        key={tab}
                        onClick={() => setActiveTab(tab)}
                        className={`flex-1 py-1.5 rounded-full text-[11px] transition-all duration-200 ${activeTab === tab
                            ? "bg-flint-panel text-flint-text-primary shadow-lg"
                            : "text-flint-text-secondary hover:text-flint-text-primary"
                            }`}
                    >
                        {tab}
                    </button>
                ))}
            </div>

            {/* Portfolio Summary */}
            {portfolio && (
                <div className="mb-4 bg-flint-subpanel p-2 rounded border border-flint-border text-[10px]">
                    <div className="flex justify-between">
                        <span className="text-flint-text-muted">Balance</span>
                        <span className="text-white">${portfolio.balance?.toFixed(2)}</span>
                    </div>
                    {Object.values(portfolio.positions || {}).map((pos: any) => (
                        <div key={pos.symbol} className="flex justify-between mt-1">
                            <span className={pos.side === 'LONG' ? "text-flint-positive" : "text-flint-negative"}>{pos.side} {pos.size} {pos.symbol}</span>
                            <span className={pos.unrealized_pnl >= 0 ? "text-flint-positive" : "text-flint-negative"}>${pos.unrealized_pnl?.toFixed(2)}</span>
                        </div>
                    ))}
                </div>
            )}

            {/* Buy/Sell toggle */}
            <div className="flex gap-2 mb-6">
                <button
                    onClick={() => setSide("BUY")}
                    className={`flex-1 py-2.5 rounded-xl font-bold transition-all duration-200 border ${side === "BUY"
                        ? "bg-flint-positive/10 border-flint-positive text-flint-positive"
                        : "bg-flint-bg border-flint-border text-flint-text-secondary hover:border-flint-text-secondary/50"
                        }`}
                >
                    Buy
                </button>
                <button
                    onClick={() => setSide("SELL")}
                    className={`flex-1 py-2.5 rounded-xl font-bold transition-all duration-200 border ${side === "SELL"
                        ? "bg-flint-negative/10 border-flint-negative text-flint-negative"
                        : "bg-flint-bg border-flint-border text-flint-text-secondary hover:border-flint-text-secondary/50"
                        }`}
                >
                    Sell
                </button>
            </div>

            {/* Inputs */}
            <div className="flex-1 overflow-auto -mx-2 px-2 scrollbar-none">
                <Field
                    label="Limit price"
                    value={price}
                    onChange={setPrice}
                    suffix="USD"
                />
                <Field
                    label="Quantity"
                    value={quantity}
                    onChange={setQuantity}
                    suffix="ES"
                />

                <div className="mt-6 mb-8">
                    <div className="flex justify-between text-flint-text-secondary text-[11px] mb-3">
                        <span className="font-semibold uppercase tracking-wider">Risk per trade</span>
                        <span className="text-flint-primary font-bold">{risk.toFixed(1)}%</span>
                    </div>
                    <div className="px-1">
                        <input
                            type="range"
                            min="0.1"
                            max="5"
                            step="0.1"
                            value={risk}
                            onChange={(e) => setRisk(parseFloat(e.target.value))}
                            className="w-full h-1.5 bg-flint-bg rounded-full appearance-none cursor-pointer accent-flint-primary"
                        />
                    </div>
                </div>

                <div className="bg-flint-bg/50 border border-flint-border rounded-xl p-4 text-flint-text-secondary text-[11px] leading-relaxed italic">
                    {side === "BUY" ? "Buy" : "Sell"} {quantity} ES at {price} with {(parseFloat(price) * 0.002).toFixed(2)} point stop (≈ ${Math.round(risk * 400)} risk) and {(parseFloat(price) * 0.003).toFixed(2)} point target.
                </div>
            </div>

            {/* Submit */}
            <div className="mt-4 pt-4 border-t border-flint-border">
                <button
                    onClick={handlePlaceOrder}
                    className={`w-full py-3 rounded-xl text-white text-xs font-bold shadow-lg transition-all active:scale-[0.98] ${side === "BUY" ? "bg-flint-positive shadow-green-500/20" : "bg-flint-negative shadow-red-500/20"
                        }`}
                >
                    Place simulated {side.toLowerCase()} order
                </button>
                <p className="mt-3 text-[10px] text-center text-flint-text-secondary leading-tight">
                    Flint executes against historical ES data logs on Solana devnet.
                </p>
            </div>
        </div>
    );
};

const Field = ({ label, value, suffix, onChange }: { label: string; value: string; suffix?: string; onChange: (v: string) => void }) => (
    <div className="mb-5">
        <div className="flex justify-between mb-2 text-flint-text-secondary text-[10px] uppercase tracking-widest font-bold">
            <span>{label}</span>
            {suffix && <span>{suffix}</span>}
        </div>
        <div className="bg-flint-bg rounded-xl px-4 py-3 border border-flint-border flex items-center justify-between hover:border-flint-primary/30 transition-all focus-within:border-flint-primary/50 focus-within:ring-1 focus-within:ring-flint-primary/20">
            <input
                type="text"
                value={value}
                onChange={(e) => onChange(e.target.value)}
                className="bg-transparent text-flint-text-primary text-sm font-bold w-full outline-none"
            />
            {suffix && (
                <span className="text-flint-text-secondary text-[10px] bg-flint-panel px-2 py-0.5 rounded-lg border border-flint-border ml-3 font-bold">
                    {suffix}
                </span>
            )}
        </div>
    </div>
);
