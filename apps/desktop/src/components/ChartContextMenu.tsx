import React from "react";

type ChartContextProps = {
    visible: boolean;
    x: number;
    y: number;
    price: number;
    time: number;
    onAddHLine: () => void;
    onAddTrend: () => void;
    onClose: () => void;
};

export const ChartContextMenu: React.FC<ChartContextProps> = ({
    visible, x, y, price, onAddHLine, onAddTrend, onClose,
}) => {
    if (!visible) return null;
    return (
        <div
            className="fixed z-[100] rounded-lg bg-flint-panel text-flint-text-primary text-[11px] shadow-2xl border border-flint-border min-w-[180px] overflow-hidden backdrop-blur-md bg-opacity-90"
            style={{ top: y, left: x }}
            onMouseLeave={onClose}
        >
            <div className="px-3 py-2 border-b border-flint-border bg-black/20 text-[10px] uppercase font-black tracking-widest text-flint-text-secondary">
                Chart Tools
            </div>
            <button
                className="w-full text-left px-3 py-2.5 hover:bg-flint-primary/10 hover:text-flint-primary transition-colors flex items-center gap-2 group"
                onClick={() => { onAddHLine(); onClose(); }}
            >
                <span className="opacity-50 group-hover:opacity-100">➖</span>
                Add Horizontal Line at <span className="font-bold text-white">{price.toFixed(2)}</span>
            </button>
            <button
                className="w-full text-left px-3 py-2.5 hover:bg-flint-primary/10 hover:text-flint-primary transition-colors flex items-center gap-2 group"
                onClick={() => { onAddTrend(); onClose(); }}
            >
                <span className="opacity-50 group-hover:opacity-100">📐</span>
                Start Trendline Here
            </button>
            <div className="border-t border-flint-border mt-1">
                <button
                    className="w-full text-left px-3 py-2 hover:bg-flint-negative/10 hover:text-flint-negative transition-colors text-[10px]"
                    onClick={onClose}
                >
                    Cancel
                </button>
            </div>
        </div>
    );
};
