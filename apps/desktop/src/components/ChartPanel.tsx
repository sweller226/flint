import React, { useEffect, useState } from "react";
import { FlintChart, Candle, FlintChartHandle } from "./FlintChart";
import { SolanaTerminal } from "./SolanaTerminal";
import { useSimulation } from "../context/SimulationContext";

const TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1D", "1W", "1M"];

// ES Futures contract quarters
const ES_CONTRACTS = [
    { code: "H", label: "Q1 (Mar)" },
    { code: "M", label: "Q2 (Jun)" },
    { code: "U", label: "Q3 (Sep)" },
    { code: "Z", label: "Q4 (Dec)" },
];

export const ChartPanel = () => {
    // const [timeframe, setTimeframe] = useState("1m"); // Removed local
    const [contract, setContract] = useState("H");

    const { candles: contextCandles, botEnabled, portfolio, timeframe, setTimeframe } = useSimulation();

    // Tool State
    const [activeTool, setActiveTool] = useState<"none" | "trendline" | "hline">("none");
    const [annotations, setAnnotations] = useState<any[]>([]);

    // Generate dynamic annotations for SL/TP based on position
    useEffect(() => {
        const newAnnotations: any[] = [];
        const pos = portfolio.positions.find(p => p.symbol === "ES=F");

        if (pos) {
            // Visualize Entry
            newAnnotations.push({
                type: "hline",
                price: pos.entryPrice,
                color: "#3B82F6", // Blue entry
                label: "ENTRY"
            });

            // Visualize Mock SL/TP (e.g., +/- 10 pts)
            const slPrice = pos.side === 'LONG' ? pos.entryPrice - 10 : pos.entryPrice + 10;
            const tpPrice = pos.side === 'LONG' ? pos.entryPrice + 20 : pos.entryPrice - 20;

            newAnnotations.push({
                type: "hline",
                price: slPrice,
                color: "#EF4444", // Red SL
                label: "SL"
            });
            newAnnotations.push({
                type: "hline",
                price: tpPrice,
                color: "#22C55E", // Green TP
                label: "TP"
            });
        }
        setAnnotations(newAnnotations);
    }, [portfolio.positions]);

    // We can support specific replay mode if needed, but for now we mirror context
    const [isReplayMode, setIsReplayMode] = useState(false);

    const chartRef = React.useRef<FlintChartHandle>(null);
    const [markers, setMarkers] = useState<any[]>([]);

    // Chart handles updates via props automatically now.
    // Ensure we trigger a resize or re-fit if needed, but usually setData handles it.

    return (
        <div className="h-full flex flex-col bg-flint-panel relative overflow-hidden rounded-xl border border-flint-border shadow-lg">
            {/* Header */}
            <div className="h-12 border-b border-flint-border flex items-center justify-between px-4 bg-flint-panel z-10">
                <div className="flex gap-4 items-center">
                    {/* Ticker/Mode Selector */}
                    <div className="flex items-center gap-2">
                        <div className="flex items-center bg-flint-subpanel rounded-lg p-1 border border-flint-border">
                            <span className="px-3 py-1 rounded-md text-[11px] font-bold bg-flint-blue text-white shadow-sm">ES</span>
                        </div>
                    </div>

                    <div className="flex items-center gap-1">
                        {TIMEFRAMES.map(t => (
                            <button
                                key={t}
                                onClick={() => setTimeframe(t)}
                                className={`px-2 py-1 rounded text-[11px] font-medium transition-all ${timeframe === t ? "text-flint-blue bg-flint-blue/10" : "text-flint-text-muted hover:text-white"}`}
                            >
                                {t}
                            </button>
                        ))}
                    </div>
                </div>
                {/* Contract Selector */}
                <div className="flex items-center bg-flint-subpanel rounded-lg p-1 border border-flint-border ml-auto">
                    {ES_CONTRACTS.map(c => (
                        <button
                            key={c.code}
                            onClick={() => setContract(c.code)}
                            className={`px-2 py-1 rounded-md text-[10px] font-bold transition-all ${contract === c.code ? "bg-flint-green text-white shadow-sm" : "text-flint-text-muted hover:text-white hover:bg-white/5"}`}
                            title={c.label}
                        >
                            {c.code}
                        </button>
                    ))}
                </div>
                {/* Status Indicator */}
                <div className="flex items-center gap-3 ml-4">
                    <div className="flex items-center gap-2">
                        <span className="flex items-center gap-1.5 text-[10px] font-bold text-flint-green"><span className="h-1.5 w-1.5 rounded-full bg-flint-green shadow-[0_0_8px_rgba(34,197,94,0.8)]"></span> LIVE SIM</span>
                    </div>
                </div>
            </div>

            <div className="flex-1 flex overflow-hidden relative">
                {/* Chart Area */}
                <div className="flex-1 relative bg-flint-bg group">
                    <FlintChart
                        ref={chartRef}
                        candles={contextCandles as any}
                        annotations={annotations}
                        markers={markers}
                        onVisibleLogicalRangeChange={() => { }}
                    />
                </div>
            </div>

            {/* Solana Terminal */}
            <div className="h-48 border-t border-flint-border">
                <SolanaTerminal />
            </div>
        </div>
    );
};