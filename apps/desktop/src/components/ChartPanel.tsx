import React, { useEffect, useState } from "react";
import { FlintChart, Candle } from "./FlintChart";
// import { TerminalChartMock } from "./TerminalChartMock";
import axios from "axios";

const TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1D", "1W", "1M"];

// ES Futures contract quarters
const ES_CONTRACTS = [
    { code: "H", label: "Q1 (Mar)" },
    { code: "M", label: "Q2 (Jun)" },
    { code: "U", label: "Q3 (Sep)" },
    { code: "Z", label: "Q4 (Dec)" },
];

// Helper to format timestamp for slider display (Removed)

export const ChartPanel = () => {
    const [timeframe, setTimeframe] = useState("1m");
    const [contract, setContract] = useState("H"); // Default to Q1 (March)

    const [candles, setCandles] = useState<Candle[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Tool State
    const [activeTool, setActiveTool] = useState<"none" | "trendline" | "hline">("none");
    const [annotations, setAnnotations] = useState<any[]>([]);

    useEffect(() => {
        const fetchHistory = async () => {
            setLoading(true);
            setError(null);
            try {
                // Determine width in seconds
                const tfMap: Record<string, number> = {
                    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
                    "1h": 3600, "4h": 14400, "1D": 86400, "1W": 604800, "1M": 2592000
                };
                const width = tfMap[timeframe] || 60;

                // Fetch data from new backend with contract, width, and date range
                const res = await axios.get(`http://localhost:8000/api/candles?contract=${contract}&limit=10000&width_seconds=${width}`, { timeout: 30000 });

                if (res.data && res.data.candles) {
                    const mapped = res.data.candles.map((c: any) => ({
                        time: Date.parse(c.timestamp) / 1000,
                        open: c.open,
                        high: c.high,
                        low: c.low,
                        close: c.close,
                    }));
                    setCandles(mapped);
                } else {
                    setError("Empty Data");
                }
            } catch (err: any) {
                console.error("Failed to fetch candles:", err);
                setError(err.message || "Connection Err");
            } finally {
                setLoading(false);
            }
        };
        fetchHistory();
    }, [timeframe, contract]);

    return (
        <div className="h-full flex flex-col bg-flint-panel relative overflow-hidden rounded-xl border border-flint-border shadow-lg">
            {/* Header */}
            <div className="h-12 border-b border-flint-border flex items-center justify-between px-4 bg-flint-panel z-10">
                <div className="flex gap-4 items-center">
                    {/* Ticker Selector Placeholder - currently fixed to ES */}
                    <div className="flex items-center bg-flint-subpanel rounded-lg p-1 border border-flint-border">
                        <span className="px-3 py-1 rounded-md text-[11px] font-bold bg-flint-blue text-white shadow-sm">ES</span>
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
                    {/* Contract Selector - Always show now since we removed symbol selector */}
                    <div className="flex items-center bg-flint-subpanel rounded-lg p-1 border border-flint-border">
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
                </div>

                <div className="flex items-center gap-3">
                    <div className="flex items-center gap-2">
                        {loading ? (
                            <span className="text-[10px] font-bold text-flint-blue animate-pulse">UPDATING...</span>
                        ) : (
                            <span className="flex items-center gap-1.5 text-[10px] font-bold text-flint-green"><span className="h-1.5 w-1.5 rounded-full bg-flint-green shadow-[0_0_8px_rgba(34,197,94,0.8)]"></span> LIVE</span>
                        )}
                    </div>
                    <div className="h-4 w-px bg-flint-border"></div>
                    <button className="p-1.5 hover:bg-white/5 rounded text-flint-text-muted hover:text-white transition-all">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 3v18M3 12h18" /></svg>
                    </button>
                    <button className="p-1.5 hover:bg-white/5 rounded text-flint-text-muted hover:text-white transition-all">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" /></svg>
                    </button>
                </div>
            </div>

            <div className="flex-1 flex overflow-hidden relative">
                {/* Tools Rail */}
                <div className="w-12 border-r border-flint-border flex flex-col items-center gap-2 py-3 bg-flint-subpanel z-10">
                    {[
                        { id: "trendline" as const, icon: "✏️", label: "Draw Trendline" },
                        { id: "hline" as const, icon: "➖", label: "Horizontal Line" },
                    ].map(tool => (
                        <button
                            key={tool.id}
                            onClick={() => setActiveTool(tool.id)}
                            className={`w-8 h-8 rounded-lg flex items-center justify-center transition-all ${activeTool === tool.id ? "bg-flint-blue text-white shadow-[0_0_10px_rgba(37,99,235,0.3)]" : "text-flint-text-muted hover:bg-white/5 hover:text-white"}`}
                            title={tool.label}
                        >
                            <span className="text-[14px]">{tool.icon}</span>
                        </button>
                    ))}

                    <div className="w-6 h-px bg-flint-border my-1"></div>

                    <button
                        onClick={() => { setAnnotations([]); setActiveTool("none"); }}
                        className="w-8 h-8 rounded-lg flex items-center justify-center text-flint-text-muted hover:text-flint-negative hover:bg-flint-negative/10 transition-all"
                        title="Clear All"
                    >
                        <span className="text-[14px]">🗑️</span>
                    </button>
                </div>

                {/* Chart Area */}
                <div className="flex-1 relative bg-flint-bg">
                    <FlintChart
                        candles={candles}
                        annotations={annotations}
                    />
                </div>
            </div>

            {/* Context Menu Placeholder */}
        </div>
    );
};
