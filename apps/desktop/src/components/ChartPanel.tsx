import React, { useEffect, useState } from "react";
// import { FlintChart, Candle } from "./FlintChart";
import { TerminalChartMock } from "./TerminalChartMock";
import axios from "axios";

const TICKERS = ["ES", "BTC"];
const TIMEFRAMES = ["1m", "5m", "15m"];

export const ChartPanel = () => {
    const [symbol, setSymbol] = useState("ES");
    const [timeframe, setTimeframe] = useState("1m");
    // const [candles, setCandles] = useState<any[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // UI State for Context Menu
    // UI State for Context Menu (Disabled)

    // Tool State
    const [activeTool, setActiveTool] = useState<"none" | "trendline" | "hline">("none");
    // const [annotations, setAnnotations] = useState<any[]>([]);

    useEffect(() => {
        const fetchHistory = async () => {
            setLoading(true);
            setError(null);
            try {
                // Hardcoded to Jan 3, 2017 for the current objective
                const res = await axios.get(`http://localhost:8000/api/candles?symbol=${symbol}&timeframe=${timeframe}&date=2017-01-03`, { timeout: 5000 });
                if (res.data && res.data.length > 0) {
                    // setCandles(res.data);
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
    }, [symbol, timeframe]);

    // Context Menu Handlers (Disabled)

    return (
        <div className="h-full flex flex-col bg-flint-panel relative overflow-hidden">
            {/* header of chart */}
            <div className="h-10 border-b border-flint-border flex items-center justify-between px-3 text-[11px] text-flint-text-secondary bg-flint-panel z-10">
                <div className="flex gap-4">
                    <div className="flex items-center gap-2">
                        {TICKERS.map(s => (
                            <button
                                key={s}
                                onClick={() => setSymbol(s)}
                                className={`px-2 py-0.5 rounded font-bold transition-all ${symbol === s ? "bg-flint-primary text-white" : "hover:text-flint-text-primary"}`}
                            >
                                {s}
                            </button>
                        ))}
                    </div>
                    <div className="w-px h-4 bg-flint-border my-auto"></div>
                    <div className="flex items-center gap-2">
                        {TIMEFRAMES.map(t => (
                            <button
                                key={t}
                                onClick={() => setTimeframe(t)}
                                className={`px-1.5 rounded transition-all ${timeframe === t ? "text-flint-primary font-bold" : "hover:text-flint-text-primary"}`}
                            >
                                {t}
                            </button>
                        ))}
                    </div>
                </div>
                <div className="flex gap-2">
                    <div className="px-2 py-1 flex items-center gap-2 text-[10px] font-black tracking-widest">
                        {loading && <span className="animate-pulse text-flint-accent">FETCHING DATA...</span>}
                        {error && <span className="text-flint-negative">ERROR: {error}</span>}
                        {!loading && !error && <span className="flex items-center gap-1.5 text-flint-accent"><span className="h-1.5 w-1.5 rounded-full bg-flint-accent shadow-[0_0_8px_#8B5CF6]"></span> REPLAY MODE</span>}
                    </div>
                    <button className="px-2 py-1 rounded hover:bg-flint-bg transition-all">Indicators</button>
                    <button className="ml-2 p-1 text-flint-text-primary">⚙️</button>
                </div>
            </div>

            <div className="flex-1 flex overflow-hidden relative">
                {/* tools column */}
                <div className="w-10 border-r border-flint-border flex flex-col items-center gap-5 py-4 text-[16px] text-flint-text-secondary bg-flint-panel z-10">
                    <button
                        onClick={() => setActiveTool("trendline")}
                        className={`hover:text-flint-primary transition-colors tooltip p-1 rounded ${activeTool === "trendline" ? "bg-flint-primary/20 text-flint-primary border border-flint-primary/30 shadow-[0_0_10px_rgba(59,130,246,0.2)]" : ""}`}
                        title="Trendline"
                    >
                        ✏️
                    </button>
                    <button
                        onClick={() => setActiveTool("hline")}
                        className={`hover:text-flint-primary transition-colors tooltip p-1 rounded ${activeTool === "hline" ? "bg-flint-primary/20 text-flint-primary border border-flint-primary/30 shadow-[0_0_10px_rgba(59,130,246,0.2)]" : ""}`}
                        title="Horizontal Line"
                    >
                        ➖
                    </button>
                    <button className="hover:text-flint-primary transition-colors tooltip" title="Fib">📐</button>
                    <button
                        onClick={() => { /* setAnnotations([]); */ setActiveTool("none"); }}
                        className="hover:text-flint-negative transition-colors tooltip"
                        title="Clear All"
                    >
                        🔄
                    </button>
                    <div className="flex-1"></div>
                    <button className="hover:text-flint-primary transition-colors text-[12px] opacity-50">📷</button>
                </div>

                {/* chart area */}
                <div className="flex-1 relative bg-black p-2">
                    <TerminalChartMock />
                </div>
            </div>

            {/* Context Menu Removed for Smoke Test */}
        </div>
    );
};
