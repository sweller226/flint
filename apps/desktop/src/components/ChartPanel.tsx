import React, { useEffect, useState } from "react";
import { FlintChart, Candle, FlintChartHandle } from "./FlintChart";
import axios from "axios";

const TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1D", "1W", "1M"];

// ES Futures contract quarters
const ES_CONTRACTS = [
    { code: "H", label: "Q1 (Mar)" },
    { code: "M", label: "Q2 (Jun)" },
    { code: "U", label: "Q3 (Sep)" },
    { code: "Z", label: "Q4 (Dec)" },
];

// Helper to format unix timestamp (seconds) to YYYY-MM-DDTHH:mm for datetime-local input
const formatForInput = (unixSeconds: number) => {
    const date = new Date(unixSeconds * 1000);
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    return `${year}-${month}-${day}T${hours}:${minutes}`;
};

export const ChartPanel = () => {
    const [timeframe, setTimeframe] = useState("1m");
    const [contract, setContract] = useState("H"); // Default to Q1 (March)

    const [candles, setCandles] = useState<Candle[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Tool State
    const [activeTool, setActiveTool] = useState<"none" | "trendline" | "hline">("none");
    const [annotations, setAnnotations] = useState<any[]>([]);

    // Backtest State
    const [isReplayMode, setIsReplayMode] = useState(false);
    const [isPlaying, setIsPlaying] = useState(false);
    const [hasSeeked, setHasSeeked] = useState(true);
    const [seekDate, setSeekDate] = useState("2026-01-01T00:00"); // Track selected date & time
    const [playbackSpeed, setPlaybackSpeed] = useState(1);
    const [isFetching, setIsFetching] = useState(false);

    // Simulation Refs
    const fullCandlesRef = React.useRef<Candle[]>([]);
    const playbackIndexRef = React.useRef(0);
    const intervalRef = React.useRef<NodeJS.Timeout | null>(null);
    const chartRef = React.useRef<FlintChartHandle>(null);
    const fetchControllerRef = React.useRef<AbortController | null>(null);

    // Stop simulation when leaving replay mode
    useEffect(() => {
        if (!isReplayMode) {
            stopSimulation();
        }

        // Cleanup: abort any pending fetches on unmount
        return () => {
            if (fetchControllerRef.current) {
                fetchControllerRef.current.abort();
            }
            stopSimulation();
        };
    }, [isReplayMode]);

    const stopSimulation = () => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
        setIsPlaying(false);
    };

    const startSimulation = () => {
        if (intervalRef.current) clearInterval(intervalRef.current);

        // User requested: playbackSpeed = candles per second.
        // e.g. 1x = 1 candle/sec
        const candlesPerSecond = playbackSpeed;
        const intervalMs = 1000 / candlesPerSecond;
        const safeInterval = Math.max(10, intervalMs);

        intervalRef.current = setInterval(() => {
            stepForward();
        }, safeInterval);

        setIsPlaying(true);
    };

    const stepForward = () => {
        const fullData = fullCandlesRef.current;
        const currentIdx = playbackIndexRef.current;

        if (currentIdx >= fullData.length - 1) {
            stopSimulation();
            return;
        }

        const nextIdx = currentIdx + 1;
        playbackIndexRef.current = nextIdx;

        // Imperatively update the chart (O(1)) without triggering React render
        if (chartRef.current) {
            const currentCandle = fullData[nextIdx];
            chartRef.current.update(currentCandle);

            // Update the seek input value in real-time
            setSeekDate(formatForInput(currentCandle.time as any));
        }
    };

    const togglePlay = () => {
        if (isPlaying) {
            stopSimulation();
        } else {
            startSimulation();
        }
    };

    // Restart simulation if speed changes while playing
    useEffect(() => {
        if (isPlaying) {
            startSimulation();
        }
    }, [playbackSpeed]);

    // Standard Data Fetching
    const fetchHistory = async (overrideDate?: string) => {
        // Cancel any pending fetch
        if (fetchControllerRef.current) {
            fetchControllerRef.current.abort();
        }

        // Use override date if provided, otherwise fallback to state if in replay mode
        const targetDate = overrideDate || (isReplayMode ? seekDate : undefined);

        console.log("[ChartPanel] fetchHistory triggered", { overrideDate, isReplayMode, seekDate, targetDate, timeframe });

        setIsFetching(true);
        setLoading(true);
        setError(null);
        stopSimulation(); // Ensure stop before loading

        // Create new abort controller for this fetch
        const controller = new AbortController();
        fetchControllerRef.current = controller;

        try {
            // Determine width in seconds
            const tfMap: Record<string, number> = {
                "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
                "1h": 3600, "4h": 14400, "1D": 86400, "1W": 604800, "1M": 2592000
            };
            const width = tfMap[timeframe] || 60;
            const limit = 100000; // Large chunk for replay

            let url = `http://localhost:8000/api/candles?contract=${contract}&limit=${limit}&width_seconds=${width}`;

            if (targetDate) {
                const dateObj = new Date(targetDate);
                const futureSeconds = limit * width;
                const endDateObj = new Date(dateObj.getTime() + (futureSeconds * 1000));
                url += `&end_time=${endDateObj.toISOString()}`;
                console.log("[ChartPanel] Replay Fetch URL Addon:", { futureSeconds, endDateStr: endDateObj.toISOString() });
            }

            console.log("[ChartPanel] Fetching URL:", url);
            const res = await axios.get(url, {
                timeout: 30000,
                signal: controller.signal
            });
            console.log("[ChartPanel] Response Code:", res.status, "Candles:", res.data?.candles?.length);

            // Check if this request was cancelled
            if (controller.signal.aborted) {
                console.log("[ChartPanel] Request was cancelled, ignoring response");
                return;
            }

            if (res.data && res.data.candles) {
                const mapped = res.data.candles.map((c: any) => ({
                    time: (Date.parse(c.timestamp) / 1000) as any,
                    open: c.open,
                    high: c.high,
                    low: c.low,
                    close: c.close,
                }));

                if (targetDate) {
                    // Replay Mode: Initialize full buffer and visible slice
                    fullCandlesRef.current = mapped;

                    // Find index of the seek date
                    const seekTs = (new Date(targetDate).getTime() / 1000);
                    let foundIdx = mapped.findIndex((c: any) => c.time > seekTs);

                    console.log("[ChartPanel] Replay Index Search:", { seekTs, foundIdx, total: mapped.length });

                    if (foundIdx === -1) foundIdx = mapped.length - 1;
                    else foundIdx = Math.max(0, foundIdx - 1);

                    playbackIndexRef.current = foundIdx;

                    // Initial set for the view
                    const initialSlice = mapped.slice(0, foundIdx + 1);
                    setCandles(initialSlice);

                    // Sync Chart
                    if (chartRef.current) {
                        console.log("[ChartPanel] Imperative setData call (Replay Init)", initialSlice.length);
                        chartRef.current.setData(initialSlice);
                    }
                } else {
                    setCandles(mapped);
                }
            } else {
                setError("Empty Data");
            }
        } catch (err: any) {
            // Ignore abort errors
            if (err.name === 'CanceledError' || err.code === 'ERR_CANCELED') {
                console.log("[ChartPanel] Fetch cancelled");
                return;
            }
            console.error("Failed to fetch candles:", err);
            setError(err.message || "Connection Err");
        } finally {
            // Only clear loading if this is still the active request
            if (fetchControllerRef.current === controller) {
                setLoading(false);
                setIsFetching(false);
                fetchControllerRef.current = null;
            }
        }
    };

    useEffect(() => {
        console.log("[ChartPanel] useEffect deps changed:", { timeframe, contract, isReplayMode, seekDate });
        // In replay mode, only auto-fetch on timeframe/contract change (not seekDate, that's manual)
        // In live mode, fetch on any change
        fetchHistory();
    }, [timeframe, contract, isReplayMode]);

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

                        {/* Replay Mode Toggle */}
                        <button
                            onClick={() => setIsReplayMode(!isReplayMode)}
                            className={`px-3 py-1 rounded-md text-[11px] font-bold border transition-all ${isReplayMode ? "bg-purple-600 border-purple-500 text-white" : "border-flint-border text-flint-text-muted hover:text-white"}`}
                        >
                            {isReplayMode ? "REPLAY ON" : "REPLAY OFF"}
                        </button>
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
                        {loading && !isReplayMode ? (
                            <span className="text-[10px] font-bold text-flint-blue animate-pulse">UPDATING...</span>
                        ) : isReplayMode ? (
                            <span className="flex items-center gap-1.5 text-[10px] font-bold text-purple-400"><span className="h-1.5 w-1.5 rounded-full bg-purple-500 shadow-[0_0_8px_rgba(168,85,247,0.8)]"></span> REPLAY</span>
                        ) : (
                            <span className="flex items-center gap-1.5 text-[10px] font-bold text-flint-green"><span className="h-1.5 w-1.5 rounded-full bg-flint-green shadow-[0_0_8px_rgba(34,197,94,0.8)]"></span> LIVE</span>
                        )}
                    </div>
                </div>
            </div>

            <div className="flex-1 flex overflow-hidden relative">
                {/* Chart Area */}
                <div className="flex-1 relative bg-flint-bg group">
                    <FlintChart
                        ref={chartRef}
                        candles={candles}
                        annotations={annotations}
                    />

                    {/* Bottom Center Replay Controls */}
                    {isReplayMode && (
                        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex items-center gap-4 bg-flint-panel/95 backdrop-blur-md border border-flint-border shadow-2xl rounded-full px-6 py-3 z-50 animate-in slide-in-from-bottom-4 duration-300">

                            {/* Date Picker (Seek) */}
                            <div className="flex items-center gap-2 pr-4 border-r border-flint-border/50">
                                <label className="text-[10px] font-bold text-flint-text-muted uppercase tracking-wider">Seek To</label>
                                <input
                                    type="datetime-local"
                                    className="bg-flint-bg/50 border border-flint-border rounded px-2 py-1 text-xs text-white focus:outline-none focus:border-flint-blue transition-colors cursor-pointer [color-scheme:dark]"
                                    value={seekDate}
                                    disabled={isFetching}
                                    onChange={(e) => {
                                        // Only update the local state, don't fetch yet
                                        if (e.target.value) {
                                            setSeekDate(e.target.value);
                                        }
                                    }}
                                    onBlur={(e) => {
                                        // Fetch when user is done selecting (picker closes)
                                        if (e.target.value && !isFetching) {
                                            const dateStr = e.target.value;
                                            setHasSeeked(false);
                                            fetchHistory(dateStr).then(() => {
                                                setHasSeeked(true);
                                            });
                                        }
                                    }}
                                />
                            </div>

                            {/* Playback Controls */}
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={togglePlay}
                                    disabled={!hasSeeked}
                                    className={`w-10 h-10 flex items-center justify-center rounded-full transition-all active:scale-95 ${!hasSeeked ? "opacity-50 cursor-not-allowed bg-flint-subpanel text-flint-text-muted" : isPlaying ? "bg-flint-blue text-white shadow-lg shadow-flint-blue/20" : "bg-flint-subpanel text-white hover:bg-white/10"}`}
                                    title={!hasSeeked ? "Select a date first" : isPlaying ? "Pause" : "Play"}
                                >
                                    {isPlaying ? (
                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16" rx="1" /><rect x="14" y="4" width="4" height="16" rx="1" /></svg>
                                    ) : (
                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" className="ml-0.5"><path d="M5 3l14 9-14 9V3z" /></svg>
                                    )}
                                </button>

                                <button
                                    onClick={stepForward}
                                    disabled={!hasSeeked}
                                    className={`w-8 h-8 flex items-center justify-center rounded-full bg-flint-subpanel text-flint-text-muted transition-all active:scale-95 ${!hasSeeked ? "opacity-50 cursor-not-allowed" : "hover:text-white hover:bg-white/10"}`}
                                    title="Step Forward"
                                >
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M5 3l10 9-10 9V3z" /><rect x="16" y="4" width="3" height="16" rx="1" /></svg>
                                </button>
                            </div>

                            <div className="w-px h-8 bg-flint-border/50"></div>

                            {/* Speed Selector */}
                            <div className="flex items-center gap-2 pl-2">
                                <span className="text-[10px] font-bold text-flint-text-muted uppercase tracking-wider">Speed</span>
                                <select
                                    value={playbackSpeed}
                                    onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
                                    className="bg-flint-bg/50 border border-flint-border text-white text-xs rounded px-2 py-1 focus:outline-none focus:border-flint-blue cursor-pointer appearance-none hover:bg-flint-bg transition-colors text-center w-20"
                                >
                                    <option value="1">1x</option>
                                    <option value="5">5x</option>
                                    <option value="15">15x</option>
                                </select>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Context Menu Placeholder */}
        </div>
    );
};