import React, { useEffect, useState, useRef, useCallback } from "react";
import { FlintChart, Candle, FlintChartHandle } from "./FlintChart";
import axios from "axios";

const ES_CONTRACTS = [
    { code: "H", label: "Q1 (Mar)" },
    { code: "M", label: "Q2 (Jun)" },
    { code: "U", label: "Q3 (Sep)" },
    { code: "Z", label: "Q4 (Dec)" },
];

const formatForInput = (unixSeconds: number) => {
    const date = new Date(unixSeconds * 1000);
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    return `${year}-${month}-${day}T${hours}:${minutes}`;
};

type ForecastData = {
    forecast: Candle[];
    actions: Array<{
        timestamp: string;
        action: string;
        price: number;
        position: number;
        pnl: number;
    }>;
    metrics: Record<string, any>;
    execution_time_ms: number;
};

export const ChartPanel = () => {
    const [contract, setContract] = useState("H");

    const [candles, setCandles] = useState<Candle[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const [activeTool, setActiveTool] = useState<"none" | "trendline" | "hline">("none");
    const [annotations, setAnnotations] = useState<any[]>([]);

    // Replay State
    const [isReplayMode, setIsReplayMode] = useState(false);
    const [isPlaying, setIsPlaying] = useState(false);
    const [seekDate, setSeekDate] = useState("2021-01-27T00:00");
    const [isFetching, setIsFetching] = useState(false);

    // Sync Ref for Loop
    const isPlayingRef = useRef(false);

    // Forecast State
    const [forecastEnabled, setForecastEnabled] = useState(true);
    const [currentForecast, setCurrentForecast] = useState<ForecastData | null>(null);
    const [forecastLoading, setForecastLoading] = useState(false);
    const lastForecastTimeRef = React.useRef<number | null>(null);
    const [markers, setMarkers] = useState<any[]>([]);

    const fullCandlesRef = React.useRef<Candle[]>([]);
    const playbackIndexRef = React.useRef(0);
    const isFirstReplayStepRef = React.useRef(true);
    const intervalRef = React.useRef<NodeJS.Timeout | null>(null);
    const chartRef = React.useRef<FlintChartHandle>(null);
    const fetchControllerRef = React.useRef<AbortController | null>(null);
    const forecastControllerRef = React.useRef<AbortController | null>(null);

    // Sync Ref with State
    useEffect(() => {
        isPlayingRef.current = isPlaying;

        if (isPlaying) {
            startSimulation();
        } else {
            stopSimulation();
        }
    }, [isPlaying]);

    useEffect(() => {
        if (!isReplayMode) {
            setIsPlaying(false);
            setCurrentForecast(null);
            setMarkers([]);
        }

        return () => {
            if (fetchControllerRef.current) fetchControllerRef.current.abort();
            if (forecastControllerRef.current) forecastControllerRef.current.abort();
            stopSimulation();
        };
    }, [isReplayMode]);

    const stopSimulation = () => {
        if (intervalRef.current) {
            clearTimeout(intervalRef.current);
            intervalRef.current = null;
        }
    };

    const shouldFetchForecast = () => {
        if (!forecastEnabled || !isReplayMode) return false;
        const currentIdx = playbackIndexRef.current;
        const fullData = fullCandlesRef.current;
        if (currentIdx >= fullData.length - 1) return false;
        const currentCandle = fullData[currentIdx];
        const currentTime = currentCandle.time as number;
        if (lastForecastTimeRef.current === null) return true;

        // Fetch every hour (3600 seconds)
        const timeSinceLastForecast = currentTime - lastForecastTimeRef.current;
        return timeSinceLastForecast >= 3600;
    };

    const fetchForecast = async (timestamp: number) => {
        if (forecastLoading) return;
        if (forecastControllerRef.current) forecastControllerRef.current.abort();

        const controller = new AbortController();
        forecastControllerRef.current = controller;
        setForecastLoading(true);

        try {
            const isoTimestamp = new Date(timestamp * 1000).toISOString();

            const res = await axios.get(`http://localhost:8000/api/forecast`, {
                params: { contract, start_ts: isoTimestamp },
                timeout: 90000,
                signal: controller.signal
            });

            if (controller.signal.aborted) return;

            const forecastCandles = res.data.forecast.map((c: any) => ({
                time: (Date.parse(c.timestamp) / 1000) as any,
                open: c.open,
                high: c.high,
                low: c.low,
                close: c.close,
            }));

            const rawActions = res.data.actions.filter((action: any) => action.executed_trade);

            // Group actions by time and keep only the one with greater absolute PnL
            const actionsMap = new Map<number, any>();
            rawActions.forEach((action: any) => {
                const time = Math.floor(Date.parse(action.timestamp) / 1000);
                const existing = actionsMap.get(time);
                if (!existing || Math.abs(action.pnl_net) > Math.abs(existing.pnl_net)) {
                    actionsMap.set(time, action);
                }
            });

            const newActionMarkers = Array.from(actionsMap.values()).map((action: any) => ({
                time: (Date.parse(action.timestamp) / 1000) as any,
                position: action.action === 'BUY' ? 'belowBar' : 'aboveBar',
                color: action.action === 'BUY' ? '#22C55E' : '#FB7185',
                shape: action.action === 'BUY' ? 'arrowUp' : 'arrowDown',
                text: `${action.action} @ ${action.price.toFixed(2)}\nPnL: ${action.pnl_net.toFixed(2)}`
            }));

            // Simply set all markers from this forecast (sorted by time for lightweight-charts)
            const sortedMarkers = newActionMarkers.sort((a, b) => (a.time as number) - (b.time as number));
            setMarkers(sortedMarkers);

            setCurrentForecast({
                forecast: forecastCandles,
                actions: res.data.actions,
                metrics: res.data.metrics,
                execution_time_ms: res.data.execution_time_ms
            });

            lastForecastTimeRef.current = timestamp;

        } catch (err: any) {
            if (err.name === 'CanceledError' || err.code === 'ERR_CANCELED') return;
            console.error("[Forecast] Failed:", err);
            setError(`Forecast failed: ${err.message}`);
        } finally {
            if (forecastControllerRef.current === controller) {
                setForecastLoading(false);
                forecastControllerRef.current = null;
            }
        }
    };

    const startSimulation = () => {
        stopSimulation();
        runSimulationLoop();
    };

    const runSimulationLoop = async () => {
        if (!isPlayingRef.current) return;

        await stepForward();

        const candlesPerSecond = 1; // 1x Speed
        const intervalMs = 1000 / candlesPerSecond;

        intervalRef.current = setTimeout(() => {
            runSimulationLoop();
        }, Math.max(10, intervalMs));
    };

    const stepForward = async () => {
        const fullData = fullCandlesRef.current;
        const currentIdx = playbackIndexRef.current;

        if (currentIdx >= fullData.length - 1) {
            setIsPlaying(false);
            return;
        }

        const nextIdx = currentIdx + 1;
        const currentCandle = fullData[nextIdx];

        // Validate candle exists and has valid time
        if (!currentCandle || typeof currentCandle.time !== 'number' || isNaN(currentCandle.time)) {
            console.warn('[stepForward] Invalid candle at index', nextIdx, currentCandle);
            setIsPlaying(false);
            return;
        }

        // AI Hook - Blocking wait
        if (shouldFetchForecast()) {
            console.log(`[stepForward] Fetching forecast @ ${currentCandle.time}`);
            await fetchForecast(currentCandle.time as number);
        }

        playbackIndexRef.current = nextIdx;

        if (chartRef.current) {
            console.log(`[stepForward] Calling update for candle at time ${currentCandle.time}`);
            chartRef.current.update(currentCandle);
            setSeekDate(formatForInput(currentCandle.time as any));

            // Force scroll ONLY on the first step of replay
            if (isFirstReplayStepRef.current) {
                chartRef.current.scrollToEnd();
                isFirstReplayStepRef.current = false;
            }
        }
    };

    const forecastLineData = React.useMemo(() => {
        if (!currentForecast || !currentForecast.forecast) return [];
        return currentForecast.forecast.map(c => ({
            time: c.time,
            value: c.close
        }));
    }, [currentForecast]);

    const fetchHistory = useCallback(async (overrideDate?: string) => {
        // Stop any running simulation immediately
        stopSimulation();
        if (fetchControllerRef.current) fetchControllerRef.current.abort();
        const targetDate = overrideDate || (isReplayMode ? seekDate : undefined);
        setIsFetching(true);
        setLoading(true);
        setError(null);
        setIsPlaying(false);

        const controller = new AbortController();
        fetchControllerRef.current = controller;

        try {
            const tfMap: Record<string, number> = {
                "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
                "1h": 3600, "4h": 14400, "1D": 86400, "1W": 604800, "1M": 2592000
            };
            const width = tfMap["1m"] || 60;
            const limit = 40000;
            let url = `http://localhost:8000/api/candles?contract=${contract}&limit=${limit}&width_seconds=${width}`;

            if (targetDate) {
                const dateObj = new Date(targetDate);
                const futureSeconds = limit * width;
                const endDateObj = new Date(dateObj.getTime() + (futureSeconds * 1000));
                url += `&end_time=${endDateObj.toISOString()}`;
            }

            const res = await axios.get(url, { timeout: 30000, signal: controller.signal });

            if (controller.signal.aborted) return;

            if (res.data && res.data.candles) {
                const mapped = res.data.candles.map((c: any) => ({
                    time: (Date.parse(c.timestamp) / 1000) as any,
                    open: c.open, high: c.high, low: c.low, close: c.close,
                }));

                if (targetDate) {
                    fullCandlesRef.current = mapped;
                    const seekTs = (new Date(targetDate).getTime() / 1000);
                    let foundIdx = mapped.findIndex((c: any) => c.time > seekTs);
                    if (foundIdx === -1) foundIdx = mapped.length - 1;
                    else foundIdx = Math.max(0, foundIdx - 1);

                    playbackIndexRef.current = foundIdx;
                    isFirstReplayStepRef.current = true;
                    const initialSlice = mapped.slice(0, foundIdx + 1);
                    setCandles(initialSlice);

                    lastForecastTimeRef.current = null;
                    setCurrentForecast(null);
                    setMarkers([]);

                    if (chartRef.current) chartRef.current.setData(initialSlice);
                } else {
                    fullCandlesRef.current = mapped;
                    playbackIndexRef.current = 0;  // Reset playback index
                    isFirstReplayStepRef.current = true;
                    setCandles(mapped);
                    if (chartRef.current) chartRef.current.setData(mapped);
                }
            } else {
                setError("Empty Data");
            }
        } catch (err: any) {
            if (err.name !== 'CanceledError' && err.code !== 'ERR_CANCELED') {
                setError(err.message || "Connection Err");
            }
        } finally {
            if (fetchControllerRef.current === controller) {
                setLoading(false);
                setIsFetching(false);
                fetchControllerRef.current = null;
            }
        }
    }, [contract, isReplayMode]);

    useEffect(() => {
        fetchHistory();
    }, [fetchHistory]);

    return (
        <div className="h-full flex flex-col bg-flint-panel relative overflow-hidden rounded-xl border border-flint-border shadow-lg">
            {/* Header */}
            <div className="h-12 border-b border-flint-border flex items-center justify-between px-4 bg-flint-panel z-10">
                <div className="flex gap-4 items-center">
                    <div className="flex items-center gap-2">
                        <div className="flex items-center bg-flint-subpanel rounded-lg p-1 border border-flint-border">
                            <span className="px-3 py-1 rounded-md text-[11px] font-bold bg-flint-blue text-white shadow-sm">ES</span>
                        </div>

                        <button
                            onClick={() => setIsReplayMode(!isReplayMode)}
                            className={`px-3 py-1 rounded-md text-[11px] font-bold border transition-all ${isReplayMode ? "bg-purple-600 border-purple-500 text-white" : "border-flint-border text-flint-text-muted hover:text-white"}`}
                        >
                            {isReplayMode ? "REPLAY ON" : "REPLAY OFF"}
                        </button>

                        {isReplayMode && (
                            <button
                                onClick={() => setForecastEnabled(!forecastEnabled)}
                                className={`px-3 py-1 rounded-md text-[11px] font-bold border transition-all ${forecastEnabled ? "bg-amber-600 border-amber-500 text-white" : "border-flint-border text-flint-text-muted hover:text-white"}`}
                            >
                                {forecastEnabled ? "AI ON" : "AI OFF"}
                            </button>
                        )}
                    </div>

                    <div className="flex items-center bg-flint-subpanel rounded-lg p-1 border border-flint-border ml-auto">
                        {ES_CONTRACTS.map(c => (
                            <button
                                key={c.code}
                                onClick={() => setContract(c.code)}
                                className={`px-2 py-1 rounded-md text-[10px] font-bold transition-all ${contract === c.code ? "bg-flint-green text-white shadow-sm" : "text-flint-text-muted hover:text-white hover:bg-white/5"}`}
                                title={c.label}
                            >
                                {c.label}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            <div className="flex-1 flex overflow-hidden relative">
                <div className="flex-1 relative bg-flint-bg group">
                    <FlintChart
                        ref={chartRef}
                        candles={candles}
                        annotations={annotations}
                        markers={markers}
                        forecastSeries={forecastLineData}
                    />

                    {/* Forecast Status Overlay (Top Left) */}
                    {isReplayMode && currentForecast && currentForecast.forecast.length > 0 && (
                        <div className="absolute top-4 left-4 bg-amber-500/10 backdrop-blur-sm border border-amber-500/30 rounded-lg px-3 py-1.5 shadow-lg pointer-events-none">
                            <div className="flex items-center gap-2">
                                <div className="w-8 h-0.5 bg-amber-500" style={{ borderTop: '2px dashed' }}></div>
                                <span className="text-[10px] font-bold text-amber-400 uppercase tracking-wider">
                                    Forecast Active
                                </span>
                            </div>
                        </div>
                    )}

                    {/* Loading Spinner */}
                    {isReplayMode && forecastLoading && (
                        <div className="absolute top-14 left-4 z-50 animate-in fade-in duration-200 pointer-events-none">
                            <div className="bg-flint-panel/90 border border-amber-500/50 rounded-lg p-2 shadow-lg flex items-center gap-3">
                                <div className="w-4 h-4 border-2 border-amber-500 border-t-transparent rounded-full animate-spin"></div>
                                <span className="text-xs font-medium text-amber-500">AI Computing...</span>
                            </div>
                        </div>
                    )}

                    {/* BOTTOM FLOATING CONTROL BAR */}
                    {isReplayMode && (
                        <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 z-40">
                            <div className="flex items-center gap-2 bg-flint-panel/80 backdrop-blur-md border border-flint-border p-2 rounded-xl shadow-2xl">
                                {/* Date Seeker */}
                                <div className="flex flex-col px-2">
                                    <label className="text-[9px] text-flint-text-muted font-bold uppercase tracking-wider mb-0.5">Seek Date</label>
                                    <input
                                        type="datetime-local"
                                        value={seekDate}
                                        onChange={(e) => {
                                            setSeekDate(e.target.value);
                                            fetchHistory(e.target.value);
                                        }}
                                        className="bg-flint-subpanel border border-flint-border rounded px-2 py-1 text-xs text-white focus:border-flint-blue outline-none w-auto min-w-[220px] [color-scheme:dark]"
                                    />
                                </div>

                                <div className="w-px h-8 bg-flint-border mx-1"></div>

                                {/* Play/Pause Main Button */}
                                <button
                                    onClick={() => setIsPlaying(!isPlaying)}
                                    className={`w-10 h-10 flex items-center justify-center rounded-full border transition-all ${isPlaying
                                        ? "bg-red-500/20 border-red-500 text-red-500 hover:bg-red-500/30"
                                        : "bg-green-500/20 border-green-500 text-green-500 hover:bg-green-500/30"
                                        }`}
                                >
                                    {isPlaying ? (
                                        // Using text fallback if icons missing, replace with <PauseIcon className="w-5 h-5" />
                                        <span className="font-bold text-xs">||</span>
                                    ) : (
                                        // Using text fallback if icons missing, replace with <PlayIcon className="w-5 h-5 ml-0.5" />
                                        <span className="font-bold text-xs">▶</span>
                                    )}
                                </button>


                            </div>
                        </div>
                    )}

                </div>
            </div>
        </div >
    );
};