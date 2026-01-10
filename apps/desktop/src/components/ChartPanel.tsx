import React, { useEffect, useRef, useState } from "react";
import { createChart, ColorType } from "lightweight-charts";

export const ChartPanel = () => {
    const ref = useRef<HTMLDivElement | null>(null);
    const [symbol, setSymbol] = useState("ES");
    const [timeframe, setTimeframe] = useState("1m");
    const chartRef = useRef<any>(null);
    const candleSeriesRef = useRef<any>(null);
    const mlSeriesRef = useRef<any>(null);

    useEffect(() => {
        if (!ref.current) return;

        const chart = createChart(ref.current, {
            layout: {
                background: { type: ColorType.Solid, color: "#0F172A" }, // flint-panel
                textColor: "#F8FAFC", // flint-text-primary
            },
            grid: {
                vertLines: { color: "#1E293B" }, // flint-border
                horzLines: { color: "#1E293B" },
            },
            timeScale: {
                timeVisible: true,
                borderColor: "#1E293B",
                rightOffset: 5,
            },
            rightPriceScale: {
                borderColor: "#1E293B",
                alignLabels: true,
            },
            crosshair: {
                vertLine: { labelBackgroundColor: "#3B82F6" },
                horzLine: { labelBackgroundColor: "#3B82F6" },
            }
        });

        chartRef.current = chart;

        const candleSeries = chart.addCandlestickSeries({
            upColor: "#10B981", // flint-secondary
            downColor: "#EF4444", // flint-negative
            borderVisible: false,
            wickUpColor: "#10B981",
            wickDownColor: "#EF4444",
        });
        candleSeriesRef.current = candleSeries;

        const mlSeries = chart.addLineSeries({
            color: '#8B5CF6', // flint-accent
            lineWidth: 2,
            lineStyle: 2,
            title: 'ML Prediction',
        });
        mlSeriesRef.current = mlSeries;

        // WebSocket Connection
        const ws = new WebSocket("ws://localhost:8000/ws/signals");

        ws.onopen = () => {
            console.log("Connected to WS");
            ws.send(JSON.stringify({ type: "SUBSCRIBE", symbol: symbol, timeframe: timeframe }));
        };

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === "CANDLE") {
                candleSeries.update(msg.data);
                if (msg.ml) {
                    mlSeries.update({
                        time: msg.data.time,
                        value: msg.ml
                    });
                }
            }
        };

        const resizeObserver = new ResizeObserver(entries => {
            if (entries.length === 0 || !entries[0].contentRect) return;
            const { width, height } = entries[0].contentRect;
            chart.applyOptions({ width, height });
        });
        resizeObserver.observe(ref.current);

        return () => {
            ws.close();
            resizeObserver.disconnect();
            chart.remove();
        };
    }, [symbol, timeframe]);

    return (
        <div className="h-full flex flex-col bg-flint-panel">
            {/* header of chart */}
            <div className="h-10 border-b border-flint-border flex items-center justify-between px-3 text-[11px] text-flint-text-secondary bg-flint-panel">
                <div className="flex gap-4">
                    <div className="flex items-center gap-2">
                        {["ES", "BTC"].map(s => (
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
                        {["1m", "5m", "15m"].map(t => (
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
                    <button className="px-2 py-1 rounded hover:bg-flint-bg transition-all">Indicators</button>
                    <button className="px-2 py-1 rounded hover:bg-flint-bg transition-all">Overlays</button>
                    <button className="ml-2 p-1 text-flint-text-primary">⚙️</button>
                </div>
            </div>

            <div className="flex-1 flex overflow-hidden">
                {/* tools column */}
                <div className="w-10 border-r border-flint-border flex flex-col items-center gap-5 py-4 text-[16px] text-flint-text-secondary bg-flint-panel">
                    <button className="hover:text-flint-primary transition-colors tooltip" title="Draw">✏️</button>
                    <button className="hover:text-flint-primary transition-colors tooltip" title="Line">➖</button>
                    <button className="hover:text-flint-primary transition-colors tooltip" title="Fib">📐</button>
                    <button className="hover:text-flint-primary transition-colors tooltip" title="Reset">🔄</button>
                    <div className="flex-1"></div>
                    <button className="hover:text-flint-primary transition-colors text-[12px] opacity-50">📷</button>
                </div>
                {/* chart area */}
                <div ref={ref} className="flex-1 relative" />
            </div>
        </div>
    );
};
