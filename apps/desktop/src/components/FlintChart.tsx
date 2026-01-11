import React, { useEffect, useRef, useImperativeHandle, forwardRef } from "react";
import {
    createChart,
    ColorType,
    IChartApi,
    ISeriesApi,
    CandlestickData,
    Time,
    SeriesMarker,
    IPriceLine,
    LineStyle,
    LineData,
} from "lightweight-charts";

export type Candle = CandlestickData<Time>;

export type Annotation = {
    type: "trendline" | "hline";
    t1?: number;
    p1?: number;
    t2?: number;
    p2?: number;
    price?: number;
    color?: string;
    label?: string;
};

export type FlintChartHandle = {
    update: (candle: Candle) => void;
    setData: (candles: Candle[]) => void;
    scrollToEnd: () => void;
};

type FlintChartProps = {
    candles: Candle[];
    forecastSeries?: LineData<Time>[];
    theme?: "dark" | "light";
    onContextMenu?: (event: { x: number; y: number; price: number; time: number }) => void;
    markers?: SeriesMarker<Time>[];
    annotations?: Annotation[];
};

export const FlintChart = forwardRef<FlintChartHandle, FlintChartProps>(({
    candles,
    forecastSeries = [],
    theme = "dark",
    onContextMenu,
    markers = [],
    annotations = [],
}, ref) => {
    const containerRef = useRef<HTMLDivElement | null>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
    const forecastLineRef = useRef<ISeriesApi<"Line"> | null>(null);
    const lastTimeRef = useRef<number | null>(null);

    const trendLinesRef = useRef<ISeriesApi<"Line">[]>([]);
    const priceLinesRef = useRef<IPriceLine[]>([]);

    useImperativeHandle(ref, () => ({
        update: (candle: Candle) => {
            // Validate candle has a valid numeric time before updating
            if (!candle || typeof candle.time !== 'number' || isNaN(candle.time)) {
                console.warn('[FlintChart.update] Invalid candle, skipping');
                return;
            }

            if (lastTimeRef.current !== null && (candle.time as number) < lastTimeRef.current) {
                console.warn(`[FlintChart.update] SKIPPING - candle time ${candle.time} < lastTime ${lastTimeRef.current}`);
                return;
            }

            console.log(`[FlintChart.update] APPLYING candle at ${candle.time}, lastTime was ${lastTimeRef.current}`);
            seriesRef.current?.update(candle);
            lastTimeRef.current = candle.time as number;
        },
        setData: (data: Candle[]) => {
            seriesRef.current?.setData(data);
            if (data.length > 0) {
                const last = data[data.length - 1];
                if (typeof last.time === 'number') {
                    lastTimeRef.current = last.time;
                }
            } else {
                lastTimeRef.current = null;
            }
        },
        scrollToEnd: () => {
            chartRef.current?.timeScale().scrollToPosition(0, true);
        }
    }));

    // 1. Chart Initialization
    useEffect(() => {
        if (!containerRef.current) return;

        // Clean up previous chart if it exists to prevent duplicates
        if (chartRef.current) {
            chartRef.current.remove();
        }

        const chart = createChart(containerRef.current, {
            layout: {
                background: { type: ColorType.Solid, color: theme === "dark" ? "#050B10" : "#FFFFFF" },
                textColor: theme === "dark" ? "#9CA3AF" : "#1F2937",
            },
            grid: {
                vertLines: { color: theme === "dark" ? "#111827" : "#E5E7EB" },
                horzLines: { color: theme === "dark" ? "#111827" : "#E5E7EB" },
            },
            timeScale: {
                timeVisible: true,
                borderColor: theme === "dark" ? "#111827" : "#E5E7EB",
                rightOffset: 12, // Add some breathing room for the forecast
            },
            crosshair: {
                mode: 1, // Magnet mode
            }
        });

        // A. Add Candlestick Series (Main Data)
        const series = chart.addCandlestickSeries({
            upColor: "#22C55E",
            downColor: "#FB7185",
            borderVisible: false,
            wickUpColor: "#22C55E",
            wickDownColor: "#FB7185",
        });

        // B. Add Forecast Series (Overlay)
        const forecastLine = chart.addLineSeries({
            color: "#3B82F6", // Blue
            lineWidth: 2,
            lineStyle: LineStyle.Dashed, // Dashed to indicate prediction
            crosshairMarkerVisible: true,
            title: "Forecast",
            lastValueVisible: false, // Hide label on axis to avoid clutter
        });

        // Set Initial Data
        if (candles && candles.length > 0) {
            series.setData(candles);
        }

        if (forecastSeries && forecastSeries.length > 0) {
            forecastLine.setData(forecastSeries);
        }

        chartRef.current = chart;
        seriesRef.current = series;
        forecastLineRef.current = forecastLine;

        // Resize Observer
        const ro = new ResizeObserver(entries => {
            if (entries[0] && chartRef.current) {
                const { width, height } = entries[0].contentRect;
                chartRef.current.applyOptions({ width, height });
            }
        });
        ro.observe(containerRef.current);

        return () => {
            ro.disconnect();
            chart.remove();
            chartRef.current = null;
        };
    }, []); // Only run once on mount

    // 2. Reactive Update: Candles
    useEffect(() => {
        if (seriesRef.current && candles && candles.length > 0) {
            const lastCandleTime = candles[candles.length - 1].time as number;

            // Guard: Don't reset the chart if we're ahead of the candles prop (replay in progress)
            // This prevents `setData` from wiping out imperatively-added candles during re-renders
            if (lastTimeRef.current !== null && lastCandleTime < lastTimeRef.current) {
                // Replay has progressed past the candles prop - don't reset
                return;
            }

            seriesRef.current.setData(candles);
            if (typeof lastCandleTime === 'number') {
                lastTimeRef.current = lastCandleTime;
            }
        } else if (seriesRef.current && candles && candles.length === 0) {
            seriesRef.current.setData([]);
            lastTimeRef.current = null;
        }
    }, [candles]);

    // 3. Reactive Update: Forecast
    useEffect(() => {
        if (forecastLineRef.current && forecastSeries) {
            forecastLineRef.current.setData(forecastSeries);
        }
    }, [forecastSeries]);

    // 4. Reactive Update: Markers (attached to forecast line)
    useEffect(() => {
        if (forecastLineRef.current) {
            forecastLineRef.current.setMarkers(markers);
        }
    }, [markers]);

    // 5. Reactive Update: Annotations
    useEffect(() => {
        if (!chartRef.current || !seriesRef.current) return;

        // Clear old annotations
        trendLinesRef.current.forEach(line => chartRef.current?.removeSeries(line));
        trendLinesRef.current = [];
        priceLinesRef.current.forEach(line => seriesRef.current?.removePriceLine(line));
        priceLinesRef.current = [];

        annotations.forEach((ann) => {
            if (ann.type === "hline" && ann.price !== undefined) {
                const pLine = seriesRef.current!.createPriceLine({
                    price: ann.price,
                    color: ann.color || "#A855F7",
                    lineWidth: 1,
                    lineStyle: LineStyle.Dashed,
                    axisLabelVisible: true,
                    title: ann.label || "",
                });
                priceLinesRef.current.push(pLine);
            } else if (ann.type === "trendline" && ann.t1 && ann.t2 && ann.p1 && ann.p2) {
                const tLine = chartRef.current!.addLineSeries({
                    color: ann.color || "#FACC15",
                    lineWidth: 2,
                });
                tLine.setData([{ time: ann.t1 as Time, value: ann.p1 }, { time: ann.t2 as Time, value: ann.p2 }]);
                trendLinesRef.current.push(tLine);
            }
        });
    }, [annotations]);

    return <div ref={containerRef} className="w-full h-full relative" />;
});

FlintChart.displayName = "FlintChart";