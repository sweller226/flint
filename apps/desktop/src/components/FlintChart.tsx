import React, { useEffect, useRef, useImperativeHandle, forwardRef } from "react";
import {
    createChart,
    ColorType,
    IChartApi,
    ISeriesApi,
    CandlestickData,
    Time,
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
};

export type FlintChartHandle = {
    update: (candle: Candle) => void;
    setData: (candles: Candle[]) => void;
};

type FlintChartProps = {
    candles: Candle[];
    theme?: "dark" | "light";
    onContextMenu?: (event: { x: number; y: number; price: number; time: number }) => void;
    markers?: any[];
    annotations?: Annotation[];
    onVisibleLogicalRangeChange?: (range: { from: number; to: number }) => void;
};

export const FlintChart = forwardRef<FlintChartHandle, FlintChartProps>(({
    candles,
    theme = "dark",
    onContextMenu,
    markers = [],
    annotations = [],
    onVisibleLogicalRangeChange,
}, ref) => {
    const containerRef = useRef<HTMLDivElement | null>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

    useImperativeHandle(ref, () => ({
        update: (candle: Candle) => {
            if (seriesRef.current) {
                seriesRef.current.update(candle);
            }
        },
        setData: (data: Candle[]) => {
            if (seriesRef.current) {
                seriesRef.current.setData(data);
            }
        }
    }));

    useEffect(() => {
        if (!containerRef.current) return;

        const chart = createChart(containerRef.current, {
            layout: {
                background: { type: ColorType.Solid, color: "#050B10" },
                textColor: "#9CA3AF",
            },
            grid: {
                vertLines: { color: "#111827" },
                horzLines: { color: "#111827" },
            },
            timeScale: {
                timeVisible: true,
                secondsVisible: false,
                borderColor: "#111827",
            },
            rightPriceScale: {
                borderColor: "#111827",
            }
        });

        // Handler for infinite scroll / chunking
        if (onVisibleLogicalRangeChange) {
            chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
                if (range) {
                    onVisibleLogicalRangeChange({ from: range.from, to: range.to });
                }
            });
        }

        const series = chart.addCandlestickSeries({
            upColor: "#22C55E",
            downColor: "#FB7185",
            borderVisible: false,
            wickUpColor: "#22C55E",
            wickDownColor: "#FB7185",
        });

        series.setData(candles);

        chartRef.current = chart;
        seriesRef.current = series;

        const ro = new ResizeObserver(entries => {
            if (!entries[0] || !chartRef.current) return;
            const { width, height } = entries[0].contentRect;
            chart.applyOptions({ width, height });
        });
        ro.observe(containerRef.current);

        // Context Menu Trigger (Custom Right Click)
        if (onContextMenu) {
            containerRef.current.oncontextmenu = (event) => {
                event.preventDefault();
                const rect = containerRef.current!.getBoundingClientRect();
                const x = event.clientX - rect.left;
                const y = event.clientY - rect.top;

                const timeScale = chart.timeScale();
                const price = series.coordinateToPrice(y);

                if (price != null) {
                    onContextMenu({
                        x: event.clientX,
                        y: event.clientY,
                        time: timeScale.coordinateToTime(x) as number,
                        price: price as number,
                    });
                }
            };
        }

        return () => {
            ro.disconnect();
            chart.remove();
        };
    }, []);

    // Prop update listener - debounced to prevent rapid updates
    useEffect(() => {
        if (!seriesRef.current || !chartRef.current) return;
        if (candles.length === 0) return;

        // Use a timeout to prevent multiple rapid updates
        const timer = setTimeout(() => {
            console.log("[FlintChart] candles prop changed:", candles.length, "- Resetting data and view");
            seriesRef.current?.setData(candles);
        }, 10);

        return () => clearTimeout(timer);
    }, [candles]);

    useEffect(() => {
        if (!seriesRef.current || !chartRef.current) return;
        if (markers.length > 0) {
            seriesRef.current.setMarkers(markers);
        }
    }, [markers]);

    // Handle Annotations (Drawing Tools)
    useEffect(() => {
        if (!chartRef.current) return;

        const lines: ISeriesApi<"Line">[] = [];

        annotations.forEach((ann) => {
            const line = chartRef.current!.addLineSeries({
                color: ann.color || (ann.type === "hline" ? "#A855F7" : "#FACC15"),
                lineWidth: 2,
                lineStyle: ann.type === "hline" ? 2 : 0, // Dotted for hline, Solid for trend
                lastValueVisible: false,
                priceLineVisible: ann.type === "hline",
            });

            if (ann.type === "hline" && ann.price !== undefined) {
                // Horizontal line across all visible range
                const lineData = candles.map(c => ({ time: c.time, value: ann.price! }));
                line.setData(lineData);
            } else if (ann.type === "trendline" && ann.t1 && ann.t2 && ann.p1 && ann.p2) {
                // Trendline between two points
                line.setData([
                    { time: ann.t1 as Time, value: ann.p1 },
                    { time: ann.t2 as Time, value: ann.p2 }
                ]);
            }
            lines.push(line);
        });

        return () => {
            lines.forEach(l => chartRef.current?.removeSeries(l));
        };
    }, [annotations, candles]); // Annotations depend on candles for 'horizontal lines' across range

    return <div ref={containerRef} className="w-full h-full relative" />;
});

FlintChart.displayName = "FlintChart";