import React, { useEffect, useRef } from 'react';
import { createChart, IChartApi, ColorType, CrosshairMode } from 'lightweight-charts';

interface ChartProps {
    candles: any[];
    ictLevels?: any;
}

export const Chart: React.FC<ChartProps> = ({ candles, ictLevels }) => {
    const chartContainer = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const candleSeriesRef = useRef<any>(null);

    useEffect(() => {
        if (!chartContainer.current) return;

        // Initialize chart
        const chart = createChart(chartContainer.current, {
            layout: {
                background: { type: ColorType.Solid, color: '#001219' },
                textColor: '#E0E0E0',
            },
            grid: {
                vertLines: { color: '#1A202C' },
                horzLines: { color: '#1A202C' },
            },
            width: chartContainer.current.clientWidth,
            height: 500,
            crosshair: {
                mode: CrosshairMode.Normal,
            }
        });

        const candleSeries = chart.addCandlestickSeries({
            upColor: '#94D2BD',
            downColor: '#FF6B6B',
            borderVisible: false,
            wickUpColor: '#94D2BD',
            wickDownColor: '#FF6B6B'
        });

        candleSeriesRef.current = candleSeries;
        chartRef.current = chart;

        const handleResize = () => {
            if (chartContainer.current)
                chart.applyOptions({ width: chartContainer.current.clientWidth });
        };
        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            chart.remove();
        };
    }, []);

    // Update Data and ICT Levels
    useEffect(() => {
        if (chartRef.current && candleSeriesRef.current && candles.length > 0) {
            const formatted = candles.map((c: any) => ({
                time: c.time, // Expecting unix timestamp (seconds)
                open: c.open,
                high: c.high,
                low: c.low,
                close: c.close
            }));

            // Use update if only 1 new candle? 
            // For simplicity, just set data logic or update logic.
            // If we are appending, we should use 'update'.
            // Here we might be receiving the Full List or Partial. 
            // Let's assume 'candles' prop is the full history for now or huge chunks.
            candleSeriesRef.current.setData(formatted);

            // Draw ICT Levels (naive implementation: remove all overlapping lines and redraw??)
            // Lightweight charts primitive lines are not easily cleared without clearing series.
            // For dynamic levels, we might use 'PriceLines' or 'CustomSeries'.
            // Simplest Hackathon way: Just one or two key lines using PriceLines.

            if (ictLevels?.liquidity) {
                // Example: Draw just the nearest BSL and SSL
                // In real app, we would manage a list of primitives.
            }
        }
    }, [candles, ictLevels]);

    return (
        <div
            ref={chartContainer}
            className="w-full h-full rounded shadow-xl border border-[#10141C]"
            style={{ minHeight: '500px' }}
        />
    );
};
