import React, { useEffect, useState } from 'react';
import Head from 'next/head';
import axios from 'axios';
import dynamic from 'next/dynamic';

// Layout Components
import { Sidebar } from '@/components/Sidebar';
import { TopBar } from '@/components/TopBar';
import { OrderPanel } from '@/components/OrderPanel';
import { BottomStrip } from '@/components/BottomStrip';
import { RightRail } from '@/components/RightRail';
import { Signal } from '@/components/SignalList';

// Dynamic import for ChartPanel to disable SSR (Lightweight Charts)
const ChartPanel = dynamic(
    () => import('@/components/ChartPanel').then((mod) => mod.ChartPanel),
    { ssr: false }
);

// Mock types
interface Candle {
    time: number;
    open: number;
    high: number;
    low: number;
    close: number;
}

export default function Terminal() {
    const [candles, setCandles] = useState<Candle[]>([
        { time: 1642425300, open: 4725.25, high: 4728.50, low: 4720.00, close: 4722.00 },
        { time: 1642425360, open: 4722.00, high: 4724.75, low: 4721.25, close: 4723.50 },
        { time: 1642425420, open: 4723.50, high: 4726.00, low: 4722.50, close: 4725.75 },
        { time: 1642425480, open: 4725.75, high: 4729.25, low: 4724.00, close: 4728.00 },
        { time: 1642425540, open: 4728.00, high: 4730.00, low: 4726.50, close: 4729.50 },
    ]);
    const [lastCandle, setLastCandle] = useState<Candle | null>(null);
    const [signals, setSignals] = useState<Signal[]>([]);
    const [ictLevels, setICTLevels] = useState<any>(null);
    const [connected, setConnected] = useState(false);
    const [txLogs, setTxLogs] = useState<string[]>([]);
    const [symbol, setSymbol] = useState("ES");
    const [timeframe, setTimeframe] = useState("1m");
    const [socket, setSocket] = useState<WebSocket | null>(null);

    useEffect(() => {
        // Connect to WebSocket
        const ws = new WebSocket('ws://localhost:8000/ws/signals');
        setSocket(ws);

        ws.onopen = () => {
            setConnected(true);
            // Initial sub
            ws.send(JSON.stringify({ type: "SUBSCRIBE", symbol: "ES", timeframe: "1m" }));
        };
        ws.onclose = () => setConnected(false);

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === 'CANDLE') {
                const c = msg.data;
                const candleObj = { time: c.time, open: c.open, high: c.high, low: c.low, close: c.close };
                setLastCandle(candleObj);
                setCandles(prev => {
                    const last = prev[prev.length - 1];
                    if (last && last.time === candleObj.time) {
                        // Update current candle
                        return [...prev.slice(0, -1), candleObj];
                    } else {
                        // New candle
                        return [...prev.slice(-199), candleObj]; // Keep last 200
                    }
                });

                if (msg.ict) {
                    setICTLevels(msg.ict);
                }
            }
            else if (msg.type === 'SIGNAL') {
                const s = msg.data;
                const newSig: Signal = {
                    type: s.side,
                    entry: s.price,
                    confidence: s.confidence,
                    reason: s.reason,
                    stop_loss: s.price * 0.99, // Mock
                    take_profit: s.price * 1.01,
                    risk_reward: 2.0,
                    timestamp: new Date().toISOString()
                };
                setSignals(prev => [newSig, ...prev].slice(0, 50));
            }
        };

        return () => ws.close();
    }, []);

    const handleSymbolChange = (newSym: string) => {
        setSymbol(newSym);
        setCandles([]); // Clear chart
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ type: "SUBSCRIBE", symbol: newSym, timeframe }));
        }
    };

    const handleTimeframeChange = (newTf: string) => {
        setTimeframe(newTf);
        setCandles([]); // Clear chart
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ type: "SUBSCRIBE", symbol, timeframe: newTf }));
        }
    };

    const handleExecute = async (signal: Signal) => {
        console.log("Executing trade for", signal);
        try {
            const res = await axios.post('http://localhost:8000/api/log-trade', {
                symbol: symbol,
                side: signal.type,
                price: signal.entry,
                size: 1.0,
                reason: signal.reason
            });
            if (res.data.success) {
                // Show Solana Badge logic can be handled via toast or log updates
                const logMsg = `Tx Confirmed: ${res.data.tx_sig.substring(0, 8)}...`;
                setTxLogs(prev => [logMsg, ...prev]);
            }
        } catch (e) {
            console.error(e);
        }
    };

    return (
        <div className="h-screen w-screen bg-[#001219] text-[#E0E0E0] flex font-sans overflow-hidden">
            <Head>
                <title>{`Flint Terminal | ${symbol} Futures`}</title>
            </Head>

            {/* Left sidebar */}
            <Sidebar />

            {/* Main column */}
            <div className="flex-1 flex flex-col min-w-0">
                <TopBar />

                {/* Middle area */}
                <div className="flex flex-1 overflow-hidden">
                    {/* Left: order form + chart (2-column inside) */}
                    <div className="flex flex-1 flex-col min-w-0">
                        <div className="flex flex-1 border-t border-[#10141C]">
                            {/* Order panel */}
                            <div className="w-[340px] border-r border-[#10141C] bg-[#050B10]">
                                <OrderPanel />
                            </div>

                            {/* Chart */}
                            <div className="flex-1 bg-[#050B10] min-w-0">
                                <ChartPanel
                                    candles={candles}
                                    ictLevels={ictLevels}
                                    symbol={symbol}
                                    onSymbolChange={handleSymbolChange}
                                    timeframe={timeframe}
                                    onTimeframeChange={handleTimeframeChange}
                                />
                            </div>
                        </div>

                        {/* Bottom balances / positions strip */}
                        <BottomStrip />
                    </div>

                    {/* Right rail: signals + AI */}
                    <div className="w-[300px] border-l border-[#10141C] bg-[#050B10]">
                        <RightRail signals={signals} onExecute={handleExecute} />
                    </div>
                </div>
            </div>
        </div>
    );
}
