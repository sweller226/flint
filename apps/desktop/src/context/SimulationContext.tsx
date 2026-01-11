import React, { createContext, useContext, useState, useEffect, useRef } from 'react';
import { REAL_DATA } from '../data/realData';

// Types
export interface Candle {
    time: number; // Unix timestamp in seconds
    open: number;
    high: number;
    low: number;
    close: number;
}

export interface Position {
    symbol: string;
    side: 'LONG' | 'SHORT';
    size: number;
    entryPrice: number;
    unrealizedPnL: number;
}

export interface Portfolio {
    balance: number;
    equity: number;
    positions: Position[];
    roi: number; // Annualized %
    totalTrades: number;
    winRate: number;
    maxDrawdown: number;
}

export interface Log {
    time: string;
    msg: string;
    type: 'system' | 'trade';
}

interface SimulationContextType {
    candles: Candle[];
    portfolio: Portfolio;
    logs: Log[];
    botEnabled: boolean;
    setBotEnabled: (enabled: boolean) => void;
    placeOrder: (side: 'BUY' | 'SELL', size: number) => void;
    lastTick: number | null;
    timeframe: string;
    setTimeframe: (tf: string) => void;
}

const SimulationContext = createContext<SimulationContextType | null>(null);

export const useSimulation = () => {
    const context = useContext(SimulationContext);
    if (!context) throw new Error("useSimulation must be used within SimulationProvider");
    return context;
};

const aggregateData = (data: Candle[], period: number): Candle[] => {
    const result: Candle[] = [];
    for (let i = 0; i < data.length; i += period) {
        const chunk = data.slice(i, i + period);
        if (chunk.length === 0) continue;

        const open = chunk[0].open;
        const close = chunk[chunk.length - 1].close;
        const high = Math.max(...chunk.map(c => c.high));
        const low = Math.min(...chunk.map(c => c.low));
        const time = chunk[0].time;

        result.push({ time, open, high, low, close });
    }
    return result;
};

// Helper to prepare real data
const prepareRealData = (timeframe: string) => {
    // Determine aggregation
    let period = 1;
    if (timeframe === '5m') period = 5;
    if (timeframe === '15m') period = 15;

    // Aggregate source data
    const aggregated = period > 1 ? aggregateData(REAL_DATA, period) : REAL_DATA;

    // For this demo, we use the 1m REAL_DATA and just slicing it.
    // In a real app, we would aggregate for 5m/15m.
    // Shift data so the "split" point is NOW.

    const REPLAY_COUNT = 200; // Candles to replay (fewer for higher TFs? Or same count?)
    // If we have fewer candles due to agg, ensure we don't slice out of bounds
    const available = aggregated.length;
    const count = Math.min(available, REPLAY_COUNT);

    const splitIdx = available - count;
    const past = aggregated.slice(0, splitIdx);
    const future = aggregated.slice(splitIdx);

    // Calculate time offset
    const lastPastTime = past[past.length - 1]?.time || Math.floor(Date.now() / 1000);
    const now = Math.floor(Date.now() / 1000);
    const offset = now - lastPastTime;

    return {
        past: past.map(c => ({ ...c, time: c.time + offset })),
        future: future.map(c => ({ ...c, time: c.time + offset })),
        offset
    };
};

export const SimulationProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    // State
    const [candles, setCandles] = useState<Candle[]>([]);
    const [portfolio, setPortfolio] = useState<Portfolio>({
        balance: 100000,
        equity: 100000,
        positions: [],
        roi: 0,
        totalTrades: 0,
        winRate: 0,
        maxDrawdown: 0
    });
    const [logs, setLogs] = useState<Log[]>([]);

    useEffect(() => {
        setLogs([{ time: new Date().toLocaleTimeString(), msg: "Simulation System Initialized", type: 'system' }]);
    }, []);
    const [botEnabled, setBotEnabled] = useState(false);
    const [lastTick, setLastTick] = useState<number | null>(null);
    const [timeframe, setTimeframe] = useState("1m");

    // Refs for loop
    const candlesRef = useRef<Candle[]>([]);
    const futureRef = useRef<Candle[]>([]); // Store future replay candles
    const portfolioRef = useRef<Portfolio>({ balance: 100000, equity: 100000, positions: [], roi: 0, totalTrades: 0, winRate: 0, maxDrawdown: 0 });
    const botEnabledRef = useRef(false);
    const timeframeRef = useRef("1m");

    // Sync Refs
    useEffect(() => { botEnabledRef.current = botEnabled; }, [botEnabled]);
    useEffect(() => {
        timeframeRef.current = timeframe;
        // Load Real Data
        const { past, future } = prepareRealData(timeframe);
        setCandles(past);
        candlesRef.current = past;
        futureRef.current = future;
    }, [timeframe]);

    // Simulation Loop
    useEffect(() => {
        const interval = setInterval(() => {
            const currentCandles = candlesRef.current;
            if (currentCandles.length === 0) return;

            const lastCandle = currentCandles[currentCandles.length - 1];
            const now = Math.floor(Date.now() / 1000); // Current seconds

            let newCandle: Candle;
            let updated = false;

            // Start Replay Logic
            const future = futureRef.current;
            let currentPrice = lastCandle.close;

            if (future.length > 0) {
                // Peek next candle
                const nextCandle = future[0];

                // For demo: pop immediately and add it (1 sec = 1 candle)
                // This gives a fast-paced replay feeling
                futureRef.current = future.slice(1);
                candlesRef.current = [...currentCandles, nextCandle];
                currentPrice = nextCandle.close;
                updated = true;
            } else {
                // Out of data - hold last price or loop?
                // Just hold.
                currentPrice = lastCandle.close;
            }

            setCandles([...candlesRef.current]);
            setLastTick(currentPrice);

            // Update PnL
            updatePortfolio(currentPrice);

            // Bot Logic - "Smart" Trading on Pivot Points
            if (botEnabledRef.current && updated) {
                // Determine phase of sine wave roughly by looking at last few candles
                // If local min -> BUY, If local max -> SELL
                if (currentCandles.length > 5) {
                    const last3 = currentCandles.slice(-3);
                    // Simple pivot detection
                    const p1 = last3[2].close;
                    const p2 = last3[1].close;
                    const p3 = last3[0].close;

                    // V shape buy
                    if (p2 < p3 && p2 < p1) {
                        // Check if we already have a long
                        const hasLong = portfolioRef.current.positions.some(p => p.side === 'LONG');
                        if (!hasLong) mockPlaceOrder('BUY', 1.0, currentPrice, "Bot: Local Min Detected");
                    }

                    // ^ shape sell
                    if (p2 > p3 && p2 > p1) {
                        const hasShort = portfolioRef.current.positions.some(p => p.side === 'SHORT');
                        if (!hasShort) mockPlaceOrder('SELL', 1.0, currentPrice, "Bot: Local Max Detected");
                    }
                }
            }

        }, 1000);

        return () => clearInterval(interval);
    }, []);

    const updatePortfolio = (currentPrice: number) => {
        const p = portfolioRef.current;
        let unrealized = 0;
        const newPositions = p.positions.map(pos => {
            const diff = currentPrice - pos.entryPrice;
            const pnl = pos.side === 'LONG' ? diff * 50 * pos.size : -diff * 50 * pos.size;
            unrealized += pnl;
            return { ...pos, unrealizedPnL: pnl };
        });

        const equity = p.balance + unrealized;

        // Max Drawdown Calculation
        // Assuming starting equity is 100000 for simplified DD relative to start, 
        // OR relative to peak equity seen so far. Let's do Peak-to-Valley.

        // We need to persist peakEquity across renders, but for now let's just use a ref or simplified logic.
        // Actually, let's treat 'balance' as the anchor if we don't track historical peak.
        // Better: Let's assume 100k is the baseline or track peak in the portfolio object if we wanted to be strict.
        // For this demo: DD = (Equity - 100000) if negative, else standard DD from peak?
        // Let's stick to strict MaxDD from Peak.

        // To do this simply without extra state: maxDrawdown is just the min(Equity - 100000) we've seen?
        // No, MaxDD is usually percentage drop from peak. 
        // Let's simplify: Display the worst PnL drop from 100k seen so far.

        let newMaxDD = p.maxDrawdown;
        const currentDn = equity - 100000;
        if (currentDn < newMaxDD) {
            newMaxDD = currentDn;
        }

        // Mock Annual ROI: (equity - start) / start * (365 / days_simulated)
        // For demo, just scale it up aggressively to show numbers
        const pnlPct = (equity - 100000) / 100000;
        const roi = pnlPct * 52 * 100; // Mocking "weekly" performance extrapolated to year

        const newPortfolio = {
            ...p,
            equity,
            positions: newPositions,
            roi,
            maxDrawdown: newMaxDD
        };
        portfolioRef.current = newPortfolio;
        setPortfolio(newPortfolio);
    };

    const mockPlaceOrder = (side: 'BUY' | 'SELL', size: number, price: number, reason: string = "Manual") => {
        const p = portfolioRef.current;
        const existing = p.positions.find(pos => pos.symbol === 'ES=F');

        // Simple netting
        let newPositions = [...p.positions];
        let newBalance = p.balance;

        let txSig = "tx_" + Math.random().toString(36).substr(2, 9);

        // Logging
        const logMsg = `${reason === "Manual" ? "ORDER" : "BOT"}: ${side} ${size} @ ${price.toFixed(2)}\nSig: ${txSig}`;
        setLogs(prev => [...prev, { time: new Date().toLocaleTimeString(), msg: logMsg, type: 'trade' }]);

        let newTrades = p.totalTrades;
        let newWins = Math.round(p.totalTrades * (p.winRate / 100)); // Reverse calc wins

        if (existing) {
            // Map side to position side
            const tradeType = side === 'BUY' ? 'LONG' : 'SHORT';

            if (existing.side !== tradeType) {
                // Closing / Flipping
                // Calc PnL on close
                const closeSize = Math.min(existing.size, size);
                const diff = price - existing.entryPrice;
                const pnl = existing.side === 'LONG' ? diff * 50 * closeSize : -diff * 50 * closeSize;
                newBalance += pnl;

                // Update Stats
                newTrades += 1;
                if (pnl > 0) newWins += 1;

                if (existing.size === size) {
                    newPositions = [];
                } else if (existing.size > size) {
                    // Partial close
                    newPositions = [{ ...existing, size: existing.size - size }];
                } else {
                    // Flip
                    const remSize = size - existing.size;
                    newPositions = [{ symbol: 'ES=F', side: tradeType, size: remSize, entryPrice: price, unrealizedPnL: 0 }];
                }
            } else {
                // Adding
                const totalSize = existing.size + size;
                const newAvg = ((existing.entryPrice * existing.size) + (price * size)) / totalSize;
                newPositions = [{ ...existing, size: totalSize, entryPrice: newAvg }];
            }
        } else {
            const tradeType = side === 'BUY' ? 'LONG' : 'SHORT';
            newPositions.push({ symbol: 'ES=F', side: tradeType, size, entryPrice: price, unrealizedPnL: 0 });
        }



        const newWinRate = newTrades > 0 ? (newWins / newTrades) * 100 : 0;
        const updated = { ...p, balance: newBalance, positions: newPositions, totalTrades: newTrades, winRate: newWinRate };
        portfolioRef.current = updated;
        setPortfolio(updated);
    };

    const placeOrder = (side: 'BUY' | 'SELL', size: number) => {
        const price = candlesRef.current[candlesRef.current.length - 1]?.close || 4800;
        mockPlaceOrder(side, size, price, "Manual");
    };

    return (
        <SimulationContext.Provider value={{
            candles,
            portfolio,
            logs,
            botEnabled,
            setBotEnabled,
            placeOrder,
            lastTick,
            timeframe,
            setTimeframe
        }}>
            {children}
        </SimulationContext.Provider>
    );
};
