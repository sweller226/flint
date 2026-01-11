// mockCandles.ts
import { Candle } from "./FlintChart";
import { Time } from "lightweight-charts";

export const mockCandles: Candle[] = [
    // time = unix seconds, 1-minute apart
    { time: 1704902400 as Time, open: 4800, high: 4804, low: 4798, close: 4802 },
    { time: 1704902460 as Time, open: 4802, high: 4806, low: 4800, close: 4805 },
    { time: 1704902520 as Time, open: 4805, high: 4810, low: 4804, close: 4809 },
    { time: 1704902580 as Time, open: 4809, high: 4812, low: 4803, close: 4804 },
    { time: 1704902640 as Time, open: 4804, high: 4807, low: 4799, close: 4800 },
    { time: 1704902700 as Time, open: 4800, high: 4801, low: 4794, close: 4796 },
    { time: 1704902760 as Time, open: 4796, high: 4800, low: 4793, close: 4798 },
];
