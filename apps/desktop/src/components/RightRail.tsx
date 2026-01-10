import React from "react";
import { Signal } from "./SignalList";

interface RightRailProps {
    signals: Signal[];
    onExecute: (signal: Signal) => void;
}

export const RightRail: React.FC<RightRailProps> = ({ signals, onExecute }) => {
    return (
        <div className="h-full flex flex-col">
            {/* Live signals */}
            <div className="flex-1 border-b border-[#10141C] p-3 overflow-hidden flex flex-col">
                <h3 className="text-[12px] font-semibold mb-2 text-[#E0E0E0]">Live Signals</h3>

                <div className="flex-1 overflow-y-auto pr-1 space-y-2">
                    {signals.length === 0 && <p className="text-[11px] text-[#9A9A9A]">Waiting for signals...</p>}

                    {signals.map((signal, i) => (
                        <div key={i} className="bg-[#001219] border border-[#10141C] rounded p-3 mb-2 text-[11px] hover:border-[#1A2E35] transition-colors">
                            <div className="flex justify-between mb-1">
                                <span className={`font-bold ${signal.type === 'LONG' ? 'text-[#94D2BD]' : 'text-[#FF6B6B]'}`}>
                                    {signal.type} ES
                                </span>
                                <span className="text-[#9A9A9A]">Now</span>
                            </div>
                            <div className="flex justify-between text-[#9A9A9A] mb-2 font-mono">
                                <span>@ {signal.entry.toFixed(2)}</span>
                                <span className="text-[#FF6B6B]">SL {signal.stop_loss.toFixed(2)}</span>
                                <span className="text-[#94D2BD]">TP {signal.take_profit.toFixed(2)}</span>
                            </div>
                            <div className="flex justify-between items-center">
                                <span className="text-[#9A9A9A] italic">
                                    {signal.reason.substring(0, 20)}...
                                </span>
                                <button
                                    onClick={() => onExecute(signal)}
                                    className="px-3 py-1.5 rounded bg-[#10141C] text-[10px] text-[#E0E0E0] hover:bg-[#00FFA3] hover:text-black font-bold transition-all"
                                >
                                    Sim Trade
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Strategy Lab */}
            <div className="h-[260px] p-3 flex flex-col">
                <h3 className="text-[12px] font-semibold mb-2 text-[#E0E0E0]">Strategy Lab (Gemini)</h3>
                <div className="bg-[#001219] border border-[#10141C] rounded p-3 mb-2 text-[11px] text-[#9A9A9A] italic flex-1 overflow-y-auto">
                    "Market is testing Asia session highs. Bearish FVG detected at 4840. Structure shift likely below 4790..."
                </div>
                <textarea
                    rows={3}
                    placeholder="Ask Flint to explain this setup or refine your ICT idea..."
                    className="w-full bg-[#001219] border border-[#10141C] rounded px-2 py-2 text-[11px] mb-2 focus:outline-none focus:border-[#03E1FF] text-[#E0E0E0]"
                />
                <button className="w-full py-2 rounded bg-gradient-to-r from-[#00FFA3] to-[#03E1FF] text-[#001219] text-[11px] font-bold hover:opacity-90 transition-opacity">
                    Send to Gemini
                </button>
            </div>
        </div>
    );
};
