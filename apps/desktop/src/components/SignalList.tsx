import React from 'react';

// Basic Type Definition for Signals
export interface Signal {
    type: 'LONG' | 'SHORT';
    entry: number;
    stop_loss: number;
    take_profit: number;
    confidence: number;
    risk_reward: number;
    reason: string;
    timestamp: string;
}

interface SignalListProps {
    signals: Signal[];
    onExecute: (signal: Signal) => void;
}

export const SignalList: React.FC<SignalListProps> = ({ signals, onExecute }) => {
    return (
        <div className="space-y-2 p-4 bg-[#10141C] rounded h-full overflow-y-auto">
            <h3 className="text-sm font-bold text-[#E0E0E0] mb-3">Live Signals</h3>
            {signals.length === 0 && <p className="text-xs text-gray-500">Waiting for signals...</p>}

            {signals.map((signal, i) => (
                <div
                    key={i}
                    className={`p-3 rounded border transition-all hover:scale-[1.02] cursor-pointer ${signal.type === 'LONG'
                            ? 'border-[#94D2BD] bg-[#001219]/50 shadow-[0_0_10px_rgba(148,210,189,0.1)]'
                            : 'border-[#FF6B6B] bg-[#001219]/50 shadow-[0_0_10px_rgba(255,107,107,0.1)]'
                        }`}
                >
                    <div className="flex justify-between items-center">
                        <div className="flex-1">
                            <div className={`font-bold text-xs uppercase ${signal.type === 'LONG' ? 'text-[#94D2BD]' : 'text-[#FF6B6B]'}`}>
                                {signal.type} @ {signal.entry.toFixed(2)}
                            </div>
                            <div className="text-[10px] text-[#9A9A9A] mt-1 italic">{signal.reason}</div>
                            <div className="flex gap-4 mt-2 text-[10px] text-[#9A9A9A]">
                                <span>Conf: {(signal.confidence * 100).toFixed(0)}%</span>
                                <span>R:R {signal.risk_reward ? signal.risk_reward.toFixed(1) : 'N/A'}</span>
                            </div>
                        </div>
                        <button
                            onClick={() => onExecute(signal)}
                            className="ml-2 px-3 py-1 bg-gradient-to-r from-[#00FFA3] to-[#03E1FF] text-[#001219] text-[10px] font-bold rounded hover:opacity-90 active:scale-95 transition-transform"
                        >
                            SIM TRADE
                        </button>
                    </div>
                </div>
            ))}
        </div>
    );
};
