import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { useSimulation } from '../context/SimulationContext';


export const SolanaTerminal = () => {
    const { logs } = useSimulation();
    const bottomRef = useRef<HTMLDivElement>(null);

    // Auto-scroll
    useEffect(() => {
        if (bottomRef.current) {
            bottomRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [logs]);

    return (
        <div className="flex flex-col h-full bg-[#0B0F19] font-mono text-xs border-t border-white/5">
            <div className="flex items-center px-4 py-2 bg-white/5 border-b border-white/5">
                <div className="w-2 h-2 rounded-full bg-green-500 mr-2 animate-pulse" />
                <span className="text-white/60 font-bold tracking-widest uppercase">Solana Net Log</span>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-2 scrollbar-thin scrollbar-thumb-white/10">
                {logs.map((log, i) => (
                    <div key={i} className="mb-2 font-mono text-[10px] border-b border-flint-border/30 pb-1 last:border-0">
                        <div className="flex justify-between text-flint-text-secondary mb-0.5">
                            <span>{log.time}</span>
                            {log.msg.includes("Sig:") && (
                                <a href="#" className="text-flint-blue hover:text-white transition-colors underline decoration-flint-blue/30">View on SolScan ↗</a>
                            )}
                        </div>
                        <div className={`${log.msg.includes("BUY") ? "text-flint-positive" : log.msg.includes("SELL") ? "text-flint-negative" : "text-flint-text-primary"}`}>
                            {log.msg.split("\n")[0]}
                        </div>
                        {log.msg.includes("Sig:") && (
                            <div className="text-flint-text-muted opacity-50 truncate mt-0.5">
                                {log.msg.split("\n")[1]}
                            </div>
                        )}
                    </div>
                ))}
                <div ref={bottomRef} />
            </div>
        </div>
    );
};
