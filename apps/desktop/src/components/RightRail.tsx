import React, { useState, useEffect, useRef } from "react";

export const RightRail = () => {
    const [messages, setMessages] = useState<any[]>([]);
    const [signals, setSignals] = useState<any[]>([]);
    const [inputText, setInputText] = useState("");
    const [isTyping, setIsTyping] = useState(false);
    const ws = useRef<WebSocket | null>(null);

    useEffect(() => {
        const socket = new WebSocket("ws://localhost:8000/ws/signals");
        ws.current = socket;

        socket.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === "SIGNAL") {
                setSignals(prev => [msg.data, ...prev].slice(0, 10));
            } else if (msg.type === "CHAT_RESPONSE") {
                setIsTyping(false);
                setMessages(prev => [...prev, { role: "ai", text: msg.text }]);
            }
        };

        return () => socket.close();
    }, []);

    const sendMessage = () => {
        if (!inputText.trim() || !ws.current) return;

        const userMsg = inputText;
        setMessages(prev => [...prev, { role: "user", text: userMsg }]);
        ws.current.send(JSON.stringify({ type: "CHAT", text: userMsg }));
        setInputText("");
        setIsTyping(true);
    };

    return (
        <div className="flex flex-col gap-4 h-full">
            {/* Signals */}
            <div className="flex-1 rounded-2xl bg-flint-panel border border-flint-border p-4 overflow-hidden flex flex-col shadow-2xl shadow-black/50">
                <div className="flex justify-between items-center mb-4">
                    <h3 className="text-[10px] font-black text-flint-text-secondary uppercase tracking-[0.2em]">Live Signals</h3>
                    <span className="h-1.5 w-1.5 rounded-full bg-flint-secondary animate-pulse"></span>
                </div>

                <div className="flex-1 overflow-auto -mx-2 px-2 space-y-3 scrollbar-none">
                    {signals.length === 0 && (
                        <p className="text-flint-text-secondary/50 text-[10px] italic text-center py-10 uppercase tracking-widest">Awaiting market events...</p>
                    )}

                    {signals.map((sig, i) => (
                        <div key={i} className="bg-flint-bg/30 border border-flint-border p-3 rounded-xl hover:border-flint-primary/50 transition-all cursor-default group relative overflow-hidden">
                            <div className={`absolute left-0 top-0 bottom-0 w-1 ${sig.side === 'LONG' ? 'bg-flint-positive' : 'bg-flint-negative'}`}></div>
                            <div className="flex justify-between items-center mb-2 pl-2">
                                <span className={`font-black text-[11px] tracking-wider ${sig.side === 'LONG' ? 'text-flint-positive' : 'text-flint-negative'}`}>
                                    {sig.side} ES
                                </span>
                                <span className="text-[9px] font-bold text-flint-text-secondary uppercase">Live</span>
                            </div>
                            <div className="text-flint-text-primary text-[11px] mb-4 pl-2 font-medium leading-relaxed">
                                {sig.reason}
                            </div>
                            <div className="pl-2">
                                <div className="w-full bg-flint-bg h-1 rounded-full overflow-hidden mb-2">
                                    <div className={`h-full transition-all duration-1000 ${sig.side === 'LONG' ? 'bg-flint-positive' : 'bg-flint-negative'}`} style={{ width: `${sig.confidence * 100}%` }}></div>
                                </div>
                                <div className="flex justify-between items-center text-[9px] text-flint-text-secondary font-black uppercase tracking-tighter">
                                    <span>Confidence {Math.round(sig.confidence * 100)}%</span>
                                    <button className="text-flint-primary hover:text-white transition-colors">Execute →</button>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};
