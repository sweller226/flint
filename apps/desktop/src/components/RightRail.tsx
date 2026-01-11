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

            {/* Strategy Lab */}
            <div className="h-[340px] rounded-2xl bg-flint-panel border border-flint-border p-4 flex flex-col shadow-2xl shadow-black/50">
                <h3 className="text-[10px] font-black text-flint-text-secondary mb-4 uppercase tracking-[0.2em]">Strategy Lab</h3>

                <div className="flex-1 overflow-auto mb-4 space-y-4 -mx-2 px-2 scrollbar-none">
                    {messages.length === 0 && (
                        <div className="bg-flint-primary/5 border border-flint-primary/10 rounded-xl p-4 text-flint-text-secondary text-[11px] leading-relaxed italic">
                            “Asia session high swept. Price entering bullish FVG—look for long scalp targets.”
                        </div>
                    )}
                    {messages.map((m, i) => (
                        <div key={i} className={`p-3 rounded-2xl text-[11px] leading-relaxed border shadow-sm ${m.role === 'user'
                                ? 'bg-flint-bg border-flint-border ml-6 text-flint-text-secondary rounded-br-none'
                                : 'bg-flint-primary/10 border-flint-primary/20 mr-6 text-flint-text-primary rounded-bl-none'
                            }`}>
                            {m.text}
                        </div>
                    ))}
                    {isTyping && (
                        <div className="text-[10px] text-flint-primary font-bold animate-pulse px-2 uppercase tracking-widest">Flint is calculating...</div>
                    )}
                </div>

                <div className="flex gap-2">
                    <textarea
                        className="flex-1 bg-flint-bg border border-flint-border rounded-xl px-4 py-3 text-flint-text-primary text-[11px] resize-none focus:outline-none focus:border-flint-primary/50 placeholder-flint-text-secondary/50 transition-all h-20 scrollbar-none"
                        placeholder="Ask strategy question..."
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                sendMessage();
                            }
                        }}
                    />
                    <button
                        onClick={sendMessage}
                        className="self-end p-3 rounded-xl bg-flint-primary text-white shadow-lg shadow-flint-primary/20 hover:scale-105 active:scale-95 transition-all disabled:opacity-50 disabled:grayscale"
                        disabled={isTyping || !inputText.trim()}
                    >
                        ↑
                    </button>
                </div>
            </div>
        </div>
    );
};
