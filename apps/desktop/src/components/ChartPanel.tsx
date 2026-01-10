import React, { useEffect, useState, useRef } from "react";
import { Chart } from "./Chart"; // Import existing Chart component

interface ChartPanelProps {
    candles: any[];
    ictLevels: any;
    symbol?: string;
    onSymbolChange?: (sym: string) => void;
    timeframe?: string;
    onTimeframeChange?: (tf: string) => void;
}

export const ChartPanel: React.FC<ChartPanelProps> = ({
    candles, ictLevels, symbol = "ES", onSymbolChange, timeframe = "1m", onTimeframeChange
}) => {
    const [isSearchOpen, setIsSearchOpen] = useState(false);
    const [searchQuery, setSearchQuery] = useState("");
    const searchInputRef = useRef<HTMLInputElement>(null);

    // Global Keydown Listener for Type-to-Search
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            // Ignore if searching or typing in another input
            if (isSearchOpen) return;
            if (document.activeElement?.tagName === "INPUT" || document.activeElement?.tagName === "TEXTAREA") return;

            // Check if character is alphanumeric
            if (/^[a-zA-Z0-9]$/.test(e.key)) {
                setIsSearchOpen(true);
                setSearchQuery(e.key.toUpperCase());
            }
        };

        window.addEventListener("keydown", handleKeyDown);
        return () => window.removeEventListener("keydown", handleKeyDown);
    }, [isSearchOpen]);

    // Focus input when search opens
    useEffect(() => {
        if (isSearchOpen && searchInputRef.current) {
            searchInputRef.current.focus();
        }
    }, [isSearchOpen]);

    const [contextMenu, setContextMenu] = useState<{ x: number, y: number, visible: boolean }>({ x: 0, y: 0, visible: false });
    const [activeTool, setActiveTool] = useState("cursor");

    // Close context menu on click
    useEffect(() => {
        const handleClick = () => setContextMenu({ ...contextMenu, visible: false });
        window.addEventListener("click", handleClick);
        return () => window.removeEventListener("click", handleClick);
    }, [contextMenu]);

    const handleContextMenu = (e: React.MouseEvent) => {
        e.preventDefault();
        setContextMenu({
            x: e.clientX,
            y: e.clientY,
            visible: true
        });
    };

    const handleSearchCheck = (e: React.KeyboardEvent) => {
        if (e.key === "Enter") {
            onSymbolChange?.(searchQuery);
            setIsSearchOpen(false);
        }
        if (e.key === "Escape") {
            setIsSearchOpen(false);
        }
    };

    return (
        <div className="h-full flex flex-col relative" onContextMenu={handleContextMenu}>
            {/* Custom Context Menu */}
            {contextMenu.visible && (
                <div
                    className="fixed z-[100] bg-[#1E222D] border border-[#2A2E39] rounded shadow-lg py-2 w-48 text-sm"
                    style={{ top: contextMenu.y, left: contextMenu.x }}
                >
                    <div className="px-4 py-2 hover:bg-[#2A2E39] cursor-pointer text-[#E0E0E0] flex justify-between">
                        <span>Reset Chart</span>
                        <span className="text-[#555]">Alt+R</span>
                    </div>
                    <div className="px-4 py-2 hover:bg-[#2A2E39] cursor-pointer text-[#E0E0E0]">Copy Price</div>
                    <div className="my-1 border-t border-[#2A2E39]"></div>
                    <div className="px-4 py-2 hover:bg-[#2A2E39] cursor-pointer text-[#E0E0E0]">Add Alert...</div>
                    <div className="px-4 py-2 hover:bg-[#2A2E39] cursor-pointer text-[#E0E0E0]">Trade {symbol}</div>
                    <div className="my-1 border-t border-[#2A2E39]"></div>
                    <div className="px-4 py-2 hover:bg-[#2A2E39] cursor-pointer text-[#E0E0E0]">Settings...</div>
                </div>
            )}

            {/* Symbol Search Modal (TradingView Style) */}
            {isSearchOpen && (
                <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={() => setIsSearchOpen(false)}>
                    <div className="bg-[#1E222D] border border-[#2A2E39] rounded-lg shadow-2xl w-[400px] overflow-hidden" onClick={e => e.stopPropagation()}>
                        <div className="p-4 border-b border-[#2A2E39] flex items-center justify-between">
                            <h3 className="text-[#E0E0E0] font-medium">Symbol Search</h3>
                            <button onClick={() => setIsSearchOpen(false)} className="text-[#9A9A9A] hover:text-white">✕</button>
                        </div>
                        <div className="p-4">
                            <input
                                ref={searchInputRef}
                                className="w-full bg-[#10141C] border border-[#2A2E39] rounded px-3 py-2 text-xl text-white uppercase outline-none focus:border-[#2962FF]"
                                value={searchQuery}
                                onChange={e => setSearchQuery(e.target.value.toUpperCase())}
                                onKeyDown={handleSearchCheck}
                                placeholder="Symbol (e.g. BTC)"
                            />
                            <div className="mt-4 text-[#9A9A9A] text-xs">
                                Press <span className="text-[#E0E0E0] font-bold">ENTER</span> to select
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Chart header: tabs like Kraken */}
            <div className="flex items-center justify-between px-3 h-10 border-b border-[#10141C] text-[11px] bg-[#050B10]">
                <div className="flex gap-2 items-center">
                    <div className="flex items-center bg-[#10141C] rounded border border-[#1A202C] px-2 py-1 cursor-pointer hover:border-[#2962FF] transition-colors" onClick={() => setIsSearchOpen(true)}>
                        <span className="text-[#9A9A9A] mr-2">Ticker:</span>
                        <span className="text-[#E0E0E0] font-bold w-16 text-center block">{symbol}</span>
                    </div>

                    <div className="flex bg-[#10141C] rounded border border-[#1A202C]">
                        {['1m', '5m', '15m', '1h', '4h'].map(tf => (
                            <button
                                key={tf}
                                onClick={() => onTimeframeChange?.(tf)}
                                className={`px-2 py-1 hover:bg-[#202530] transition-colors ${timeframe === tf ? 'text-[#00E5FF] font-bold' : 'text-[#9A9A9A]'}`}
                            >
                                {tf}
                            </button>
                        ))}
                    </div>
                </div>
                <div className="flex gap-2 text-[#9A9A9A]">
                    <button className="px-2 py-1 hover:bg-[#10141C] rounded transition-colors">Candles</button>
                    <button className="px-2 py-1 hover:bg-[#10141C] rounded transition-colors">Indicators</button>
                    <button className="px-2 py-1 hover:bg-[#10141C] rounded transition-colors">Layouts</button>
                </div>
            </div>

            {/* Chart area */}
            <div className="flex-1 flex relative">
                {/* Left tools column (Genuine TV interactions) */}
                <div className="w-12 border-r border-[#10141C] flex flex-col items-center gap-1 py-2 text-[#9A9A9A] text-[16px] bg-[#050B10]">
                    {['cursor', 'trend', 'fib', 'brush', 'text', 'ruler'].map(tool => (
                        <button
                            key={tool}
                            className={`w-8 h-8 flex items-center justify-center rounded transition-colors ${activeTool === tool ? 'text-[#2962FF] bg-[#10141C]' : 'hover:text-[#E0E0E0] hover:bg-[#10141C]'}`}
                            onClick={() => setActiveTool(tool)}
                        >
                            {tool === 'cursor' && '✛'}
                            {tool === 'trend' && '╱'}
                            {tool === 'fib' && '≡'}
                            {tool === 'brush' && '🖌️'}
                            {tool === 'text' && 'T'}
                            {tool === 'ruler' && '📏'}
                        </button>
                    ))}
                    <div className="flex-1"></div>
                    <button className="w-8 h-8 flex items-center justify-center rounded hover:text-[#E0E0E0] hover:bg-[#10141C] mb-2">🗑️</button>
                </div>

                {/* Chart container */}
                <div className="flex-1 bg-[#001219] relative">
                    <Chart candles={candles} ictLevels={ictLevels} />
                </div>
            </div>
        </div>
    );
};
