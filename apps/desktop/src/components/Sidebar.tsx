import React from "react";
import { FiActivity, FiCpu, FiLayers } from "react-icons/fi";

const items = [
    { id: "trade", icon: FiActivity, label: "Trade" },
    { id: "ai", icon: FiCpu, label: "AI" },
    { id: "logs", icon: FiLayers, label: "Logs" },
];

export const Sidebar = () => (
    <aside className="w-16 bg-black border-r border-flint-border flex flex-col items-center py-6">
        {/* Logo */}
        <div className="mb-8">
            <div className="h-10 w-10 rounded-2xl bg-gradient-to-br from-flint-primary to-flint-accent flex items-center justify-center text-sm font-black text-white shadow-lg shadow-flint-primary/20 cursor-pointer hover:rotate-6 transition-transform">
                F
            </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 flex flex-col gap-4">
            {items.map((item, idx) => {
                const Icon = item.icon; // Extract the component
                return (
                    <button
                        key={item.id}
                        className={`h-11 w-11 rounded-2xl flex items-center justify-center text-xl transition-all duration-300 group relative
                  ${idx === 0
                                ? "bg-flint-primary/10 text-flint-primary"
                                : "text-flint-text-secondary hover:text-flint-text-primary hover:bg-flint-panel"}`}
                        title={item.label}
                    >
                        <Icon /> {/* Render as component */}
                        {idx === 0 && <div className="absolute left-0 w-1 h-1/2 bg-flint-primary rounded-r-full" />}
                    </button>
                );
            })}
        </nav>
    </aside>
);