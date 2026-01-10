import React from "react";
import { FiHome, FiActivity, FiLayers, FiClock, FiHelpCircle } from "react-icons/fi";

const navItems = [
    { label: "Trade", icon: FiActivity, active: true },
    { label: "Markets", icon: FiLayers },
    { label: "Analytics", icon: FiActivity },
    { label: "History", icon: FiClock },
];

export const Sidebar: React.FC = () => {
    return (
        <aside className="w-[80px] bg-[#050B10] border-r border-[#10141C] flex flex-col items-center py-4">
            {/* Logo */}
            <div className="mb-6 flex items-center justify-center">
                <div className="h-9 w-9 rounded-full bg-gradient-to-br from-[#00FFA3] via-[#03E1FF] to-[#DC1FFF] flex items-center justify-center text-xs font-black text-black">
                    F
                </div>
            </div>

            {/* Nav items */}
            <nav className="flex-1 flex flex-col gap-2">
                {navItems.map(item => (
                    <button
                        key={item.label}
                        className={`h-10 w-10 rounded-xl flex items-center justify-center text-[18px] transition-colors
              ${item.active ? "bg-[#10141C] text-[#E0E0E0]" : "text-[#666a73] hover:bg-[#0D151D] hover:text-[#E0E0E0]"}
            `}
                    >
                        <item.icon />
                    </button>
                ))}
            </nav>

            {/* Help / collapse */}
            <div className="mt-auto mb-2 flex flex-col items-center gap-3">
                <button className="h-9 w-9 rounded-xl flex items-center justify-center text-[#666a73] hover:bg-[#0D151D] hover:text-[#E0E0E0]">
                    <FiHelpCircle />
                </button>
            </div>
        </aside>
    );
};
