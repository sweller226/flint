import React from 'react';

export const Button = ({ children, onClick, variant = 'primary', className = '' }: any) => {
    const baseStyle = "px-4 py-2 rounded font-bold transition-all disabled:opacity-50 select-none";
    const variants: any = {
        primary: "bg-flint-gradient text-[#001219] hover:opacity-90 active:scale-95",
        secondary: "bg-panel text-flint-text border border-gray-700 hover:border-gray-500",
        danger: "bg-flint-sell text-[#001219] hover:opacity-90",
        outline: "border border-[#00FFA3] text-[#00FFA3] hover:bg-[#00FFA3]/10"
    };
    return (
        <button className={`${baseStyle} ${variants[variant]} ${className}`} onClick={onClick}>
            {children}
        </button>
    );
};

export const Card = ({ children, title, className = '' }: any) => (
    <div className={`bg-panel border border-gray-800 rounded p-4 text-flint-text ${className}`}>
        {title && <h3 className="text-sm font-bold mb-3 text-flint-text-muted uppercase tracking-wider">{title}</h3>}
        {children}
    </div>
);

export const Badge = ({ children, variant = 'neutral' }: any) => {
    const variants: any = {
        neutral: "bg-gray-800 text-gray-400",
        buy: "bg-flint-buy/20 text-flint-buy border border-flint-buy/50",
        sell: "bg-flint-sell/20 text-flint-sell border border-flint-sell/50",
        solana: "bg-[#9945FF]/20 text-[#14F195] border border-[#9945FF]/50" // Solana colors
    };
    return (
        <span className={`text-[10px] font-mono px-2 py-0.5 rounded ${variants[variant]}`}>
            {children}
        </span>
    )
}
