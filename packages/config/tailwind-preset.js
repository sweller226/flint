/** @type {import('tailwindcss').Config} */
module.exports = {
    theme: {
        extend: {
            colors: {
                background: "#001219",      // Primary Background
                panel: "#10141C",           // Surface/Cards
                "flint-buy": "#94D2BD",     // Long/Positive
                "flint-sell": "#FF6B6B",    // Short/Negative
                "flint-warn": "#E9D8A6",    // Warning
                "flint-text": "#E0E0E0",    // Primary Text
                "flint-text-muted": "#9A9A9A", // Secondary Text
            },
            backgroundImage: {
                "flint-gradient": "linear-gradient(120deg, #00FFA3, #03E1FF, #DC1FFF)", // Brand Gradient
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', 'sans-serif'],
                mono: ['JetBrains Mono', 'monospace'],
                display: ['Inter', 'sans-serif'],
            }
        },
    },
    plugins: [],
}
