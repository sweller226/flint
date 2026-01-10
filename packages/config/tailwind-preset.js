/** @type {import('tailwindcss').Config} */
module.exports = {
    theme: {
        extend: {
            colors: {
                flint: {
                    bg: "#020617",     // Midnight
                    panel: "#0F172A",  // Deep Slate
                    border: "#1E293B", // Border Slate
                    primary: "#3B82F6", // Professional Blue (Phantom)
                    secondary: "#10B981", // Emerald
                    text: {
                        primary: "#F8FAFC", // White/Slate-50
                        secondary: "#94A3B8", // Muted Slate
                    },
                    positive: "#22C55E", // Green-500
                    negative: "#EF4444", // Red-500
                    accent: "#8B5CF6",   // Purple-500
                },
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
