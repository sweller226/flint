/** @type {import('tailwindcss').Config} */
module.exports = {
    presets: [
        require('../../packages/config/tailwind-preset.js')
    ],
    theme: {
        extend: {
            // Colors inherited from preset
            colors: {
                flint: {
                    bg: "#020814",
                    panel: "#050B10",
                    subpanel: "#07111D",
                    border: "#111827",
                    blue: "#2563EB",
                    green: "#22C55E",
                    violet: "#A855F7",
                    positive: "#16A34A",
                    negative: "#DC2626",
                    warning: "#FACC15",
                    text: {
                        primary: "#F9FAFB",
                        secondary: "#9CA3AF",
                        muted: "#6B7280",
                    },
                },
            },
            backgroundImage: {
                'flint-gradient': 'linear-gradient(135deg, #00FFA3 0%, #03E1FF 45%, #DC1FFF 100%)',
            },
            fontFamily: {
                sans: ["Inter", "system-ui", "sans-serif"],
            },
        },
    },
    content: [
        "./src/**/*.{js,ts,jsx,tsx}",
        "../../packages/ui/**/*.{js,ts,jsx,tsx}"
    ],
}
