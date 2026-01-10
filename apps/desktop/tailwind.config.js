/** @type {import('tailwindcss').Config} */
module.exports = {
    presets: [
        require('../../packages/config/tailwind-preset.js')
    ],
    theme: {
        extend: {
            // Colors inherited from preset
            colors: {},
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
