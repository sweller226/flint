/** @type {import('tailwindcss').Config} */
module.exports = {
    presets: [
        require('../../packages/config/tailwind-preset.js')
    ],
    content: [
        "./src/**/*.{js,ts,jsx,tsx}",
        "../../packages/ui/**/*.{js,ts,jsx,tsx}"
    ],
}
