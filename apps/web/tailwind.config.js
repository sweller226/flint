/** @type {import('tailwindcss').Config} */
module.exports = {
    presets: [
        require('../../packages/config/tailwind-preset.js')
    ],
    content: [
        './pages/**/*.{js,ts,jsx,tsx,mdx}',
        './components/**/*.{js,ts,jsx,tsx,mdx}',
        '../../packages/ui/**/*.{js,ts,jsx,tsx}'
    ],
}
