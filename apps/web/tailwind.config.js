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

    theme: {
    extend: {
      colors: {
        primary: 'var(--primary)',
        secondary: 'var(--secondary)',
        bgDark: 'var(--bg-dark)',
        panelDark: 'var(--panel-dark)',
        borderDark: 'var(--border-dark)',
        textMain: 'var(--text-main)',
        textMuted: 'var(--text-muted)',
      },
    },
  },
}