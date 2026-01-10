/** @type {import('next').NextConfig} */
const nextConfig = {
    reactStrictMode: true,
    transpilePackages: ['@flint/ui', '@flint/config'],
    images: {
        unoptimized: true,
    },
}

module.exports = nextConfig
