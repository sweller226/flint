/** @type {import('next').NextConfig} */
const nextConfig = {
    reactStrictMode: true,
    transpilePackages: ['@flint/ui', '@flint/config'],
    images: {
        unoptimized: true,
    },
    output: 'export',
    assetPrefix: process.env.NODE_ENV === 'production' ? './' : undefined,
}

module.exports = nextConfig
