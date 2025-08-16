/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    appDir: true,
  },
  typescript: {
    // 生产环境下也进行类型检查
    ignoreBuildErrors: false,
  },
  eslint: {
    // 生产环境下也进行 ESLint 检查
    ignoreDuringBuilds: false,
  },
  // API 路由重写，支持后端服务代理
  async rewrites() {
    return [
      {
        source: '/api/backend/:path*',
        destination: process.env.BACKEND_URL 
          ? `${process.env.BACKEND_URL}/:path*`
          : 'http://localhost:8000/:path*',
      },
    ];
  },
  // 环境变量配置
  env: {
    BACKEND_URL: process.env.BACKEND_URL || 'http://localhost:8000',
    WS_URL: process.env.WS_URL || 'ws://localhost:8000',
    JWT_SECRET: process.env.JWT_SECRET || 'dipmaster-secret-key',
  },
  // 图片优化配置
  images: {
    domains: ['localhost'],
    formats: ['image/webp', 'image/avif'],
  },
  // 性能优化
  swcMinify: true,
  // 压缩配置
  compress: true,
  // 产品模式优化
  productionBrowserSourceMaps: false,
};

module.exports = nextConfig;