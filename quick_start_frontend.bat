@echo off
echo 🚀 启动DipMaster前端仪表板...
echo.

cd /d "G:\Github\Quant\DipMaster-Trading-System\frontend"

echo 📦 检查Node.js环境...
node --version
npm --version

echo.
echo 📦 安装依赖(如果需要)...
if not exist "node_modules" (
    echo 正在安装前端依赖...
    npm install --legacy-peer-deps
) else (
    echo 依赖已存在，跳过安装
)

echo.
echo 🎯 启动Next.js开发服务器...
echo 📍 前端地址: http://localhost:3000
echo 💡 按 Ctrl+C 停止服务器
echo.

npm run dev