# 应急预案生成系统 - Next.js 示例项目

此项目为你提供的 React 原型的 Next.js (App Router) 示例实现，包含：
- 前端页面（components/EmergencyPlanPrototype.jsx）直接来源于你的画布原型；
- Tailwind CSS 基本配置（已在项目中预配置）；
- 服务端 API 路由示例（/app/api/generate/route.js）用于将请求代理到扣子（Kouzi）无代码工作流 API，避免在浏览器暴露密钥；
- 使用说明与运行步骤。

## 快速开始（macOS / VSCode）

1. 安装 Node.js（建议使用 nvm 安装 LTS）
2. 在项目目录中安装依赖：
```bash
cd emergency-nextjs
npm install
```

3. 配置环境变量（在项目根创建 `.env.local`）：
```
KOUZI_API_URL=https://api.your-kouzi-provider.example
KOUZI_API_KEY=your_kouzi_api_key_here
```

4. 运行开发服务器：
```bash
npm run dev
```
打开浏览器并访问 `http://localhost:3000`

## 说明
- 前端发起生成请求时，建议通过 Next.js 的 API 路由 `/api/generate` 转发给扣子（Kouzi），以便在服务端安全地使用 API Key。你可以按需修改 `app/api/generate/route.js` 中对外部 API 的调用细节（URL、请求体字段等）。
- 项目已包含 Tailwind 的基础配置。若你希望更改样式或引入字体（Noto Sans SC），可在 `styles/globals.css` 中编辑。

