import '../styles/globals.css'

export const metadata = {
  title: '应急预案生成系统（原型）',
  description: '交互原型 - Next.js 示例（含 Tailwind）',
}

export default function RootLayout({ children }) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  )
}
