# Emergency Backend (FastAPI + LangChain Skeleton)

本目录提供一个最小可运行的 FastAPI 后端，包含：
- 大纲生成接口 `/api/outline`
- 章节生成接口 `/api/section`
- 一次性完整生成 `/api/full`
- 流式 SSE 生成 `/api/stream`

目前默认使用 Mock 逻辑（不依赖真实模型）。后续可将 `_mock_outline`、`_mock_write_section` 换成 LangChain + 你选择的模型供应商。

## 本地运行

1. 创建并激活 Python 环境（任选其一）

```bash
# Python venv 示例
python3 -m venv .venv
source .venv/bin/activate
source /Users/wangqian/codes/emergency-nextjs/.venv/bin/activate

# 或使用 conda
# conda create -n emergency-backend python=3.11 -y
# conda activate emergency-backend
```

2. 安装依赖

```bash
pip install -r backend/requirements.txt
```

3. 配置环境变量

```bash
cp backend/.env.example backend/.env
# 编辑 backend/.env 设置模型与端口
```

4. 启动服务

```bash
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

5. 自检
- http://127.0.0.1:8000/healthz 应返回 `{ "status": "ok" }`
- POST http://127.0.0.1:8000/api/outline

```json
{
  "topic": "校园地震应急预案",
  "audience": "校方与师生",
  "tone": "professional",
  "format": "plan",
  "max_sections": 3
}
```

## 与前端对接

- 推荐在 Next.js 的 `app/api/generate/route.js` 中转发到 `http://127.0.0.1:8000/api/stream`，并在前端采用 EventSource/ReadableStream 渲染。
- 你也可以直接从组件里 fetch 后端接口（注意 CORS）。

## 切换到真实 LLM

- 在 `requirements.txt` 安装相应 SDK，并在 `app.py` 中替换 `_mock_outline` 与 `_mock_write_section`。
- 可以使用 LangChain PromptTemplate + LLM + Parser 的组合，或直接调用“扣子”平台的工作流 HTTP API。
