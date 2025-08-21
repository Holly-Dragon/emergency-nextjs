import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic'; // 强制动态执行以支持流式传输

export async function POST(req) {
  try {
    const body = await req.json();
    const backendUrl = process.env.BACKEND_URL || 'http://127.0.0.1:8000';
    const resp = await fetch(`${backendUrl}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
      body: JSON.stringify({
        scene: body.scene || '',
      }),
    });
    if (!resp.ok) {
      const errorText = await resp.text();
      return NextResponse.json({ ok: false, error: '后端错误', details: errorText }, { status: resp.status });
    }
    const stream = new ReadableStream({
      async start(controller) {
        const reader = resp.body.getReader();
        function pump() {
          reader.read().then(({ done, value }) => {
            if (done) return controller.close();
            controller.enqueue(value);
            pump();
          }).catch(err => controller.error(err));
        }
        pump();
      }
    });
    return new Response(stream, { headers: { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' } });
  } catch (err) {
    console.error('内部服务器错误:', err);
    return NextResponse.json({ ok: false, error: String(err) }, { status: 500 });
  }
}
