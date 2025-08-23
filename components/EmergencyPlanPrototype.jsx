"use client";

import React, { useState, useEffect, useRef } from 'react';

export default function EmergencyPlanPrototype() {
  const [screen, setScreen] = useState('home'); // home, progress, result, storage, db
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // 注意：避免在 SSR 阶段使用 Date.now()/toLocaleString() 生成动态内容，
  // 否则会导致 hydration 不一致。先设为空，待客户端 useEffect 再填充。
  const [history, setHistory] = useState([]);

  const [databases, setDatabases] = useState(() => {
    return [
      { id: 1, name: '矿井事故模板库', added: '2025-08-10' },
      { id: 2, name: '化工泄漏知识库', added: '2025-08-12' },
    ];
  });

  const [inputText, setInputText] = useState('南京市中山陵风景区出现踩踏事故，造成50人死亡');
  const [useLocalDB, setUseLocalDB] = useState(true);

  // --- 状态重构 ---
  const [outline, setOutline] = useState([]); // 存储最终的完整大纲
  const [rawGeneratedText, setRawGeneratedText] = useState(''); // 右侧原始输出
  const [statusMessage, setStatusMessage] = useState('等待开始...'); // 顶部状态信息
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentGeneratingSection, setCurrentGeneratingSection] = useState(null); // 当前生成的小节标题
  const [finishedSections, setFinishedSections] = useState(new Set()); // 已完成的小节标题集合

  const abortControllerRef = useRef(null);
  const [resultText, setResultText] = useState('');
  const [navAnchors, setNavAnchors] = useState([]);

  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  // 在客户端填充历史记录，避免 SSR/CSR 时间不一致造成的 hydration 错误
  useEffect(() => {
    const clientHistory = Array.from({ length: 4 }).map((_, i) => ({
      id: i + 1,
      title: `预案记录 ${4 - i}`,
      time: new Date(Date.now() - i * 3600 * 1000).toLocaleString(),
    })).slice(0, 10);
    setHistory(clientHistory);
  }, []);

  const startGeneration = async () => {
    setScreen('progress');
    // --- 重置所有状态 ---
    setOutline([]);
    setRawGeneratedText('');
    setStatusMessage('准备连接...');
    setIsGenerating(true);
    setCurrentGeneratingSection(null);
    setFinishedSections(new Set());
    setResultText('');
    setNavAnchors([]);
    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scene: inputText }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        throw new Error(`API 请求失败: ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';

        for (const part of parts) {
          // if (!part.startsWith('data:')) continue;

          const lines = part.split('\n');
          let eventName = 'message';
          let dataString = '';

          for (const line of lines) {
            if (line.startsWith('event:')) eventName = line.slice(6).trim();
            else if (line.startsWith('data:')) dataString += line.slice(5);
          }
          
          if (!dataString) continue;

          try {
            const payload = JSON.parse(dataString);

            switch (eventName) {
              case 'status_update':
                setStatusMessage(payload.message);
                break;
              
              case 'outline_chunk':
                // 流式更新大纲
                if (payload.outline) {
                  setOutline(payload.outline);
                }
                break;

              case 'outline_finished':
                // 大纲最终确定
                setStatusMessage("大纲生成完毕，准备生成内容...");
                setOutline(payload.outline);
                break;

              case 'chapter_start':
                setRawGeneratedText(prev => prev + payload.title_md);
                break;

              case 'section_start':
                setCurrentGeneratingSection(payload.title);
                setRawGeneratedText(prev => prev + payload.title_md);
                break;
              
              case 'content_chunk':
                setRawGeneratedText(prev => prev + payload);
                break;

              case 'section_end':
                setFinishedSections(prev => new Set(prev).add(payload.title));
                setCurrentGeneratingSection(null);
                break;

              case 'full_content_finished':
                setStatusMessage("✅ 预案生成完毕！");
                setResultText(payload.full_content);
                // 从最终负载中提取导航锚点
                const finalOutline = payload.full_outline.outline || [];
                const anchors = finalOutline.flatMap(c => [c.title, ...c.sections]);
                setNavAnchors(anchors);
                setScreen('result'); // 跳转到结果页
                setIsGenerating(false);
                return; // 提前退出循环

              case 'error':
                throw new Error(payload.error);
            }
          } catch (e) {
            console.error("解析或处理事件时出错:", e, "原始数据:", dataString);
          }
        }
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        setStatusMessage("🛑 用户已停止生成");
      } else {
        setStatusMessage(`❌ 发生错误: ${error.message}`);
      }
    } finally {
      setIsGenerating(false);
      setCurrentGeneratingSection(null);
    }
  };

  const stopGeneration = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  const downloadResult = () => {
    const blob = new Blob([resultText || ''], { type: 'text/plain;charset=utf-8' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `预案_${new Date().toISOString()}.txt`;
    link.click();
  };

  const addDatabase = (name) => {
    const id = Date.now();
    setDatabases((d) => [{ id, name, added: new Date().toISOString().slice(0, 10) }, ...d]);
  };
  const deleteDatabase = (id) => setDatabases((d) => d.filter((x) => x.id !== id));

  // --- 渲染逻辑 ---
  const renderOutlineItem = (item, index) => {
    const isGenerating = currentGeneratingSection === item;
    const isFinished = finishedSections.has(item);
    
    let itemClass = "transition-all duration-300";
    if (isFinished) {
      itemClass += " text-gray-400 line-through";
    } else if (isGenerating) {
      itemClass += " text-purple-600 font-bold animate-pulse";
    } else {
      itemClass += " text-gray-600";
    }

    return <li key={index} className={itemClass}>{item}</li>;
  };

  return (
    <div className="min-h-screen font-sans" style={{ fontFamily: 'Noto Sans SC, system-ui, -apple-system, \"PingFang SC\", \"Hiragino Sans GB\", \"Microsoft Yahei\", sans-serif' }}>
      <div className="flex h-screen bg-gradient-to-b from-white to-white">
        <aside className={`transition-all duration-200 bg-white/70 backdrop-blur p-4 flex flex-col ${sidebarCollapsed ? 'w-16' : 'w-72'}`}>
          <div className="flex items-center justify-between mb-4 flex-shrink-0">
            {!sidebarCollapsed && <h2 className="text-lg font-semibold text-[#6326a2]">工具栏</h2>}
            <button className="text-sm px-2 py-1 border rounded-md" onClick={() => setSidebarCollapsed((s) => !s)}>{sidebarCollapsed ? '▶' : '◀'}</button>
          </div>

          {!sidebarCollapsed && (
            <div className="flex flex-col flex-grow h-0">
              <div className="mb-4 flex flex-col flex-grow h-0">
                <h3 className="text-sm text-gray-600 flex-shrink-0">历史记录</h3>
                <div className="relative flex-grow mt-2">
                  <ul className="absolute inset-0 overflow-auto space-y-2 pr-1">
                    {history.map((h) => (
                      <li key={h.id} className="text-sm p-2 bg-purple-50 rounded-md cursor-pointer hover:bg-purple-100" onClick={() => { setScreen('result'); setResultText(`来自历史：${h.title}\\n（示例内容）`); }}>
                        <div className="font-medium">{h.title}</div>
                        <div className="text-xs text-gray-500">{h.time}</div>
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="mt-2 text-right flex-shrink-0">
                  <button className="text-xs underline" onClick={() => setScreen('storage')}>查看全部存储</button>
                </div>
              </div>

              <div className="flex-shrink-0">
                <h3 className="text-sm text-gray-600">自建数据库</h3>
                <div className="mt-2 space-y-2">
                  <button className="w-full text-left p-2 bg-white border rounded-md hover:bg-purple-50" onClick={() => setScreen('db')}>打开数据库管理</button>
                </div>
              </div>
            </div>
          )}

          {sidebarCollapsed && (
            <div className="flex flex-col items-center mt-6 space-y-4">
              <button title="历史记录" onClick={() => setScreen('storage')}>📜</button>
              <button title="数据库" onClick={() => setScreen('db')}>🗄️</button>
            </div>
          )}
        </aside>

        <main className="flex-1 p-6 bg-[linear-gradient(90deg,#F7F4FF,white)] flex flex-col">
          <header className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-2xl font-bold text-[#6326a2]">应急预案生成系统</h1>
            </div>
            <nav className="space-x-3">
              <button className={`px-3 py-1 rounded ${screen === 'home' ? 'bg-[#7429c6] text-white' : 'bg-white border'}`} onClick={() => setScreen('home')}>初始界面</button>
              <button className={`px-3 py-1 rounded ${screen === 'progress' ? 'bg-[#7429c6] text-white' : 'bg-white border'}`} onClick={() => setScreen('progress')}>生成过程</button>
              <button className={`px-3 py-1 rounded ${screen === 'result' ? 'bg-[#7429c6] text-white' : 'bg-white border'}`} onClick={() => setScreen('result')}>生成结果</button>
              <button className={`px-3 py-1 rounded ${screen === 'storage' ? 'bg-[#7429c6] text-white' : 'bg-white border'}`} onClick={() => setScreen('storage')}>预案存储</button>
              <button className={`px-3 py-1 rounded ${screen === 'db' ? 'bg-[#7429c6] text-white' : 'bg-white border'}`} onClick={() => setScreen('db')}>数据库</button>
            </nav>
          </header>

          <section className="bg-white rounded-xl p-6 shadow-sm flex-1">
            {screen === 'home' && (
              <div className="grid grid-cols-2 gap-6 h-full">
                <div className="border rounded-lg p-4 flex flex-col">
                  <h3 className="text-lg font-medium text-[#6326a2] mb-2">输入突发事件场景</h3>
                  <textarea className="flex-1 p-3 border rounded-md resize-none" value={inputText} onChange={(e) => setInputText(e.target.value)} />
                  <div className="mt-3 flex items-center justify-between">
                    <label className="flex items-center gap-2"><input type="checkbox" checked={useLocalDB} onChange={(e) => setUseLocalDB(e.target.checked)} /> 使用本地数据库</label>
                    <button className="px-5 py-2 bg-[#7429c6] text-white rounded-md" onClick={startGeneration}>生成预案</button>
                  </div>
                </div>

                <div className="border rounded-lg p-4">
                  <h3 className="text-lg font-medium text-[#6326a2] mb-2">提示与说明</h3>
                  <ul className="list-disc ml-5 text-sm text-gray-600 space-y-2">
                    <li>场景描述越详尽，生成结果越精确。</li>
                    <li>可选择是否使用本地数据库进行 RAG 检索。</li>
                    <li>历史记录最多显示 10 条，点击可快速查看。</li>
                    <li>点击“生成预案”进入生成进度页面，可随时停止。</li>
                  </ul>
                </div>
              </div>
            )}

            {screen === 'progress' && (
              <div className="grid grid-cols-4 gap-6 h-full">
                <aside className="col-span-1 bg-purple-50 p-4 rounded-lg flex flex-col">
                  <h4 className="font-medium text-[#6326a2] mb-3 flex-shrink-0">任务进度</h4>
                  <div className="overflow-auto flex-grow">
                    <ol className="space-y-3 text-sm pr-2">
                      {outline.length === 0 && <li className="text-gray-500">{statusMessage}</li>}
                      {outline.map((item, index) => (
                        <li key={index} className="p-2 rounded bg-white shadow-sm">
                          <div className="font-semibold text-[#6326a2]">{item.title}</div>
                          {item.sections && item.sections.length > 0 && (
                            <ul className="mt-1 pl-4 list-disc list-inside">
                              {item.sections.map((section, sIndex) => renderOutlineItem(section, `${index}-${sIndex}`))}
                            </ul>
                          )}
                        </li>
                      ))}
                    </ol>
                  </div>
                </aside>

                <main className="col-span-3 bg-white p-4 rounded-lg flex flex-col">
                  <div className="flex items-center justify-between mb-3 flex-shrink-0">
                    <h4 className="font-medium text-[#6326a2]">生成工作流 ({statusMessage})</h4>
                    <div className="space-x-2">
                      {!isGenerating && <button className="px-3 py-1 border rounded" onClick={startGeneration}>重新开始</button>}
                      {isGenerating && <button className="px-3 py-1 bg-red-500 text-white rounded" onClick={stopGeneration}>停止</button>}
                    </div>
                  </div>

                  <div className="flex-grow h-0 overflow-auto border rounded p-3 bg-gray-50 font-mono text-xs">
                      {rawGeneratedText ? <pre className="whitespace-pre-wrap break-words">{rawGeneratedText}</pre> : <div className="text-gray-400">暂无日志，点击开始以生成。</div>}
                  </div>

                  <div className="mt-4 text-right text-sm text-gray-500 flex-shrink-0">注：此区为后端实时输出的原始数据流。</div>
                </main>
              </div>
            )}

            {screen === 'result' && (
              <div className="grid grid-cols-4 gap-6 h-full">
                <nav className="col-span-1 bg-white p-4 rounded-lg sticky top-6 h-full overflow-auto border">
                  <h4 className="font-medium text-[#6326a2] mb-3">内容导航</h4>
                  <ul className="space-y-2 text-sm">
                    {navAnchors.length === 0 && <li className="text-gray-400">暂无目录</li>}
                    {navAnchors.map((a, i) => (
                      <li key={i} className="cursor-pointer p-2 hover:bg-purple-50 rounded" onClick={() => alert(`跳转到：${a}`)}>{a}</li>
                    ))}
                  </ul>
                </nav>

                <article className="col-span-3 bg-white p-4 rounded-lg flex flex-col">
                  <div className="flex items-start justify-between flex-shrink-0">
                    <h4 className="font-medium text-[#6326a2]">预案文本（可编辑）</h4>
                    <div className="space-x-2">
                      <button className="px-3 py-1 border rounded" onClick={() => { alert('保存到存储（示例）'); setHistory((h) => [{ id: Date.now(), title: `预案 - ${new Date().toLocaleString()}`, time: new Date().toLocaleString() }, ...h].slice(0, 10)); }}>保存</button>
                      <button className="px-3 py-1 border rounded" onClick={downloadResult}>下载</button>
                    </div>
                  </div>

                  <textarea className="w-full flex-grow h-0 mt-3 p-3 border rounded resize-none" value={resultText} onChange={(e) => setResultText(e.target.value)} />

                  <div className="mt-3 text-xs text-gray-500 flex-shrink-0">提示：编辑区支持基本的文本修改，保存后会进入“预案存储”列表。</div>
                </article>
              </div>
            )}

            {screen === 'storage' && (
              <div>
                <h3 className="text-lg font-medium text-[#6326a2] mb-3">按时间展示的预案存储</h3>

                {(() => {
                  const groups = history.reduce((acc, item) => {
                    const d = new Date(item.time).toISOString().slice(0, 10);
                    if (!acc[d]) acc[d] = [];
                    acc[d].push(item);
                    return acc;
                  }, {});

                  const sortedDates = Object.keys(groups).sort((a, b) => b.localeCompare(a));

                  if (sortedDates.length === 0) return <div className="text-gray-400">暂无存储记录</div>;

                  return (
                    <div>
                      {sortedDates.map((date) => (
                        <div key={date} className="mb-6">
                          <div className="text-sm text-gray-500 font-medium mb-2">{date}</div>

                          <div className="grid grid-cols-5 gap-4">
                            {groups[date].map((h) => (
                              <div key={h.id} className="bg-white p-4 rounded-lg border shadow-sm">
                                <div className="font-medium truncate">{h.title}</div>
                                <div className="text-xs text-gray-500">{h.time}</div>
                                <p className="mt-2 text-sm text-gray-700">示例预案摘要：快速响应流程、资源调配、通讯联络人...</p>
                                <div className="mt-3 flex space-x-2">
                                  <button className="px-3 py-1 border rounded" onClick={() => { setResultText(`来自存储：${h.title}\n（示例内容）`); setScreen('result'); }}>查看</button>
                                  <button className="px-3 py-1 border rounded" onClick={() => setHistory((prev) => prev.filter(x => x.id !== h.id))}>删除</button>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  );
                })()}
              </div>
            )}

            {screen === 'db' && (
              <div>
                <h3 className="text-lg font-medium text-[#6326a2] mb-3">用户自建数据库管理</h3>
                <div className="flex items-center gap-3 mb-4">
                  <input id="new-db-name" placeholder="输入数据库名称" className="border rounded px-3 py-2 flex-1" />
                  <button className="px-4 py-2 bg-[#7429c6] text-white rounded" onClick={() => {
                    const el = document.getElementById('new-db-name');
                    if (el && el.value.trim()) { addDatabase(el.value.trim()); el.value = ''; }
                  }}>新增数据库</button>
                </div>

                <div className="flex overflow-x-auto space-x-4 py-2">
                  {databases.length === 0 && <div className="text-gray-400">暂无数据库</div>}
                  {databases.map((d) => (
                    <div key={d.id} className="min-w-[260px] bg-white p-4 rounded-lg border shadow-sm">
                      <div className="font-medium">{d.name}</div>
                      <div className="text-xs text-gray-500">添加时间：{d.added}</div>
                      <p className="mt-2 text-sm text-gray-700">说明：用于 RAG 检索的本地知识集合。</p>
                      <div className="mt-3 flex space-x-2">
                        <button className="px-3 py-1 border rounded" onClick={() => alert('示例：打开数据库详情')}>打开</button>
                        <button className="px-3 py-1 border rounded" onClick={() => deleteDatabase(d.id)}>删除</button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

          </section>
        </main>
      </div>
    </div>
  );
}
