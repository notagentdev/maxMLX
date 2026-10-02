import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { browserAvailable, withBrowser } from './harness.mjs';

test('chat persists conversations and their settings, and works like omlx', { skip: !browserAvailable, timeout: 90000 }, async () => {
  const page = await readFile(new URL('../../python/maxtoken/server/console.html', import.meta.url));
  const store = new Map();        // the server's chat files
  let profiles = [{ name: 'System Default', content: '' }];
  const completions = [];          // request bodies of /v1/chat/completions
  const calls = [];                // every other request, for assertions
  let reply = 0;

  await withBrowser(async (req, res) => {
    const url = new URL(req.url, 'http://x');
    let raw = '';
    for await (const chunk of req) raw += chunk;
    const body = raw ? JSON.parse(raw) : {};
    const json = (data, status = 200) => { res.statusCode = status; res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify(data)); };
    if (url.pathname === '/console') { res.setHeader('Content-Type', 'text/html'); return res.end(page); }
    if (url.pathname === '/health') return json({ status: 'ok', model: 'fixture-model', maintenance: 'serving', uptime_s: 5, version: 'test' });
    if (url.pathname === '/admin/stats') return json({ uptime_s: 5, model: { id: 'fixture-model', ctx: 32768 }, requests: {}, throughput: {} });
    if (url.pathname === '/v1/chat/completions') {
      completions.push(body);
      const last = body.messages.at(-1).content;
      res.setHeader('Content-Type', 'text/event-stream');
      if (last.startsWith('slow')) {
        res.write('data: {"choices":[{"delta":{"content":"partial answer"}}]}\n\n');
        await new Promise(resolve => { res.on('close', resolve); setTimeout(resolve, 8000); });
        return res.end();
      }
      const n = ++reply;
      const long = Array.from({ length: 30 }, (_, i) => `Paragraph ${i} of a long answer.`).join('\n\n');
      const text = last.startsWith('long') ? long : last.startsWith('markup') ? 'Click [here](javascript:alert(1)) <img src=x onerror=window.pwned=1> **bold**' : `Reply ${n} to ${last}`;
      res.write(`data: ${JSON.stringify({ choices: [{ delta: { reasoning_content: `thinking ${n}` } }] })}\n\n`);
      await new Promise(resolve => setTimeout(resolve, 60));
      res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: text } }] })}\n\n`);
      res.write(`data: ${JSON.stringify({ usage: { prompt_tokens: 50 * n, completion_tokens: 20 } })}\n\ndata: [DONE]\n\n`);
      return res.end();
    }
    if (url.pathname === '/admin/chat-profiles') {
      if (req.method === 'PUT') { profiles = body.profiles; calls.push({ method: 'PUT', path: url.pathname, body }); }
      return json({ profiles });
    }
    if (url.pathname === '/admin/chats') {
      calls.push({ method: req.method, path: url.pathname, query: url.search });
      if (req.method === 'DELETE') { const n = store.size; store.clear(); return json({ deleted: n }); }
      const q = (url.searchParams.get('q') || '').toLowerCase();
      const chats = [...store.values()].filter(d => !q || JSON.stringify(d).toLowerCase().includes(q))
        .map(d => ({ id: d.id, title: d.title, pinned: !!d.pinned, manualTitle: !!d.manualTitle, updatedAt: d.updatedAt, model: d.model, messageCount: d.messages.length }));
      return json({ chats });
    }
    const match = url.pathname.match(/^\/admin\/chats\/([\w-]+)$/);
    if (match) {
      const id = match[1];
      calls.push({ method: req.method, path: url.pathname, body });
      if (req.method === 'PUT') { store.set(id, body); return json({ id }); }
      if (req.method === 'DELETE') return store.delete(id) ? json({ success: true }) : json({ detail: 'Chat not found' }, 404);
      return store.has(id) ? json(store.get(id)) : json({ detail: 'Chat not found' }, 404);
    }
    json({});
  }, async ({ base, evaluate, waitFor, command, errors }) => {
    const type = (id, value) => evaluate(`(() => { const e = document.getElementById("${id}"); e.value = ${JSON.stringify(value)}; e.dispatchEvent(new Event("input", { bubbles: true })); })()`);
    const send = async text => { await type('input', text); await evaluate('document.getElementById("sendBtn").click()'); };
    const settled = n => waitFor(`!document.getElementById("sendBtn").disabled && document.querySelectorAll("#messages .msg.assistant").length >= ${n}`);
    const saved = id => store.get(id);
    const current = () => evaluate('chat.cur.id');
    // Polls the fake server's files: saving is asynchronous and debounced.
    const until = async (predicate, label) => {
      for (let i = 0; i < 100; i++) { if (predicate()) return; await new Promise(resolve => setTimeout(resolve, 50)); }
      assert.fail('Timed out waiting for ' + label);
    };
    const reload = async () => {
      await evaluate('window.__stale = true');
      await command('Page.reload');
      await waitFor('!window.__stale && typeof chat !== "undefined" && chat.cur');
    };
    const lastAssistant = '[...document.querySelectorAll(".msg.assistant")].at(-1)';

    await command('Page.navigate', { url: base + '/console#chat' });
    await waitFor('typeof chat !== "undefined" && chat.cur');

    // The system prompt is not part of the conversation any more: it lives in the settings panel.
    assert.equal(await evaluate('document.querySelector("details.prompt-panel")'), null);
    assert.equal(await evaluate('!!document.getElementById("sysPrompt").closest("#chatPanel") && !document.getElementById("sysPrompt").closest("#chatMain")'), true);
    assert.equal(await evaluate('document.getElementById("chatEmpty").hidden'), false);
    assert.equal(store.size, 0, 'an empty draft is not saved');

    // First message: title, history, saved document, reasoning, stats.
    await send('Hello');
    await settled(1);
    const id1 = await current();
    assert.deepEqual(completions[0].messages, [{ role: 'user', content: 'Hello' }]);
    assert.equal(completions[0].stream, true);
    assert.equal(completions[0].temperature, undefined);
    assert.equal(await evaluate('document.querySelector(".msg.assistant .mbody").textContent'), 'Reply 1 to Hello');
    assert.equal(await evaluate('document.querySelector(".msg.assistant .think").hidden'), false);
    assert.equal(await evaluate('document.querySelector(".msg.assistant .think").classList.contains("open")'), false, 'reasoning is collapsed by default');
    assert.match(await evaluate('document.querySelector(".msg.assistant .mmeta").textContent'), /20 tok/);
    assert.equal(await evaluate('document.querySelector(".chat-row.active .ctitle").textContent'), 'Hello');
    await until(() => saved(id1)?.messages.length === 2 && saved(id1).messages[1].stats, 'the first exchange to be saved');
    assert.equal(saved(id1).title, 'Hello');
    assert.deepEqual(saved(id1).messages.map(m => m.role), ['user', 'assistant']);
    assert.equal(saved(id1).messages[1].thinking, 'thinking 1');
    assert.equal(saved(id1).messages[1].stats.prompt_tokens, 50);
    assert.equal(saved(id1).contextTokens, 70);
    assert.equal(JSON.stringify(saved(id1)).includes('"_'), false, 'transient fields are never stored');
    await waitFor('!document.getElementById("ctxCap").hidden');
    assert.match(await evaluate('document.getElementById("ctxText").textContent'), /70 \/ 32\.8K/);
    assert.equal(await evaluate('document.getElementById("pfTokens").textContent'), '50 / 20');

    // Settings and the system prompt belong to the chat, are sent with every request and are saved.
    await type('csTemp', '0.3'); await type('csTopK', '40'); await type('csThink', 'off');
    await evaluate('document.querySelector("[data-tab=profile]").click()');
    await type('sysPrompt', 'Be terse');
    await until(() => saved(id1).systemPrompt === 'Be terse' && saved(id1).settings.top_k === 40, 'the settings to be saved');
    await send('Second');
    await settled(2);
    assert.deepEqual(completions[1].messages, [
      { role: 'system', content: 'Be terse' }, { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Reply 1 to Hello' }, { role: 'user', content: 'Second' }]);
    assert.equal(completions[1].temperature, 0.3);
    assert.equal(completions[1].top_k, 40);
    assert.deepEqual(completions[1].thinking, { type: 'disabled' });
    assert.equal(completions[1].top_p, undefined);
    assert.deepEqual(saved(id1).settings, { temperature: 0.3, top_k: 40, thinking: 'off' });
    assert.equal(saved(id1).systemPrompt, 'Be terse');

    // Regenerate adds a variant; history uses the answer that is shown.
    await evaluate('document.querySelectorAll(".msg.assistant")[1].querySelector("[aria-label=Regenerate]").click()');
    await waitFor('!document.getElementById("sendBtn").disabled && document.querySelectorAll(".variant").length === 2');
    assert.deepEqual(completions[2].messages.map(m => m.content), ['Be terse', 'Hello', 'Reply 1 to Hello', 'Second']);
    assert.equal(await evaluate('document.querySelectorAll(".msg.assistant")[1].querySelector(".mbody").textContent'), 'Reply 3 to Second');
    await evaluate('document.querySelector(".variant button").click()');
    assert.equal(await evaluate('document.querySelectorAll(".msg.assistant")[1].querySelector(".mbody").textContent'), 'Reply 2 to Second');
    await send('Third');
    await settled(3);
    assert.deepEqual(completions[3].messages.map(m => m.content).slice(-3), ['Second', 'Reply 2 to Second', 'Third'], 'the selected response is the history');

    // Edit rewrites the conversation from that message on.
    await evaluate('document.querySelectorAll(".msg.user")[1].querySelector("[aria-label=Edit]").click()');
    await evaluate('(() => { const t = document.querySelector(".msg.user.editing textarea"); t.value = "Second, reworded"; t.dispatchEvent(new Event("input")); })()');
    await evaluate('[...document.querySelectorAll(".msg.user.editing .btn")].find(b => b.textContent.includes("regenerate")).click()');
    await waitFor('!document.getElementById("sendBtn").disabled && document.querySelectorAll(".msg.user").length === 2 && document.querySelectorAll(".msg.assistant").length === 2');
    assert.equal(completions.at(-1).messages.at(-1).content, 'Second, reworded');
    assert.equal(await evaluate('document.querySelectorAll(".variant").length'), 0);

    // Branching copies the visible conversation into a new chat.
    await evaluate('document.querySelectorAll(".msg.assistant")[0].querySelector("[aria-label=\\"More actions\\"]").click()');
    await evaluate('[...document.querySelectorAll(".menu button")].find(b => b.textContent.includes("Branch")).click()');
    await waitFor(`chat.cur.id !== "${id1}"`);
    const branch = await current();
    assert.equal(await evaluate('chat.cur.title'), 'Branch of Hello');
    assert.deepEqual(await evaluate('chat.cur.messages.map(m => m.role)'), ['user', 'assistant']);
    assert.equal(await evaluate('chat.cur.settings.temperature'), 0.3);
    await waitFor('chat.list.length === 2');
    assert.equal(saved(branch).manualTitle, true);

    // A reload restores everything from the server: the list, the open chat, its settings.
    await reload();
    await waitFor('chat.list.length === 2 && document.querySelectorAll("#messages .msg").length === 2');
    assert.equal(await current(), branch, 'the chat that was open is open again');
    assert.equal(await evaluate('document.getElementById("csTemp").value'), '0.3');
    assert.equal(await evaluate('document.getElementById("csThink").value'), 'off');
    assert.equal(await evaluate('document.getElementById("sysPrompt").value'), 'Be terse');
    assert.equal(await evaluate('document.querySelector(".msg.assistant .mbody").textContent'), 'Reply 1 to Hello');
    await evaluate(`openChat("${id1}")`);
    await waitFor(`document.querySelectorAll("#messages .msg").length === 4`);
    assert.equal(await evaluate('document.querySelectorAll(".msg.user")[1].textContent.includes("Second, reworded")'), true);

    // List actions: pin moves a chat to the top; delete needs a second click; search asks the server.
    await evaluate(`document.querySelector('.chat-row[data-id="${branch}"] [aria-label="Pin chat"]').click()`);
    await until(() => saved(branch).pinned === true, 'the pin to be saved');
    assert.equal(await evaluate('chat.list[0].id'), branch);
    await evaluate(`document.querySelector('.chat-row[data-id="${branch}"] [aria-label="Rename chat"]').click()`);
    await evaluate('(() => { const i = document.querySelector(".chat-row input"); i.value = "Renamed"; i.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true })); })()');
    await until(() => saved(branch).title === 'Renamed', 'the new title to be saved');
    assert.equal(saved(branch).manualTitle, true);
    await type('chatSearch', 'reworded');
    await waitFor('document.querySelectorAll(".chat-row").length === 1');
    assert.equal(await evaluate('document.querySelector(".chat-row .ctitle").textContent'), 'Hello');
    await type('chatSearch', 'zzz-nothing');
    await waitFor('document.querySelector(".chat-empty")?.textContent.includes("No chats match")');
    await type('chatSearch', '');
    await waitFor('document.querySelectorAll(".chat-row").length === 2');
    await evaluate(`document.querySelector('.chat-row[data-id="${branch}"] [aria-label="Delete chat"]').click()`);
    assert.equal(store.has(branch), true, 'one click only arms the delete');
    await evaluate(`document.querySelector('.chat-row[data-id="${branch}"] .armed').click()`);
    await until(() => !store.has(branch), 'the chat to be deleted');
    assert.equal(await evaluate('chat.list.length'), 1);

    // Profiles: the active one fills the system prompt; Save writes the text back to it.
    await evaluate('document.querySelector("[data-tab=profile]").click(); document.getElementById("profAdd").click()');
    await waitFor('chat.profiles.length === 2 && document.querySelector(".prof input")');
    await evaluate('(() => { const i = document.querySelector(".prof input"); i.value = "Coder"; i.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true })); })()');
    await waitFor('chat.profiles.some(p => p.name === "Coder") && !document.querySelector(".prof input")');
    await type('sysPrompt', 'Write tests first.');
    await evaluate('document.getElementById("profSave").click()');
    await waitFor(`chat.profiles.find(p => p.name === "Coder").content === "Write tests first."`);
    assert.equal(profiles.find(p => p.name === 'Coder').content, 'Write tests first.');
    assert.equal(await evaluate('chat.cur.activeProfile'), 'Coder');

    // A new chat starts from the default profile and does not inherit the other chat's settings.
    await evaluate('document.getElementById("newChatBtn").click()');
    assert.equal(await evaluate('chat.cur.settings && Object.keys(chat.cur.settings).length'), 0);
    assert.equal(await evaluate('document.getElementById("sysPrompt").value'), '');
    assert.equal(await evaluate('document.getElementById("csProfile").value'), 'System Default');

    // A document without the optional fields (hand-made, imported, older) still opens and works.
    store.set('chat_legacy', { id: 'chat_legacy', title: 'Imported', updatedAt: '2026-01-01T00:00:00+00:00', messages: [{ role: 'user', content: 'old question' }, { role: 'assistant', content: 'old answer' }] });
    await evaluate('loadChatList()');
    await evaluate('localStorage.chatCurrent = "chat_legacy"');
    await reload();
    assert.equal(await current(), 'chat_legacy', 'the remembered chat is restored even without optional fields');
    assert.equal(await evaluate('document.querySelector(".msg.assistant .mbody").textContent'), 'old answer');
    assert.deepEqual(await evaluate('chat.cur.settings'), {});
    await type('csTemp', '0.9');
    await until(() => saved('chat_legacy').settings?.temperature === 0.9, 'the legacy chat to save its settings');
    store.delete('chat_legacy');
    await evaluate('newChat()');

    // Remote text never becomes markup.
    await send('markup please');
    await settled(1);
    assert.equal(await evaluate('window.pwned'), undefined);
    assert.equal(await evaluate('document.querySelectorAll("#messages img").length'), 0);
    assert.equal(await evaluate('document.querySelectorAll("#messages a").length'), 0, 'javascript: links are not links');
    assert.equal(await evaluate('document.querySelector(".msg.assistant .mbody strong").textContent'), 'bold');

    // Timeline: one dot per question once the thread scrolls; hover names it, click jumps to it.
    assert.equal(await evaluate('document.getElementById("timeline").hidden'), true, 'a short thread needs no rail');
    await send('long answer please');
    await settled(2);
    await waitFor('!document.getElementById("timeline").hidden && document.querySelectorAll(".tl-dot").length === 2');
    assert.equal(await evaluate('document.querySelector(".tl-dot.active") === document.querySelectorAll(".tl-dot")[1]'), true, 'the latest question is current at the bottom');
    await evaluate('document.querySelectorAll(".tl-dot")[0].dispatchEvent(new MouseEvent("mouseenter"))');
    assert.equal(await evaluate('document.getElementById("tlTip").hidden'), false);
    assert.equal(await evaluate('document.getElementById("tlTip").textContent'), 'markup please');
    await evaluate('document.querySelectorAll(".tl-dot")[0].click()');
    await waitFor('document.getElementById("messages").scrollTop < 40');
    await waitFor('document.querySelectorAll(".tl-dot")[0].classList.contains("active")');
    await evaluate('document.querySelectorAll(".tl-dot")[0].dispatchEvent(new MouseEvent("mouseleave"))');
    assert.equal(await evaluate('document.getElementById("tlTip").hidden'), true);

    // Stopping keeps what was generated.
    await send('slow one');
    await waitFor(`${lastAssistant}?.querySelector(".mbody").textContent === "partial answer" && !document.getElementById("stopBtn").disabled`);
    await evaluate('document.getElementById("stopBtn").click()');
    await waitFor('!document.getElementById("sendBtn").disabled');
    assert.match(await evaluate(`${lastAssistant}.querySelector(".mmeta").textContent`), /stopped/);
    const stoppedId = await current();
    await until(() => saved(stoppedId)?.messages.at(-1)?.stopped === true, 'the stopped answer to be saved');
    assert.equal(saved(stoppedId).messages.at(-1).content, 'partial answer');

    // History tab: export is a file, delete-all needs a second click.
    await evaluate('document.querySelector("[data-tab=chat]").click(); document.getElementById("deleteAllChats").click()');
    assert.equal(store.size, 2, 'one click only arms delete-all');
    await evaluate('document.getElementById("deleteAllChats").click()');
    await waitFor('chat.list.length === 0 && document.getElementById("chatEmpty").hidden === false');
    assert.equal(store.size, 0);
    assert.ok(calls.some(c => c.method === 'DELETE' && c.query === '?all=true'));

    // Narrow screens: both side panels stay out of the way.
    await command('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
    await reload();
    assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth'), true, 'chat must not overflow on mobile');
    assert.equal(await evaluate('getComputedStyle(document.getElementById("chatSide")).display'), 'none');
    assert.equal(await evaluate('getComputedStyle(document.getElementById("chatPanel")).display'), 'none');
    await evaluate('document.getElementById("togglePanel").click()');
    assert.equal(await evaluate('getComputedStyle(document.getElementById("chatPanel")).display'), 'flex');
    assert.deepEqual(errors, []);
  });
});
