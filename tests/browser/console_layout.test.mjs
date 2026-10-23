import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { access, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { once } from 'node:events';
import test from 'node:test';

// Uses an existing Chromium browser and Node's native CDP WebSocket client.
// Override MAXMLX_BROWSER_EXECUTABLE for Chrome/Chromium on other systems.
const executable = process.env.MAXMLX_BROWSER_EXECUTABLE
  || '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser';
const available = await access(executable).then(() => true, () => false);

test('dashboard, settings and chat work in a real browser', { skip: !available, timeout: 60000 }, async () => {
  const page = await readFile(new URL('../../python/maxmlx/server/console.html', import.meta.url));
  const mutations = [];
  let serving = true, failSettings = false, delaySettings = 0;
  const values = {temperature: 0.7, top_k: 20, top_p: 0.95, max_output_tokens: 4096, max_reasoning_tokens: 0, thinking: 'model', max_seq_len_override: 16384, moe_backend: 'offload', moe_cache_size: 128, moe_cache_rate: null, moe_cache_auto: false, memory_ratio: 0.9, max_running_req: 4, max_extend_tokens: 8192, draft_model: null, draft_tokens: 3, cache_type: 'radix', prefix_cache_dir: '~/.maxmlx/prefix-cache', prefix_cache_disk_gb: 20, enable_cache_report: true, served_model_name: 'fixture-model', tool_call_parser: 'qwen25', reasoning_parser: 'qwen3', server_host: '127.0.0.1', server_port: 1919};
  const active = {...values};
  const live = new Set(['temperature', 'top_k', 'top_p', 'max_output_tokens', 'max_reasoning_tokens', 'thinking', 'max_seq_len_override', 'moe_cache_size']);
  const settingsDoc = () => ({model: '/fixture/model', settings_file: '/fixture/settings.json', values: {...values}, active: {...active}, pending_restart: Object.keys(values).filter(key => !live.has(key) && values[key] !== active[key]), model_defaults: {temperature: 0.7, top_k: 20, top_p: 0.95, max_output_tokens: 32768, thinking: 'model'}});
  const server = createServer(async (req, res) => {
    if (req.url === '/console') {
      res.setHeader('Content-Type', 'text/html');
      return res.end(page);
    }
    const fixtures = {
      '/health': { status: 'ok', model: 'fixture-model', maintenance: 'serving', uptime_s: 720, version: 'test' },
      '/admin/stats': {
        uptime_s: 720, model: { id: 'fixture-model', moe: true, attn: 'hybrid', ctx: 32768 },
        throughput: { decode_tps: 78.5 }, requests: { completed: 12, active: 0, prompt_tokens_total: 2000, completion_tokens_total: 800, ttft_mean_ms: 250 },
        vram_bytes: 3 * 2 ** 30, kv: { used_pages: 16, total_pages: 64 },
      },
      '/admin/cache/status': {
        geometry: { moe_cache_size: active.moe_cache_size, num_experts: 64, num_moe_layers: 4, unit_bytes: { moe_per_expert: 1024 ** 2 } },
  context: { ceiling: 32768, current: active.max_seq_len_override }, generation: {temperature: active.temperature, top_k: active.top_k, top_p: active.top_p, max_output_tokens: active.max_output_tokens, thinking: active.thinking},
  reasoning_budget: active.max_reasoning_tokens, prefix_disk: { bytes: 2 ** 30, budget_bytes: 20 * 2 ** 30, blocks: 4, hits: 3 },
      },
      '/admin/settings': settingsDoc(),
  '/admin/requests': { entries: [{ ts: 1700000000, path: '/v1/chat/completions', model: 'fixture-model', prompt_tokens: 100, completion_tokens: 20, ttft_ms: 250, status: 200 }] },
    };
    if (req.method === 'POST') {
      let raw = '';
      for await (const chunk of req) raw += chunk;
      const body = raw ? JSON.parse(raw) : {};
      mutations.push({ path: req.url, body });
      if (req.url === '/v1/chat/completions') {
        res.setHeader('Content-Type', 'text/event-stream');
        res.write('data: {"choices":[{"delta":{"reasoning_content":"Considering the question."}}]}\n\n');
        await new Promise(resolve => setTimeout(resolve, 200));
        if (!res.destroyed) {
          res.end('data: {"choices":[{"delta":{"content":"Hello from the local model."}}]}\n\ndata: {"usage":{"prompt_tokens":100,"completion_tokens":20}}\n\ndata: [DONE]\n\n');
        }
        return;
      }
      res.setHeader('Content-Type', 'application/json');
  if (req.url === '/admin/settings') {
    if (delaySettings) await new Promise(resolve => setTimeout(resolve, delaySettings));
    if (failSettings) { res.statusCode = 500; return res.end(JSON.stringify({error: 'fixture disk failure'})); }
    Object.assign(values, body);
    for (const key of Object.keys(body)) if (live.has(key)) active[key] = body[key];
    return res.end(JSON.stringify({status: 'ok', settings: settingsDoc()}));
  }
  if (req.url === '/admin/reload') {
    for (const key of Object.keys(values)) if (!key.startsWith('server_')) active[key] = values[key];
  }
  return res.end(JSON.stringify({ status: 'ok', ...body }));
    }
    res.setHeader('Content-Type', 'application/json');
    if (req.url === '/health' && !serving) return res.end(JSON.stringify({ status: 'error', message: 'fixture failure' }));
    res.end(JSON.stringify(fixtures[req.url] || {}));
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const base = `http://127.0.0.1:${server.address().port}`;
  const profile = await mkdtemp(join(tmpdir(), 'maxmlx-ui-'));
  const browser = spawn(executable, ['--headless=new', '--no-first-run', '--no-default-browser-check', '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank'], { stdio: ['ignore', 'ignore', 'pipe'] });
  let socket;
  try {
    const endpoint = await new Promise((resolve, reject) => {
      let output = '';
      const timer = setTimeout(() => reject(new Error('Browser did not expose CDP')), 15000);
      browser.once('error', error => { clearTimeout(timer); reject(error); });
      browser.stderr.on('data', chunk => {
        output += chunk;
        const match = output.match(/DevTools listening on (ws:\/\/[^\s]+)/);
        if (match) { clearTimeout(timer); resolve(match[1]); }
      });
    });
    socket = new WebSocket(endpoint);
    await once(socket, 'open');
    let sequence = 0;
    const pending = new Map();
    const errors = [];
    socket.addEventListener('message', event => {
      const message = JSON.parse(event.data);
      if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails.text + ': ' + (message.params.exceptionDetails.exception?.description || ''));
      if (message.id && pending.has(message.id)) {
        const { resolve, reject } = pending.get(message.id);
        pending.delete(message.id);
        if (message.error) reject(new Error(message.error.message));
        else resolve(message.result);
      }
    });
    const call = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
      const id = ++sequence;
      pending.set(id, { resolve, reject });
      socket.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
    });
    const { targetId } = await call('Target.createTarget', { url: 'about:blank' });
    const { sessionId } = await call('Target.attachToTarget', { targetId, flatten: true });
    const command = (method, params = {}) => call(method, params, sessionId);
    await command('Runtime.enable');
    const evaluate = async expression => {
      const result = await command('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
      assert.equal(result.exceptionDetails, undefined, JSON.stringify(result.exceptionDetails));
      return result.result.value;
    };
    const waitFor = async expression => {
      for (let i = 0; i < 100; i++) {
        if (await evaluate(expression)) return;
        await new Promise(resolve => setTimeout(resolve, 50));
      }
      assert.fail(`Timed out: ${expression}`);
    };
    const screenshot = async name => {
      const directory = process.env.MAXMLX_UI_SCREENSHOT_DIR;
      if (!directory) return;
      const { data } = await command('Page.captureScreenshot', { format: 'png' });
      await writeFile(join(directory, name + '.png'), Buffer.from(data, 'base64'));
    };
    await command('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
    await command('Page.navigate', { url: base + '/console' });
    await waitFor('document.getElementById("memTxt")?.textContent === "3 GiB"');
    assert.equal(await evaluate('document.querySelector(".view.active").id'), 'dashboardView');
    assert.equal(await evaluate('document.getElementById("mTps").textContent'), '78.5');
    assert.equal(await evaluate('document.getElementById("genTemp").closest("section").id'), 'settingsView');
    assert.equal(await evaluate('document.querySelectorAll(".nav").length'), 4);
    await screenshot('dashboard');

    await evaluate('document.querySelector("[data-view=settingsView]").click()');
    assert.equal(await evaluate('location.hash'), '#settings');
    assert.equal(await evaluate('document.querySelector(".nav[aria-current=page]").dataset.view'), 'settingsView');
    await waitFor('settingsOnline && document.getElementById("genTopK").value === "20"');
  assert.equal(await evaluate('document.querySelectorAll("#settingsSave").length'), 1);
assert.equal(await evaluate('document.getElementById("settingsSave").closest(".settings-panel")'), null);
await screenshot('settings');
  await evaluate('window.setInput = (id, value) => { const input = document.getElementById(id); input.value = value; input.dispatchEvent(new Event("input", {bubbles:true})); }; setInput("genTemp", "0.9"); setInput("genMax", "8192"); setInput("genTopK", "40"); setInput("genTopP", "0.85"); setInput("genThink", "off");');
  await evaluate('pollCache(); pollSettings(); document.getElementById("input").focus()');
  assert.equal(await evaluate('document.getElementById("genTemp").value'), '0.9');
  await evaluate('document.getElementById("settingsSave").click()');
  await waitFor('document.getElementById("settingsSaveMsg").textContent.includes("Saved") && busySettings.size === 0');
  assert.deepEqual(mutations.find(item => item.path === '/admin/settings').body, { temperature: 0.9, top_k: 40, top_p: 0.85, max_output_tokens: 8192, thinking: 'off' });
  const saveCount = mutations.filter(item => item.path === '/admin/settings').length;
  await evaluate('setInput("ctxRange", "3"); setInput("thinkRange", "2"); showSettingsTab("memorySettings"); setInput("diskBudget", "10"); document.getElementById("settingsSave").click()');
  await waitFor('document.getElementById("settingsSaveMsg").textContent.includes("Saved") && busySettings.size === 0');
  assert.equal(mutations.filter(item => item.path === '/admin/settings').length, saveCount + 1, 'one Save must send all modified categories in one request');
  const combined = mutations.filter(item => item.path === '/admin/settings').at(-1).body;
  assert.equal(combined.max_seq_len_override, 8192);
  assert.equal(combined.max_reasoning_tokens, 512);
  assert.equal(combined.prefix_cache_disk_gb, 10);
  await evaluate('showSettingsTab("memorySettings"); document.querySelector("[data-preset=minimum]").click(); document.getElementById("settingsSave").click()');
  await waitFor('document.getElementById("settingsSaveMsg").textContent.includes("Saved") && busySettings.size === 0');
  assert.ok(mutations.some(item => item.body.moe_cache_size === 64));
  await screenshot('settings-memory');
  await evaluate('setInput("draftMode", "mtp"); setInput("maxRequests", "2"); document.getElementById("settingsSave").click()');
  await waitFor('document.getElementById("settingsSaveMsg").textContent.includes("restart required") && busySettings.size === 0');
  assert.equal(values.draft_model, 'mtp');
  assert.equal(active.draft_model, null);
  assert.equal(values.moe_cache_size, 64, 'saving MTP must not silently change the slot budget');
  assert.equal(values.prefix_cache_disk_gb, 10);
  await evaluate('showSettingsTab("serverSettings"); setInput("serverPort", "2020"); document.getElementById("settingsSave").click()');
  await waitFor('document.getElementById("settingsSaveMsg").textContent.includes("server restart") && busySettings.size === 0');
  await evaluate('window.confirm = () => true; document.getElementById("reloadBtn").click()');
  await waitFor('(document.getElementById("reloadMsg").textContent.includes("workers relaunched") || document.getElementById("reloadMsg").textContent === "reloaded") && busySettings.size === 0');
  assert.equal(active.draft_model, 'mtp');
  assert.equal(active.server_port, 1919, 'worker restart must not pretend to rebind the HTTP server');
  assert.ok(mutations.some(item => item.path === '/admin/reload' && item.body.apply_saved));
  await evaluate('showSettingsTab("integrationSettings"); document.getElementById("agentChoice").value = "codex"; document.getElementById("agentChoice").dispatchEvent(new Event("change"))');
  assert.equal(await evaluate('document.getElementById("apiAgent").textContent'), `mt launch codex --server ${base}`);

  delaySettings = 200;
  await evaluate('showSettingsTab("modelSettings"); setInput("genTemp", "1.1"); document.getElementById("settingsSave").click()');
  await waitFor('busySettings.has("save")');
  await evaluate('setInput("genTemp", "1.2")');
  await waitFor('document.getElementById("settingsSaveMsg").textContent.includes("newer edits") && busySettings.size === 0');
  assert.equal(await evaluate('document.getElementById("genTemp").value'), '1.2');
  assert.equal(values.temperature, 1.1);
  delaySettings = 0;
  await evaluate('document.getElementById("settingsSave").click()');
  await waitFor('!dirtySettings.has("generation") && busySettings.size === 0');
  failSettings = true;
  await evaluate('setInput("genTopK", "31"); document.getElementById("settingsSave").click()');
  await waitFor('document.getElementById("settingsSaveMsg").textContent.includes("Not saved") && busySettings.size === 0');
  assert.equal(await evaluate('document.getElementById("genTopK").value'), '31');
  assert.equal(await evaluate('dirtySettings.has("generation")'), true);
  failSettings = false;
  await evaluate('document.getElementById("settingsSave").click()');
  await waitFor('!dirtySettings.has("generation") && busySettings.size === 0');
  await command('Page.reload');
  await waitFor('typeof settingsOnline !== "undefined" && settingsOnline && document.getElementById("genTopK").value === "31"');
  assert.equal(await evaluate('document.getElementById("genTemp").value'), '1.2');
  const beforeInvalid = mutations.filter(item => item.path === '/admin/settings').length;
  await evaluate('showView("settingsView"); window.setInput = (id, value) => { const input = document.getElementById(id); input.value = value; input.dispatchEvent(new Event("input", {bubbles:true})); }; showSettingsTab("memorySettings"); setInput("maxRequests", "0"); showSettingsTab("modelSettings"); setInput("genTemp", "1.3"); document.getElementById("settingsSave").click()');
  assert.equal(await evaluate('document.querySelector(".settings-panel.active").id'), 'memorySettings');
  assert.equal(mutations.filter(item => item.path === '/admin/settings').length, beforeInvalid);
  await evaluate('showSettingsTab("memorySettings"); setInput("maxRequests", "2"); showSettingsTab("modelSettings"); setInput("genTemp", "1.2"); document.getElementById("settingsSave").click()');
  await waitFor('dirtySettings.size === 0 && document.getElementById("settingsSave").disabled');

    await evaluate('document.querySelector("[data-view=chatView]").click()');
    await screenshot('chat');
    await evaluate('document.getElementById("input").value = "Hello"; document.getElementById("sendBtn").click()');
    await waitFor('document.querySelector(".msg.assistant")?.textContent.includes("Hello from the local model.")');
    assert.equal(await evaluate('document.getElementById("chatEmpty").hidden'), true);
    assert.equal(await evaluate('[...document.querySelectorAll(".spinner")].filter(node => node.offsetParent !== null).length'), 0);
    assert.equal(await evaluate('document.getElementById("stopBtnTop").disabled'), true);
    const request = mutations.find(item => item.path === '/v1/chat/completions').body;
    assert.equal(request.model, 'fixture-model');
    assert.deepEqual(request.messages, [{ role: 'user', content: 'Hello' }]);
    assert.equal(request.temperature, undefined);
    await evaluate('document.getElementById("input").value = "Stop this"; document.getElementById("sendBtn").click()');
    await waitFor('!document.getElementById("stopBtn").disabled');
    await evaluate('document.getElementById("stopBtn").click()');
    await waitFor('document.getElementById("stopBtn").disabled');
    assert.equal(await evaluate('[...document.querySelectorAll(".spinner")].filter(node => node.offsetParent !== null).length'), 0);

    await evaluate('location.hash = "logs"');
    await waitFor('document.getElementById("requestLog").open && document.getElementById("logRows").textContent.includes("fixture-model")');
    assert.equal(await evaluate('document.querySelector(".view.active").id'), 'dashboardView');
    await evaluate('showView("settingsView"); showView("chatView"); history.back()');
    await waitFor('document.querySelector(".view.active").id === "settingsView"');
    serving = false;
    await evaluate('pollHealth()');
    assert.equal(await evaluate('document.getElementById("runDot").style.background'), 'var(--bad)');

    for (const view of ['dashboardView', 'settingsView', 'chatView']) {
      await command('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
      await evaluate(`showView("${view}")`);
      assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth'), true, `${view} must not overflow on mobile`);
    }
    await screenshot('chat-mobile');
    assert.deepEqual(errors, []);
  } finally {
    socket?.close();
    const exited = once(browser, 'exit');
    browser.kill();
    await exited;
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    await rm(profile, { recursive: true, force: true });
  }
});
