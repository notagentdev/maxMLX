import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { browserAvailable, withBrowser } from './harness.mjs';

const GIB = 1024 ** 3;
const model = (id, extra = {}) => ({ repo_id: id, name: id.split('/').pop(), downloads: 1500, likes: 12, trending_score: 1, size: 4 * GIB, size_formatted: '4.0 GB', params: 7e9, params_formatted: '7.0B', ...extra });

test('downloads tab browses, searches, queues and configures', { skip: !browserAvailable, timeout: 60000 }, async () => {
  const page = await readFile(new URL('../../python/maxtoken/server/console.html', import.meta.url));
  const requests = [];
  let tasks = [];
  let msAvailable = true;
  const config = { models_dir: '/fixture/models', configured: { models_dir: '', hf_endpoint: '', ms_endpoint: '' }, env_models_dir: false, system_memory_bytes: 32 * GIB };
  const trending = [model('mlx-community/Alpha-4bit'), model('mlx-community/Huge-8bit', { size: 31 * GIB, size_formatted: '31.0 GB' }), model('acme/<img src=x onerror=window.pwned=1>')];

  await withBrowser(async (req, res) => {
    const url = new URL(req.url, 'http://x');
    let body = {};
    if (req.method === 'POST') {
      let raw = '';
      for await (const chunk of req) raw += chunk;
      body = raw ? JSON.parse(raw) : {};
    }
    if (url.pathname === '/console') { res.setHeader('Content-Type', 'text/html'); return res.end(page); }
    res.setHeader('Content-Type', 'application/json');
    const send = (data, status = 200) => { res.statusCode = status; res.end(JSON.stringify(data)); };
    const path = url.pathname;
    if (!path.startsWith('/admin/downloads')) return send({});
    requests.push({ method: req.method, path, query: Object.fromEntries(url.searchParams), body });
    if (path === '/admin/downloads/config') {
      if (req.method === 'POST') {
        if (body.hf_endpoint === 'bad') return send({ detail: 'endpoint must be an http(s) URL' }, 400);
        Object.assign(config.configured, body);
        return send({ success: true });
      }
      return send({ ...config, ms_available: msAvailable });
    }
    const [, , , source, action, id] = path.split('/');
    if (action === 'recommended') return send({ trending, popular: [model('mlx-community/Popular-4bit')], hf_token_invalid: false });
    if (action === 'search') return send({ models: [model('mlx-community/Found-4bit', { size: 9 * GIB })], total: 1 });
    if (action === 'tasks') return send({ tasks });
    if (action === 'start') {
      tasks.push({ task_id: 'new', repo_id: body.repo_id, status: 'pending', progress: 0, total_size: 0, downloaded_size: 0, speed_bps: 0, error: '', created_at: 9 });
      return send({ success: true });
    }
    if (action === 'cancel') { tasks = tasks.filter(t => t.task_id !== id); return send({ success: true }); }
    if (action === 'task') { tasks = tasks.filter(t => t.task_id !== id); return send({ success: true }); }
    if (action === 'model-info') return send({ repo_id: url.searchParams.get('repo_id'), model_card: '# Card\n<img src=x onerror=window.pwned=1>', files: [{ name: 'model.safetensors', size: 1, size_formatted: '4.0 GB' }], tags: ['mlx'], downloads: 1, likes: 2, size_formatted: '4.0 GB', params_formatted: '7.0B' });
    return send({ detail: 'unexpected ' + path }, 404);
  }, async ({ base, evaluate, waitFor, command, errors }) => {
    const type = (id, value) => evaluate(`(() => { const e = document.getElementById("${id}"); e.value = ${JSON.stringify(value)}; e.dispatchEvent(new Event("input", { bubbles: true })); })()`);
    await command('Page.navigate', { url: base + '/console#downloads' });
    await waitFor('document.querySelectorAll("#dlList .dl-row").length === 3');
    assert.equal(await evaluate('document.querySelector(".view.active").id'), 'downloadsView');
    assert.equal(await evaluate('document.getElementById("dlDir").textContent'), '/fixture/models');
    assert.equal(await evaluate('document.querySelector(".nav.active").dataset.view'), 'downloadsView');

    // How much of this Mac's memory a model takes, and a warning once it will not fit.
    const fits = JSON.parse(await evaluate('JSON.stringify([...document.querySelectorAll("#dlList .dl-row > .fit")].map(e => e.className + "|" + e.textContent))'));
    assert.match(fits[0], /^fit\|13% of memory/);
    assert.match(fits[1], /^fit over\|97% of memory/);

    // Repo names are remote data: markup in them must stay text.
    assert.equal(await evaluate('window.pwned'), undefined);
    assert.equal(await evaluate('document.querySelectorAll("#dlList img").length'), 0);

    // Manual download: disabled until something is typed; the token travels with it.
    assert.equal(await evaluate('document.getElementById("dlStart").disabled'), true);
    await type('dlRepo', 'mlx-community/Manual-4bit');
    await type('dlToken', 'hf_secret');
    assert.equal(await evaluate('document.getElementById("dlStart").disabled'), false);
    await evaluate('document.getElementById("dlAdd").requestSubmit()');
    await waitFor('document.querySelectorAll("#dlQueue .dl-job").length === 1');
    assert.deepEqual(requests.find(r => r.path === '/admin/downloads/hf/start').body, { repo_id: 'mlx-community/Manual-4bit', token: 'hf_secret' });
    assert.equal(await evaluate('document.getElementById("dlRepo").value'), '');
    assert.match(await evaluate('document.getElementById("dlOk").textContent'), /Manual-4bit/);

    // A list row downloads that repository.
    await evaluate('document.querySelectorAll("#dlList .dl-row .btn:not(.ghost)")[0].click()');
    await waitFor('dl.tasks.hf.some(t => t.repo_id === "mlx-community/Alpha-4bit")');

    // Queue: running first, history after; failed rows offer Retry, finished rows Remove.
    tasks = [
      { task_id: 'f', repo_id: 'mlx-community/Failed-4bit', status: 'failed', progress: 0, total_size: 0, downloaded_size: 0, speed_bps: 0, error: 'Repository not found', created_at: 5 },
      { task_id: 'r', repo_id: 'mlx-community/Running-4bit', status: 'downloading', progress: 50, total_size: 8 * GIB, downloaded_size: 4 * GIB, speed_bps: 50 * 1024 ** 2, error: '', created_at: 1 },
    ];
    await evaluate('pollDownloads()');
    await waitFor('document.querySelectorAll("#dlQueue .dl-job").length === 2');
    assert.equal(await evaluate('document.querySelector("#dlQueue .dl-job .name").textContent'), 'mlx-community/Running-4bitdownloading');
    const running = await evaluate('document.querySelector("#dlQueue .dl-job .stat").textContent');
    assert.match(running, /50%/); assert.match(running, /4\.0 GB \/ 8\.0 GB/); assert.match(running, /50\.0 MB\/s/); assert.match(running, /left/);
    assert.equal(await evaluate('document.querySelector("#dlQueue .dl-job .dl-bar > i").style.width'), '50%');
    assert.equal(await evaluate('[...document.querySelectorAll("#dlQueue .dl-job:nth-child(2) .btn")].map(b => b.textContent).join()'), 'Retry,Remove');
    assert.equal(await evaluate('document.querySelector("#dlQueue .dl-job .btn").textContent'), 'Cancel');
    await evaluate('document.querySelector("#dlQueue .dl-job:nth-child(2) .btn:last-child").click()');
    await waitFor('document.querySelectorAll("#dlQueue .dl-job").length === 1');
    assert.ok(requests.some(r => r.method === 'DELETE' && r.path === '/admin/downloads/hf/task/f'));
    await evaluate('document.querySelector("#dlQueue .dl-job .btn").click()');
    await waitFor('document.getElementById("dlQueueCard").hidden');
    assert.ok(requests.some(r => r.method === 'POST' && r.path === '/admin/downloads/hf/cancel/r'));

    // Search: filters are sent in the server's units and the list follows the chosen order.
    await evaluate('document.getElementById("dlFilterBtn").click()');
    await type('dlMinParams', '3'); await type('dlMaxSize', '2');
    await type('dlSort', 'largest');
    await evaluate('document.getElementById("dlSort").value = "largest"');
    await type('dlQuery', 'qwen');
    await evaluate('document.getElementById("dlSearchForm").requestSubmit()');
    await waitFor('document.querySelector("#dlList .title")?.textContent === "mlx-community/Found-4bit"');
    const search = requests.filter(r => r.path === '/admin/downloads/hf/search').at(-1).query;
    assert.deepEqual(search, { q: 'qwen', sort: 'largest', limit: '100', mlx_only: 'true', min_params: String(3e9), max_size: String(2 * GIB) });
    assert.equal(await evaluate('document.querySelector("#dlBrowse [aria-pressed=true]").dataset.tab'), 'search');
    assert.equal(await evaluate('document.getElementById("dlOrder").value + document.getElementById("dlDir2").textContent'), 'size↓');
    assert.equal(await evaluate('JSON.parse(localStorage.hfSearchHistory)[0]'), 'qwen');

    // Info dialog: the model card is shown as text, never as markup.
    await evaluate('document.querySelector("#dlList .btn.ghost").click()');
    await waitFor('document.getElementById("dlDetail").open && document.querySelector("#dlDetailBody pre")');
    assert.match(await evaluate('document.getElementById("dlDetailBody").textContent'), /<img src=x onerror=window\.pwned=1>/);
    assert.equal(await evaluate('document.querySelectorAll("#dlDetailBody img").length'), 0);
    assert.equal(await evaluate('window.pwned'), undefined);
    await evaluate('document.getElementById("dlDetailGet").click()');
    assert.equal(await evaluate('document.getElementById("dlDetail").open'), false);
    await waitFor('dl.tasks.hf.some(t => t.repo_id === "mlx-community/Found-4bit")');

    // Folder and mirror: a server-side rejection keeps the dialog open and says why.
    await evaluate('document.getElementById("dlSettingsBtn").click()');
    await type('dlCfgHf', 'bad');
    await evaluate('document.getElementById("dlCfgSave").click()');
    await waitFor('!document.getElementById("dlCfgError").hidden');
    assert.equal(await evaluate('document.getElementById("dlCfgError").textContent'), 'endpoint must be an http(s) URL');
    assert.equal(await evaluate('document.getElementById("dlSettings").open'), true);
    await type('dlCfgHf', 'https://hf-mirror.com'); await type('dlCfgDir', '/data/models');
    await evaluate('document.getElementById("dlCfgSave").click()');
    await waitFor('!document.getElementById("dlSettings").open');
    assert.deepEqual(requests.filter(r => r.method === 'POST' && r.path === '/admin/downloads/config').at(-1).body, { hf_endpoint: 'https://hf-mirror.com', ms_endpoint: '', models_dir: '/data/models' });

    // ModelScope: its own list, no filters, bare repo names get the mlx-community owner.
    await evaluate('document.getElementById("dlSrcMs").click()');
    await waitFor('document.getElementById("dlSrcMs").getAttribute("aria-pressed") === "true" && document.querySelectorAll("#dlList .dl-row").length === 3');
    assert.equal(await evaluate('document.getElementById("dlFilterBtn").hidden'), true);
    assert.equal(await evaluate('document.getElementById("dlMsMissing").hidden'), true);
    await type('dlRepo', 'Bare-4bit');
    await evaluate('document.getElementById("dlAdd").requestSubmit()');
    await waitFor('dl.tasks.ms.some(t => t.repo_id === "mlx-community/Bare-4bit")');

    // Without the SDK the source says so instead of failing silently.
    msAvailable = false;
    await evaluate('loadDownloadConfig()');
    await waitFor('!document.getElementById("dlMsMissing").hidden');

    await command('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
    assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth'), true, 'downloads must not overflow on mobile');
    assert.deepEqual(errors, []);
  });
});
