import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { browserAvailable, withBrowser } from './harness.mjs';

// 1x1 PNG
const PNG = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==', 'base64');

test('chat creates images in image mode and with /image, and keeps them in the conversation', { skip: !browserAvailable, timeout: 90000 }, async () => {
  const page = await readFile(new URL('../../python/maxmlx/server/console.html', import.meta.url));
  const store = new Map();
  const generations = [], completions = [], cancels = [], uploads = [], videoJobs = [];
  const videoPolls = new Map();
  const refFile = join(await mkdtemp(join(tmpdir(), 'maxmlx-ref-')), 'ref.png');
  await writeFile(refFile, PNG);
  let job = null;

  await withBrowser(async (req, res) => {
    const url = new URL(req.url, 'http://x');
    let raw = '';
    for await (const chunk of req) raw += chunk;
    const body = raw ? JSON.parse(raw) : {};
    const json = (data, status = 200) => { res.statusCode = status; res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify(data)); };
    if (url.pathname === '/console') { res.setHeader('Content-Type', 'text/html'); return res.end(page); }
    if (url.pathname === '/health') return json({ status: 'ok', model: 'fixture-model', maintenance: 'serving', uptime_s: 5, version: 'test' });
    if (url.pathname === '/admin/stats') return json({ uptime_s: 5, model: { id: 'fixture-model', ctx: 32768 }, requests: {}, throughput: {} });
    if (url.pathname === '/admin/images/status') return json({ state: job ? 'generating' : 'ready', model: 'flux2-klein-4b', job, waiting: 0, models: [{ id: '/m/flux2-klein-4b', name: 'flux2-klein-4b', size_bytes: 1 }] });
    if (url.pathname === '/admin/images/cancel') { cancels.push(body.request_id); if (job?.request_id === body.request_id) job.cancelled = true; return json({ cancelled: true }); }
    if (url.pathname === '/admin/videos/status') return json({ models: [{ id: '/m/ltx-2.3-mlx-q4', name: 'ltx-2.3-mlx-q4' }], running: null, queued: 0 });
    if (url.pathname === '/admin/videos' && req.method === 'POST') {
      const id = 'b'.repeat(23) + (videoJobs.length + 1);
      videoJobs.push({ id, body });
      return json({ id, status: 'queued', model: 'ltx-2.3-mlx-q4' });
    }
    const vjob = url.pathname.match(/^\/admin\/videos\/([0-9a-f]{24})$/);
    if (vjob) {
      const n = (videoPolls.get(vjob[1]) || 0) + 1;
      videoPolls.set(vjob[1], n);
      const job = videoJobs.find(j => j.id === vjob[1]);
      if (!job) return json({ error: 'unknown video job (the server may have restarted)' }, 404);
      const base = { id: job.id, prompt: job.body.prompt, width: 768, height: 512, frames: 97, seconds: 4.04, seed: 77, started: Date.now() / 1000 - 5 };
      if (n < (job.body.prompt.startsWith('long') ? 6 : 3)) return json({ ...base, status: 'running', phase: `Stage 1 · step ${n} of 8` });
      return json({ ...base, status: 'done', phase: 'Done', url: `/admin/videos/files/${job.id}.mp4`, elapsed: 312.5 });
    }
    if (url.pathname.startsWith('/admin/videos/files/')) { res.setHeader('Content-Type', 'video/mp4'); return res.end(Buffer.alloc(0)); }
    if (url.pathname === '/admin/images/uploads') {
      uploads.push(body.data.slice(0, 22));
      const id = 'a'.repeat(23) + uploads.length;
      return json({ id, url: `/admin/images/files/${id}.png`, width: 1, height: 1 });
    }
    if (url.pathname.startsWith('/admin/images/files/')) { res.setHeader('Content-Type', 'image/png'); return res.end(PNG); }
    if (url.pathname === '/v1/images/generations') {
      generations.push(body);
      job = { request_id: body.request_id, phase: 'denoising', step: 0, total: body.steps || 4 };
      const slow = body.prompt.startsWith('slow');
      for (let i = 1; i <= (slow ? 40 : 12) && !job.cancelled; i++) { job.step = Math.min(i, job.total); await new Promise(r => setTimeout(r, 150)); }
      const cancelled = job.cancelled; job = null;
      if (cancelled) return json({ error: { message: 'cancelled' } }, 499);
      const id = (generations.length.toString(16)).padStart(24, '0');
      const [w, h] = (body.size || '1024x1024').split('x').map(Number);
      return json({ created: 1, data: [{ url: `/admin/images/files/${id}.png`, seed: body.seed ?? 1234, width: w, height: h, steps: body.steps || 4, seconds: 0.5, model: 'flux2-klein-4b' }] });
    }
    if (url.pathname === '/v1/chat/completions') {
      completions.push(body);
      res.setHeader('Content-Type', 'text/event-stream');
      res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: 'It shows a fox.' } }] })}\n\ndata: [DONE]\n\n`);
      return res.end();
    }
    if (url.pathname === '/admin/chat-profiles') return json({ profiles: [{ name: 'System Default', content: '' }] });
    if (url.pathname === '/admin/chats') return json({ chats: [...store.values()].map(d => ({ id: d.id, title: d.title, updatedAt: d.updatedAt, messageCount: d.messages.length })) });
    const match = url.pathname.match(/^\/admin\/chats\/([\w-]+)$/);
    if (match) {
      if (req.method === 'PUT') { store.set(match[1], body); return json({ id: match[1] }); }
      return store.has(match[1]) ? json(store.get(match[1])) : json({ detail: 'Chat not found' }, 404);
    }
    json({});
  }, async ({ base, evaluate, waitFor, command, errors }) => {
    const send = async text => evaluate(`(() => { const e = document.getElementById("input"); e.value = ${JSON.stringify(text)}; document.getElementById("sendBtn").click(); })()`);
    const idle = () => waitFor('!document.getElementById("sendBtn").disabled');
    const until = async (predicate, label) => {
      for (let i = 0; i < 200; i++) { if (await predicate()) return; await new Promise(r => setTimeout(r, 50)); }
      assert.fail('Timed out waiting for ' + label);
    };

    await command('Page.navigate', { url: base + '/console#chat' });
    await waitFor('typeof chat !== "undefined" && chat.cur && !document.getElementById("imgMode").disabled');

    // The image model's state is visible: a dot on the chat's image button, a row on the dashboard.
    await waitFor('document.getElementById("imgMode").dataset.state === "ready"');
    assert.equal(await evaluate('document.getElementById("imgTxt").textContent'), 'Loaded');
    assert.match(await evaluate('document.getElementById("csImgModel").textContent'), /flux2-klein-4b · loaded/);

    // Image mode: the composer turns into an image prompt.
    await evaluate('document.getElementById("imgMode").click()');
    assert.equal(await evaluate('document.getElementById("sendBtn").textContent'), 'Create');
    await evaluate('document.getElementById("csImgSize").value = "768x1024"; document.getElementById("csImgSize").dispatchEvent(new Event("input"))');
    await send('a red fox in the snow');
    await until(() => evaluate('document.querySelector(".msg.assistant .phase")?.textContent.includes("step")'), 'step progress');
    await until(() => evaluate('!!document.querySelector(".msg.assistant .mimage img")'), 'the image');
    await idle();
    assert.equal(generations[0].prompt, 'a red fox in the snow');
    assert.equal(generations[0].size, '768x1024');
    assert.equal(generations[0].response_format, 'url');
    assert.match(await evaluate('document.querySelector(".msg.assistant .mmeta").textContent'), /768×1024 · seed 1234 · 4 steps/);
    assert.equal(await evaluate('document.querySelector(".msg.user .img-tag")?.textContent'), 'IMAGE');
    assert.match(await evaluate('document.querySelector(".msg.assistant .mimage img").getAttribute("src")'), /^\/admin\/images\/files\/[0-9a-f]{24}\.png$/);
    await until(() => evaluate('(img => img.complete && img.naturalWidth === 1)(document.querySelector(".msg.assistant .mimage img"))'), 'the image to actually load');

    // Back to text: the language model is told an image was made, not sent the pixels.
    await evaluate('document.getElementById("imgMode").click()');
    await send('what did you draw?');
    await until(() => completions.length === 1, 'the chat request');
    await idle();
    const sent = completions[0].messages.map(m => m.content);
    assert.deepEqual(sent, ['Create an image: a red fox in the snow', '[Image created for: a red fox in the snow]', 'what did you draw?']);

    // "/image" works without the toggle.
    await send('/image a lighthouse');
    await until(() => generations.length === 2, 'the /image request');
    await until(() => evaluate('document.querySelectorAll(".msg.assistant .mimage img").length === 2'), 'the second image');
    await idle();
    assert.equal(generations[1].prompt, 'a lighthouse');

    // Reference images: attached through the file picker, uploaded at once, sent as ids.
    const { root } = await command('DOM.getDocument');
    const { nodeId } = await command('DOM.querySelector', { nodeId: root.nodeId, selector: '#attachInput' });
    await command('DOM.setFileInputFiles', { nodeId, files: [refFile] });
    await until(() => evaluate('chat.attachments.length === 1 && !!chat.attachments[0].id'), 'the upload');
    assert.equal(uploads[0], 'data:image/png;base64,');
    assert.equal(await evaluate('chat.imageMode'), true, 'attaching turns on image mode');
    await send('make it blue');
    await until(() => generations.length === 3, 'the reference request');
    await idle();
    assert.deepEqual(generations[2].references, ['a'.repeat(23) + '1']);
    assert.equal(await evaluate('document.getElementById("attachRow").hidden'), true, 'sending clears the attachments');
    assert.equal(await evaluate('[...document.querySelectorAll(".msg.user")].at(-1).querySelectorAll(".refs img").length'), 1);
    // A generated image can become the next reference.
    await evaluate('[...document.querySelectorAll(".mimage .img-acts button")].filter(b => b.textContent === "Use as reference").at(-1).click()');
    assert.equal(await evaluate('chat.attachments.length'), 1);
    await send('now at night');
    await until(() => generations.length === 4, 'the follow-up reference request');
    await idle();
    assert.match(generations[3].references[0], /^[0-9a-f]{24}$/);

    // Animate: a generated image becomes the first frame of a video job.
    await evaluate('[...document.querySelectorAll(".mimage .img-acts button")].filter(b => b.textContent === "Animate")[0].click()');
    assert.equal(await evaluate('chat.videoMode && !chat.imageMode && chat.attachments.length === 1'), true);
    await evaluate('document.getElementById("csVidSecs").value = "4"; document.getElementById("csVidSecs").dispatchEvent(new Event("input"))');
    await send('the fox turns its head');
    await until(() => videoJobs.length === 1, 'the video job');
    assert.equal(videoJobs[0].body.seconds, 4);
    assert.match(videoJobs[0].body.image, /^[0-9a-f]{24}$/);
    assert.equal(await evaluate('[...document.querySelectorAll(".msg.user")].at(-1).querySelector(".img-tag").textContent'), 'VIDEO');
    await until(() => evaluate('!!document.querySelector(".mvideo video")'), 'the video player');
    assert.match(await evaluate('document.querySelector(".mvideo video").getAttribute("src")'), /^\/admin\/videos\/files\/b{23}1\.mp4$/);
    assert.match(await evaluate('[...document.querySelectorAll(".msg.assistant .mmeta")].at(-1).textContent'), /768×512 · 4.04s · 97 frames · seed 77 · rendered in 5:12/);
    // A text turn after it sees the video as text.
    await evaluate('document.getElementById("vidMode").click()');
    await send('nice');
    await until(() => completions.length === 2, 'the second chat request');
    await idle();
    assert.ok(completions[1].messages.some(m => m.content === '[Video created for: the fox turns its head]'));

    // A reload while a video renders resumes watching it from the saved job id.
    await evaluate('document.getElementById("vidMode").click()');
    await send('long shot of the coast');
    await until(() => videoJobs.length === 2, 'the second video job');
    const chatId = await evaluate('chat.cur.id');
    await until(() => store.get(chatId)?.messages.at(-1)?.videoJob === videoJobs[1].id, 'the job id to be saved');
    await evaluate('window.__stale = true');
    await command('Page.reload');
    await waitFor('!window.__stale && typeof chat !== "undefined" && chat.cur');
    await evaluate(`openChat(${JSON.stringify(chatId)})`);
    await until(() => evaluate('document.querySelectorAll(".mvideo video").length === 2'), 'the resumed video');
    await until(() => store.get(chatId)?.messages.at(-1)?.video?.url?.endsWith('2.mp4'), 'the resumed video to be saved');
    assert.equal(await evaluate('chat.videoMode'), false, 'a fresh page starts in text mode');

    // Stop cancels the render and leaves no empty answer behind.
    await send('/image slow sunset');
    await until(() => evaluate('document.querySelector("#stopBtn:not(:disabled)") !== null'), 'stop to arm');
    await new Promise(r => setTimeout(r, 300));
    await evaluate('document.getElementById("stopBtn").click()');
    await idle();
    assert.equal(cancels.length, 1);
    const id = await evaluate('chat.cur.id');
    await until(() => store.get(id)?.messages.length === 17, 'the saved chat');
    const kinds = store.get(id).messages.map(m => [m.role, m.kind || '', !!m.image]);
    assert.deepEqual(kinds, [
      ['user', 'image', false], ['assistant', 'image', true], ['user', '', false], ['assistant', '', false],
      ['user', 'image', false], ['assistant', 'image', true], ['user', 'image', false], ['assistant', 'image', true],
      ['user', 'image', false], ['assistant', 'image', true], ['user', 'video', false], ['assistant', 'video', false],
      ['user', '', false], ['assistant', '', false], ['user', 'video', false], ['assistant', 'video', false],
      ['user', 'image', false],
    ]);
    assert.equal(store.get(id).messages[11].video.url, `/admin/videos/files/${'b'.repeat(23)}1.mp4`);
    assert.equal(store.get(id).messages[11].videoJob, undefined, 'a finished video forgets its job id');
    assert.equal(store.get(id).messages[6].refs[0].id, 'a'.repeat(23) + '1');
    assert.deepEqual(errors, []);
  });
});
