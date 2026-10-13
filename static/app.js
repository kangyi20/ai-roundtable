"use strict";

let debateId = localStorage.getItem("debate_id") || null;
let snapshot = null;
let autoRunning = false;
let es = null;
let pendingDraft = null; // 提炼后待确认的草稿（含服务端 draft_id 与 L3 体贴版处境）

const STATE_LABEL = {
  IDLE: "待开始",
  A_SPEAKING: "A 方思考中",
  A_WAIT_CONFIRM: "A 方待确认",
  B_SPEAKING: "B 方思考中",
  B_WAIT_CONFIRM: "B 方待确认",
  ROUND_END: "讨论结束，可生成结论",
  SUMMARIZING: "正在生成结论",
  SCORING: "正在评分",
  DONE: "已完成",
};
const STATUS_LABEL = {
  draft: "草稿",
  confirmed: "已确认",
  rejected: "已驳回",
  rejected_twice: "二次驳回",
  note: "备注",
  dodge: "放弃",
};

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}
function fmt(n) {
  return Number(n || 0).toLocaleString("en-US");
}

async function loadMeta() {
  const res = await fetch("/api/meta");
  const meta = await res.json();
  const wl = meta.defaults.word_limit;
  const len = document.getElementById("f_length");
  if ([200, 300, 400].includes(wl)) len.value = String(wl);
  document.getElementById("f_rounds").value = meta.defaults.max_rounds;
  document.getElementById("f_early").checked = !!meta.defaults.early_stop;
  document.getElementById("f_mock").checked = !!meta.defaults.mock_mode;
  document.getElementById("f_input_budget").value = meta.defaults.input_budget ?? 3500;
  document.getElementById("f_output_max_tokens").value = meta.defaults.output_max_tokens ?? 0;
  document.getElementById("f_reasoning_budget").value = meta.defaults.reasoning_budget ?? 2048;
  document.getElementById("f_reasoning_open_close").value = meta.defaults.reasoning_budget_open_close ?? 4096;
  document.getElementById("f_hard_cap").value = meta.defaults.hard_cap ?? 8192;
  document.getElementById("f_strict").checked = !!meta.defaults.strict_evidence;
  document.getElementById("f_overrun").value = meta.defaults.overrun_mode || "penalty";
  document.getElementById("f_forfeit").value = meta.defaults.forfeit_policy || "skip";
  document.getElementById("models").innerHTML = Object.entries(meta.models)
    .map(([k, v]) => `<div class="row"><b>${k}</b>：${esc(v)}</div>`)
    .join("");
  document.getElementById("f_load").innerHTML =
    '<option value="">—</option>' +
    meta.debates
      .map((d) => `<option value="${d.debate_id}">${d.debate_id.slice(0, 15)} | ${esc(d.topic)} | 第${d.round_no}轮 | ${d.state}</option>`)
      .join("");
}

function showThinking(label) {
  hideThinking();
  const el = document.createElement("div");
  el.id = "thinking";
  el.className = "thinking";
  el.innerHTML = `<span class="dots"><span></span><span></span><span></span></span> ${esc(label)}`;
  document.getElementById("feed").appendChild(el);
  const feed = document.getElementById("feed");
  feed.scrollTop = feed.scrollHeight;
}
function hideThinking() {
  document.getElementById("thinking")?.remove();
}

function renderMessage(m) {
  if (m.speaker === "SYSTEM" || m.speaker === "REFEREE") {
    const d = document.createElement("div");
    d.className = "notice" + (m.speaker === "REFEREE" ? " referee" : "");
    d.textContent = m.content;
    return d;
  }
  const side = m.speaker.toLowerCase();
  const names = { A: "A 方", B: "B 方" };
  const div = document.createElement("div");
  div.className = "msg msg-" + side + (m.status === "draft" ? " draft" : "");
  const av = document.createElement("div");
  av.className = "avatar av-" + side;
  av.textContent = m.speaker;
  const body = document.createElement("div");
  body.className = "body";
  const head = document.createElement("div");
  head.className = "head";
  head.innerHTML =
    `<b>${names[m.speaker]}</b>` +
    `<span class="meta-dot">·</span><span>第${m.round_no}轮</span>` +
    `<span class="status-pill">${STATUS_LABEL[m.status] || m.status}</span>` +
    (m.model ? `<span class="model-pill">${esc(m.model)}</span>` : "");
  const content = document.createElement("div");
  content.className = "content";
  content.textContent = m.content;
  body.appendChild(head);
  body.appendChild(content);
  div.appendChild(av);
  div.appendChild(body);
  return div;
}

function renderFeed(messages) {
  const feed = document.getElementById("feed");
  feed.innerHTML = "";
  if (!messages || messages.length === 0) {
    feed.innerHTML = '<div class="empty"><div class="empty-icon">🌸</div><div>在右侧配置话题，或从历史讨论中选择一条记录继续</div></div>';
    return;
  }
  for (const m of messages) feed.appendChild(renderMessage(m));
  feed.scrollTop = feed.scrollHeight;
}

function renderConclusion(c) {
  const el = document.getElementById("conclusion");
  if (!c) {
    el.hidden = true;
    return;
  }
  el.hidden = false;
  const badge = c.aligned
    ? '<div class="badge ok">🤝 双方最终达成一致</div>'
    : '<div class="badge no">🔀 双方仍存在分歧</div>';
  const card = (icon, title, items) =>
    items && items.length
      ? `<div class="card"><div class="ct">${icon} ${title}</div><ul>${items.map((i) => `<li>${esc(i)}</li>`).join("")}</ul></div>`
      : "";
  el.innerHTML =
    `<h2>🎯 讨论结论</h2>${badge}` +
    card("✅", "双方共识", c.consensus) +
    card("⚡", "主要分歧", c.disagreements) +
    card("❓", "待解决问题", c.open_questions) +
    (c.takeaway ? `<div class="takeaway">💡 一句话结论：${esc(c.takeaway)}</div>` : "");
}

function renderScore(score) {
  const el = document.getElementById("score");
  if (!score) {
    el.hidden = true;
    return;
  }
  el.hidden = false;
  const dims = { logic: "逻辑", evidence: "论据", rebuttal: "回应互动", clarity: "清晰度" };
  const winner = score.winner === "A" ? "A 方" : score.winner === "B" ? "B 方" : "平局";
  const side = (label, obj) =>
    obj
      ? `<div class="card"><div class="ct">${label}</div><ul>${Object.entries(obj).map(([k, v]) => `<li>${dims[k] || k}：${v}</li>`).join("")}</ul></div>`
      : "";
  el.innerHTML =
    `<h2>🏆 质量评估</h2><div class="badge ok">胜出：${winner}</div>` +
    side("A 方", score.A) +
    side("B 方", score.B) +
    (score.reason ? `<div class="takeaway">📝 ${esc(score.reason)}</div>` : "");
}

async function doExport() {
  const res = await fetch(`/api/${debateId}/export`, { method: "POST" });
  const data = await res.json();
  if (data.error) {
    alert(data.error);
    return;
  }
  const names = { transcript: "讨论记录", score: "评分结果", config: "配置快照", conclusion: "结论" };
  document.getElementById("exportLinks").innerHTML = (data.kinds || [])
    .map((k) => `<a class="btn" href="/api/${debateId}/export/${k}" download>⬇ ${names[k] || k}</a>`)
    .join("");
}

function renderControls(s) {
  const startPanel = document.getElementById("startPanel");
  const controlPanel = document.getElementById("controlPanel");
  if (!s || s.state === "IDLE") {
    startPanel.hidden = false;
    controlPanel.hidden = true;
    return;
  }
  startPanel.hidden = true;
  controlPanel.hidden = false;

  const st = s.state;
  const manual = document.getElementById("manualControls");
  manual.innerHTML = "";
  const addBtn = (label, fn, cls) => {
    const b = document.createElement("button");
    b.className = "btn" + (cls ? " " + cls : "");
    b.textContent = label;
    b.onclick = fn;
    manual.appendChild(b);
  };

  if (st === "A_SPEAKING") addBtn("让 A 发言", () => act("speak/A", "A 正在思考…"));
  if (st === "A_WAIT_CONFIRM") {
    addBtn("确认 A", () => act("confirm/A"), "confirm");
    addBtn("驳回 A", () => act("reject/A"), "reject");
  }
  if (st === "B_SPEAKING") addBtn("让 B 发言", () => act("speak/B", "B 正在思考…"));
  if (st === "B_WAIT_CONFIRM") {
    addBtn("确认 B", () => act("confirm/B"), "confirm");
    addBtn("驳回 B", () => act("reject/B"), "reject");
  }
  if (st === "ROUND_END" || (st === "DONE" && !s.conclusion))
    addBtn("生成结论", () => act("summarize", "正在生成结论…"));
  if (st === "ROUND_END" || (st === "DONE" && !s.score))
    addBtn("终局评分", () => act("score", "正在评分…"), "confirm");
  addBtn("导出", doExport);

  const autoBtn = document.getElementById("btn_auto_toggle");
  if (autoRunning) {
    autoBtn.textContent = "⏸ 暂停自动讨论";
    autoBtn.onclick = stopAuto;
    autoBtn.disabled = false;
  } else {
    autoBtn.textContent = st === "DONE" ? "已完成" : "▶ 自动讨论";
    autoBtn.onclick = startAuto;
    autoBtn.disabled = st === "DONE";
  }

  document.querySelector(".note-box").style.display =
    st === "IDLE" || st === "DONE" ? "none" : "flex";
}

function render(s) {
  snapshot = s;
  if (!s) {
    document.getElementById("m_in").textContent = "0";
    document.getElementById("m_out").textContent = "0";
    document.getElementById("m_cost").textContent = "¥0.0000";
    document.getElementById("m_round").textContent = "–";
    document.getElementById("statusbar").textContent = "";
    document.getElementById("stances").hidden = true;
    document.getElementById("conclusion").hidden = true;
    renderScore(null);
    renderQA([], null);
    document.getElementById("exportLinks").innerHTML = "";
    renderFeed([]);
    renderControls(null);
    return;
  }
  document.getElementById("m_in").textContent = fmt(s.totals.usage_in);
  document.getElementById("m_out").textContent = fmt(s.totals.usage_out);
  document.getElementById("m_cost").textContent = "¥" + (s.cost_cny || 0).toFixed(4);
  document.getElementById("m_round").textContent = `${s.round_no}/${s.max_rounds}`;
  const dotClass = autoRunning ? "state-dot" : "state-dot manual";
  const modeText = autoRunning ? "🟢 自动推进中" : "⚪ 手动";
  document.getElementById("statusbar").innerHTML =
    `<span><span class="${dotClass}"></span>第 ${s.round_no} / ${s.max_rounds} 轮 · 状态：${STATE_LABEL[s.state] || s.state}</span>` +
    `<span>${modeText}</span>`;

  const stances = document.getElementById("stances");
  stances.hidden = false;
  stances.innerHTML =
    `<div class="row"><strong>A 当前立场</strong>：${esc(s.stance_a)}</div>` +
    `<div class="row"><strong>B 当前立场</strong>：${esc(s.stance_b)}</div>` +
    (s.converged ? '<div class="converged">✅ 双方已达成一致</div>' : "") +
    '<div class="hint">立场可随讨论漂移，一方可能被另一方说服。</div>';

  renderConclusion(s.conclusion);
  renderScore(s.score);
  renderQA(s.qa || [], s.state);
  renderFeed(s.messages);
  renderControls(s);
}

/* ---------------- 讨论结束后向裁判追问 ---------------- */

function renderQA(qa, state) {
  const panel = document.getElementById("qaPanel");
  panel.hidden = state !== "DONE";
  const hist = document.getElementById("qaHistory");
  hist.innerHTML = (qa || [])
    .map((item) => {
      const isUser = item.role === "user";
      return (
        `<div class="qa-item ${isUser ? "user" : "judge"}">` +
        `<span class="qa-role">${isUser ? "你" : "裁判"}</span>` +
        `${isUser ? esc(item.content) : esc(item.content)}</div>`
      );
    })
    .join("");
  if (hist.lastElementChild) hist.lastElementChild.scrollIntoView({ block: "nearest" });
}

async function askJudge() {
  const box = document.getElementById("f_question");
  const question = box.value.trim();
  if (!question || !debateId) return;
  const btn = document.getElementById("btn_ask");
  btn.disabled = true;
  btn.textContent = "裁判正在思考…";
  try {
    const res = await fetch(`/api/${debateId}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    if (data.error) {
      alert(data.error);
      return;
    }
    box.value = "";
    render(data);
  } catch (e) {
    alert("追问失败：" + e);
  } finally {
    btn.disabled = false;
    btn.textContent = "提问";
  }
}

async function act(path, thinkingLabel) {
  if (thinkingLabel) showThinking(thinkingLabel);
  const res = await fetch(`/api/${debateId}/${path}`, { method: "POST" });
  const data = await res.json();
  hideThinking();
  if (data.error) {
    alert(data.error);
    return;
  }
  render(data);
}

async function start(auto) {
  const problem = document.getElementById("f_problem").value.trim();
  // 写了困惑但还没提炼过 → 先自动提炼，弹出确认区（没有独立的「提炼」按钮）
  if (problem && !pendingDraft) {
    await runExtract();
    return;
  }
  const num = (id) => Number(document.getElementById(id).value);
  const body = {
    topic: document.getElementById("f_topic").value.trim(),
    pro_side: document.getElementById("f_pro").value.trim(),
    con_side: document.getElementById("f_con").value.trim(),
    draft_id: pendingDraft ? pendingDraft.draft_id : "",
    user_context_soft: pendingDraft ? pendingDraft.user_context_soft : "",
    config: {
      word_limit: num("f_length"),
      max_rounds: num("f_rounds"),
      early_stop: document.getElementById("f_early").checked,
      mock_mode: document.getElementById("f_mock").checked,
      input_budget: num("f_input_budget"),
      output_max_tokens: num("f_output_max_tokens"),
      reasoning_budget: num("f_reasoning_budget"),
      reasoning_budget_open_close: num("f_reasoning_open_close"),
      hard_cap: num("f_hard_cap"),
      strict_evidence: document.getElementById("f_strict").checked,
      overrun_mode: document.getElementById("f_overrun").value,
      forfeit_policy: document.getElementById("f_forfeit").value,
    },
  };
  const res = await fetch("/api/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (data.error) {
    alert(data.error);
    return;
  }
  debateId = data.debate_id;
  localStorage.setItem("debate_id", debateId);
  pendingDraft = null;
  resetDraftUI();
  render(data);
  if (auto) startAuto();
}

/* ---------------- 困惑 → 议题提炼 ---------------- */

async function runExtract() {
  const problem = document.getElementById("f_problem").value.trim();
  if (!problem) return;
  const box = document.getElementById("draftBox");
  box.hidden = false;
  box.innerHTML = '<div class="draft-loading">🔍 裁判正在理解你的困惑…</div>';
  try {
    const res = await fetch("/api/extract", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: problem,
        mock_mode: document.getElementById("f_mock").checked,
      }),
    });
    const data = await res.json();
    if (data.error) {
      box.innerHTML = `<div class="draft-error">${esc(data.error)}</div>`;
      return;
    }
    if (data.topic) document.getElementById("f_topic").value = data.topic;
    if (data.pro_side) document.getElementById("f_pro").value = data.pro_side;
    if (data.con_side) document.getElementById("f_con").value = data.con_side;
    pendingDraft = {
      draft_id: data.draft_id || "",
      user_context_soft: data.user_context_soft || "",
    };
    box.innerHTML =
      '<div class="draft-title">✨ 裁判已经理解你的困惑，请确认后开始</div>' +
      (data.user_context_soft ? `<div class="draft-soft">${esc(data.user_context_soft)}</div>` : "") +
      (data.note ? `<div class="draft-note">双方的真正分歧：${esc(data.note)}</div>` : "") +
      '<div class="draft-hint">议题与双方立场已自动填到下面，可直接修改；确认无误后点「✅ 确认，开始讨论」。</div>';
    box.hidden = false;
    updateStartButtons();
  } catch (e) {
    box.innerHTML = `<div class="draft-error">提炼失败：${esc(String(e))}</div>`;
  }
}

function resetDraftUI() {
  const box = document.getElementById("draftBox");
  if (box) {
    box.hidden = true;
    box.innerHTML = "";
  }
  updateStartButtons();
}

function updateStartButtons() {
  const b1 = document.getElementById("btn_start");
  const b2 = document.getElementById("btn_autostart");
  if (!b1 || !b2) return;
  if (pendingDraft) {
    b1.textContent = "✅ 确认，开始讨论";
    b2.textContent = "✅ 确认并自动讨论";
  } else {
    b1.textContent = "开始讨论（手动）";
    b2.textContent = "自动讨论";
  }
}

function startAuto() {
  if (!debateId) return;
  fetch(`/api/${debateId}/auto/start`, { method: "POST" }).then(() => {
    autoRunning = true;
    renderControls(snapshot);
    openEvents();
  });
}

function stopAuto() {
  if (debateId) fetch(`/api/${debateId}/auto/stop`, { method: "POST" });
  autoRunning = false;
  if (es) {
    es.close();
    es = null;
  }
  hideThinking();
  renderControls(snapshot);
}

function openEvents() {
  if (es) es.close();
  es = new EventSource(`/api/${debateId}/events`);
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.type === "state") {
      snapshot = ev.data;
      render(ev.data);
    } else if (ev.type === "thinking") {
      showThinking(ev.label);
    } else if (ev.type === "done") {
      autoRunning = false;
      hideThinking();
      if (es) es.close();
      es = null;
      renderControls(snapshot);
    } else if (ev.type === "error") {
      alert(ev.message);
      autoRunning = false;
      if (es) es.close();
      es = null;
      renderControls(snapshot);
    }
  };
}

async function fetchState() {
  const res = await fetch(`/api/state/${debateId}`);
  if (res.status === 404) {
    alert("讨论不存在");
    return;
  }
  render(await res.json());
}

document.addEventListener("DOMContentLoaded", async () => {
  await loadMeta();

  document.getElementById("btn_start").onclick = () => start(false);
  document.getElementById("btn_autostart").onclick = () => start(true);
  document.getElementById("btn_new").onclick = () => {
    if (es) es.close();
    es = null;
    autoRunning = false;
    debateId = null;
    localStorage.removeItem("debate_id");
    snapshot = null;
    pendingDraft = null;
    resetDraftUI();
    render(null);
  };
  // 改动困惑描述后，之前提炼的草稿就过期了，需要重新提炼
  document.getElementById("f_problem").addEventListener("input", () => {
    if (pendingDraft) {
      pendingDraft = null;
      resetDraftUI();
    }
  });
  function doLoad() {
    const id = document.getElementById("f_load").value;
    if (!id) return;
    debateId = id;
    localStorage.setItem("debate_id", id);
    document.getElementById("exportLinks").innerHTML = "";
    fetchState();
  }
  document.getElementById("f_load").onchange = doLoad;
  document.getElementById("btn_load").onclick = doLoad;
  document.getElementById("btn_ask").onclick = askJudge;
  document.getElementById("btn_note").onclick = async () => {
    const text = document.getElementById("f_note").value.trim();
    if (!text) return;
    const res = await fetch(`/api/${debateId}/note`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, is_public: true }),
    });
    const data = await res.json();
    document.getElementById("f_note").value = "";
    if (data.error) {
      alert(data.error);
      return;
    }
    render(data);
  };

  if (debateId) fetchState();
});
