/* ragbench · painel — JavaScript puro, sem bibliotecas e sem nada da internet.
 *
 * Tudo o que vem dos resultados (perguntas, respostas dos modelos, documentos) entra na página como TEXTO
 * (textContent), nunca como HTML: uma resposta de modelo não consegue injetar marcação nem script.
 *
 * As regras de "falhou?" e de "em que etapa?" são as mesmas de ragbench/analysis.py (is_fail, stage_of,
 * compute). O teste tests/test_ui.py roda este arquivo no Node e confere que os
 * dois lados dão os mesmos números.
 */
(function () {
"use strict";

const BOOT = (typeof window !== "undefined" && window.RB_BOOT) || {};
const EMB = (typeof window !== "undefined" && window.RB_DATA) || null;      // dados embutidos (arquivo exportado)
const STAGES = ["busca", "raciocinio", "formato", "indefinido"];
const STAGE_LABEL = { ok: "acertou", busca: "busca", raciocinio: "raciocínio", formato: "formato", indefinido: "indefinido", misto: "etapas diferentes" };
const STAGE_HELP = {
  busca: "faltou evidência: nem todos os parágrafos de suporte do gabarito foram recuperados",
  raciocinio: "todos os parágrafos de suporte foram recuperados e a resposta saiu errada",
  formato: "a resposta certa está lá, mas não na forma pedida (o gabarito aparece dentro da resposta, ou a resposta é um pedaço dele)",
  indefinido: "o dataset não anota os parágrafos de suporte e o gabarito não é localizável nos documentos",
};
const PAPERS = ["ragcritic", "dynakrag"];
const PAPER_LABEL = { ragcritic: "RAG-Critic", dynakrag: "DynaKRAG" };
const KIND_LABEL = { busca: "busca", evidencia: "evidência", raciocinio: "raciocínio", acao: "ação", critico: "crítico", plano: "plano",
  execucao: "execução", resposta: "resposta", nota: "nota" };
const FAMILY_ORDER = ["base", "principal", "ablacao", "variante", "baseline", "outro"];
const FAMILY_LABEL = { principal: "método principal", base: "ponto de partida", baseline: "baselines do artigo", ablacao: "ablações", variante: "variantes e diagnósticos", outro: "outros" };

// ============================================================================ regras (espelho de analysis.py)
function isFail(c, metric, thr) {
  if (metric === "em") return c.e < 1;
  if (metric === "acc") return c.a < 1;
  return c.f < thr;
}

function stageOf(c, metric, thr) {
  if (!isFail(c, metric, thr)) return "ok";
  if (c.a >= 1 || c.n) return "formato";
  const cov = c.s != null ? c.s >= 1 : c.i;
  if (cov == null) return "indefinido";
  return cov ? "raciocinio" : "busca";
}

function mean(xs) { return xs.length ? xs.reduce(function (a, b) { return a + b; }, 0) / xs.length : NaN; }

/* Estatísticas para um conjunto de métodos — mesma conta de analysis.compute(). */
function compute(matrix, keys, metric, thr) {
  const cells = matrix.cells, qs = matrix.questions;
  keys = keys.filter(function (k) { return cells[k]; });
  const baseOf = {};
  matrix.methods.forEach(function (m) { baseOf[m.key] = m.base; });
  const per = {};
  keys.forEach(function (k) {
    const col = cells[k], idx = [];
    col.forEach(function (c, i) { if (c) idx.push(i); });
    const st = { ok: 0, busca: 0, raciocinio: 0, formato: 0, indefinido: 0 };
    idx.forEach(function (i) { st[stageOf(col[i], metric, thr)]++; });
    const n = idx.length, fails = n - st.ok, b = baseOf[k];
    let breaks = 0, fixes = 0, nBase = 0;
    if (b && cells[b]) idx.forEach(function (i) {
      const cb = cells[b][i];
      if (!cb) return;
      nBase++;
      const fb = isFail(cb, metric, thr), fm = isFail(col[i], metric, thr);
      if (!fb && fm) breaks++;
      if (fb && !fm) fixes++;
    });
    per[k] = {
      n: n, em: mean(idx.map(function (i) { return col[i].e; })), f1: mean(idx.map(function (i) { return col[i].f; })),
      fails: fails, rate: n ? fails / n : NaN,
      stages: { busca: st.busca, raciocinio: st.raciocinio, formato: st.formato, indefinido: st.indefinido },
      abst: idx.filter(function (i) { return col[i].b && isFail(col[i], metric, thr); }).length,
      tokens: mean(idx.filter(function (i) { return col[i].t != null; }).map(function (i) { return col[i].t; })),
      calls: mean(idx.filter(function (i) { return col[i].c != null; }).map(function (i) { return col[i].c; })),
      base: b && cells[b] ? b : null, n_base: nBase, breaks: breaks, fixes: fixes,
    };
  });
  const common = [];
  if (keys.length) for (let i = 0; i < qs.length; i++) if (keys.every(function (k) { return cells[k][i]; })) common.push(i);
  const n = common.length, fail = {}, stg = {};
  keys.forEach(function (k) {
    fail[k] = common.map(function (i) { return isFail(cells[k][i], metric, thr); });
    stg[k] = common.map(function (i) { return stageOf(cells[k][i], metric, thr); });
  });
  const counts = common.map(function (_, j) { return keys.filter(function (k) { return fail[k][j]; }).length; });
  const allIdx = [];
  counts.forEach(function (c, j) { if (keys.length && c === keys.length) allIdx.push(common[j]); });
  const sizes = keys.map(function (k) { return fail[k].filter(Boolean).length; });
  const pairs = [];
  for (let x = 0; x < keys.length; x++) for (let y = x + 1; y < keys.length; y++) {
    const a = keys[x], b = keys[y], na = sizes[x], nb = sizes[y];
    let both = 0;
    for (let j = 0; j < n; j++) if (fail[a][j] && fail[b][j]) both++;
    const union = na + nb - both;
    pairs.push({ a: a, b: b, fa: na, fb: nb, both: both, jaccard: union ? both / union : NaN });
  }
  const same = {};
  counts.forEach(function (c, j) {
    if (!keys.length || c !== keys.length) return;
    const first = stg[keys[0]][j];
    const key = keys.every(function (k) { return stg[k][j] === first; }) ? first : "misto";
    same[key] = (same[key] || 0) + 1;
  });
  const byType = {};
  common.forEach(function (i, j) {
    const t = qs[i].type || "—";
    if (!byType[t]) { byType[t] = { n: 0 }; keys.forEach(function (k) { byType[t][k] = 0; }); }
    byType[t].n++;
    keys.forEach(function (k) { if (fail[k][j]) byType[t][k]++; });
  });
  const hist = [];
  for (let k = 0; k <= keys.length; k++) hist.push(counts.filter(function (c) { return c === k; }).length);
  return {
    keys: keys, metric: metric, thr: thr, methods: per, n: n, common: common,
    all_fail: { n: allIdx.length, idx: allIdx },
    none_fail: counts.filter(function (c) { return c === 0; }).length, hist: hist, pairs: pairs, same_stage: same, by_type: byType,
  };
}

const CORE = { isFail: isFail, stageOf: stageOf, compute: compute };
if (typeof module !== "undefined" && module.exports) { module.exports = CORE; return; }      // usado pelo teste no Node
if (typeof document === "undefined") return;

// ============================================================================ DOM
const PROPS = { value: 1, checked: 1, disabled: 1, selected: 1, hidden: 1, type: 1, title: 1, placeholder: 1, open: 1,
                href: 1, src: 1, alt: 1, id: 1, name: 1, min: 1, max: 1, step: 1, tabIndex: 1, htmlFor: 1, target: 1, rel: 1 };
function add(el, kid) {
  if (kid == null || kid === false) return;
  if (Array.isArray(kid)) kid.forEach(function (k) { add(el, k); });
  else el.appendChild(kid instanceof Node ? kid : document.createTextNode(String(kid)));
}
function h(tag, attrs) {
  const el = document.createElement(tag);
  if (attrs) Object.keys(attrs).forEach(function (k) {
    const v = attrs[k];
    if (v == null || v === false) return;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k === "style") el.style.cssText = v;
    else if (k.slice(0, 2) === "on") el.addEventListener(k.slice(2), v);
    else if (PROPS[k]) el[k] = v;
    else el.setAttribute(k, v === true ? "" : v);
  });
  for (let i = 2; i < arguments.length; i++) add(el, arguments[i]);
  return el;
}
function $(id) { return document.getElementById(id); }
function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }

function fnum(x, d) {
  if (x == null || x !== x) return "—";
  return x.toLocaleString("pt-BR", { minimumFractionDigits: d == null ? 1 : d, maximumFractionDigits: d == null ? 1 : d });
}
function fpct(x, d) { return x == null || x !== x ? "—" : fnum(100 * x, d == null ? 1 : d); }
function fint(x) { return x == null || x !== x ? "—" : Math.round(x).toLocaleString("pt-BR"); }
function clipText(s, n) { s = String(s == null ? "" : s).replace(/\s+/g, " ").trim(); return s.length <= n ? s : s.slice(0, n - 1) + "…"; }
function plural(n, one, many) { return n + " " + (n === 1 ? one : many); }
const MEM = {};
function store(k, v) {
  // No arquivo exportado a preferência vale só enquanto a página está aberta: ele pode ser aberto de file:// ou
  // embutido em outra página, onde o armazenamento do navegador não é confiável (e o arquivo não deve gravar nada
  // no navegador de quem o recebe).
  if (BOOT.export) { if (v === undefined) return k in MEM ? MEM[k] : null; MEM[k] = v; return null; }
  try { if (v === undefined) return localStorage.getItem("rb." + k); localStorage.setItem("rb." + k, v); } catch (e) { /* sem armazenamento: segue sem lembrar */ }
  return null;
}

// ============================================================================ estado
const S = {
  profile: BOOT.profile, dataset: null, tab: "testagem",
  metric: "f1", thr: 0.5, sel: [], userSel: false,
  sort: "dif", show: "falhas", who: "any", stage: "", type: "", q: "", page: 0, bases: false,
  dataTab: "perguntas", split: "eval", dq: "", dpage: 0, cq: "", cpage: 0, ebTag: "", ebLabel: "", ebShow: "", ebPage: 0,
  report: null,
};
let ST = null;          // /api/state
let MX = null;          // matriz do dataset atual
let RUN = null;         // estado da rodada
let PLAN = null;
const FORM = { datasets: null, papers: PAPERS.slice(), tiers: [1] };
let pollTimer = null, lastRunState = null;

// ============================================================================ API
function keyOf(path, params) {
  const ks = Object.keys(params || {}).filter(function (k) { return params[k] != null && params[k] !== ""; }).sort();
  return path + "?" + ks.map(function (k) { return k + "=" + params[k]; }).join("&");
}
function api(path, params) {
  if (EMB) {
    const k = keyOf(path, params);
    return k in EMB ? Promise.resolve(EMB[k]) : Promise.reject(new Error("Este dado não foi incluído no arquivo exportado."));
  }
  const ks = Object.keys(params || {}).filter(function (k) { return params[k] != null && params[k] !== ""; });
  const qs = ks.map(function (k) { return encodeURIComponent(k) + "=" + encodeURIComponent(params[k]); }).join("&");
  return fetch(path + (qs ? "?" + qs : ""), { cache: "no-store" }).then(function (r) {
    return r.json().then(function (j) { if (!r.ok) throw new Error(j.error || ("erro " + r.status)); return j; });
  });
}
function post(path, body) {
  return fetch(path, { method: "POST", headers: { "Content-Type": "application/json", "X-RagBench-Token": BOOT.token || "" },
                       body: JSON.stringify(body || {}) })
    .then(function (r) { return r.json().then(function (j) { if (!r.ok) throw new Error(j.error || ("erro " + r.status)); return j; }); });
}

// ============================================================================ pequenos componentes
const tip = $("tip");
function showTip(ev, nodes) {
  clear(tip); add(tip, nodes); tip.hidden = false;
  const r = tip.getBoundingClientRect();
  let x, y;
  if (ev.clientX != null && ev.type !== "focus") { x = ev.clientX + 14; y = ev.clientY + 16; }
  else { const t = ev.target.getBoundingClientRect(); x = t.left; y = t.bottom + 8; }
  if (x + r.width > window.innerWidth - 8) x = Math.max(8, window.innerWidth - r.width - 8);
  if (y + r.height > window.innerHeight - 8) y = Math.max(8, y - r.height - 36);
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
function hideTip() { tip.hidden = true; }
function withTip(el, fn) {
  el.addEventListener("pointermove", function (ev) { showTip(ev, fn()); });
  el.addEventListener("pointerleave", hideTip);
  el.addEventListener("focus", function (ev) { showTip(ev, fn()); });
  el.addEventListener("blur", hideTip);
  return el;
}
function stageChip(stage) {
  if (stage === "ok") return h("span", { class: "mark ok" }, h("i", { "aria-hidden": "true" }, "✓"), "acertou");
  return h("span", { class: "stage", title: STAGE_HELP[stage] || "" }, h("span", { class: "dot " + stage }), STAGE_LABEL[stage] || stage);
}
function failMark(fail) {
  return h("span", { class: "mark " + (fail ? "bad" : "ok") }, h("i", { "aria-hidden": "true" }, fail ? "✗" : "✓"), fail ? "falhou" : "acertou");
}
function legend(items) {
  return h("div", { class: "legend" }, items.map(function (s) {
    return h("span", { title: STAGE_HELP[s] || "" }, h("span", { class: "dot " + s }), s === "ok" ? "acertou" : STAGE_LABEL[s]);
  }));
}
function seg(options, value, onpick) {
  return h("div", { class: "seg", role: "group" }, options.map(function (o) {
    return h("button", { type: "button", "aria-pressed": String(o[0] === value), onclick: function () { onpick(o[0]); }, title: o[2] || null }, o[1]);
  }));
}
function table(head, rows, opts) {
  opts = opts || {};
  const thead = h("thead", null, h("tr", null, head.map(function (c) {
    return typeof c === "string" ? h("th", { text: c }) : h("th", { class: c.cls || null, title: c.title || null }, c.label);
  })));
  return h("div", { class: "tablewrap" }, h("table", null, thead, h("tbody", null, rows)));
}
function pager(page, total, per, onpage) {
  const pages = Math.max(1, Math.ceil(total / per));
  if (pages <= 1) return null;
  return h("div", { class: "pager" },
    h("span", null, "página " + (page + 1) + " de " + pages),
    h("button", { class: "btn sm", type: "button", disabled: page <= 0, onclick: function () { onpage(page - 1); } }, "‹ anterior"),
    h("button", { class: "btn sm", type: "button", disabled: page >= pages - 1, onclick: function () { onpage(page + 1); } }, "próxima ›"));
}
function emptyBox(title, text, extra) { return h("div", { class: "card" }, h("div", { class: "empty" }, h("b", { text: title }), text, extra ? h("div", { style: "margin-top:10px" }, extra) : null)); }
function minfo(key) { return (MX && MX.methods.filter(function (m) { return m.key === key; })[0]) || { key: key, label: key, short: key, family: "outro", paper: "" }; }
function mlabel(key) { return minfo(key).short || minfo(key).label; }
function criterionText() {
  return S.metric === "em" ? "EM = 0" : S.metric === "acc" ? "o gabarito não aparece na resposta" : "F1 < " + fnum(S.thr, 2);
}

// ============================================================================ carga
function hasResults() { return !!(ST && ST.datasets.some(function (d) { return d.key === S.dataset && d.n_methods > 0; })); }
function loadProfiles() {       // a lista de perfis muda quando uma rodada termina (no arquivo exportado é fixa)
  if (EMB) return Promise.resolve();
  return api("/api/profiles", {}).then(function (ps) { BOOT.profiles = ps; }).catch(function () {});
}
function loadState() {
  return api("/api/state", { profile: S.profile }).then(function (st) {
    const first = !ST;
    ST = st;
    if (first || S.metricProfile !== S.profile) { S.metric = st.compare.metric; S.thr = st.compare.thr; S.metricProfile = S.profile; }
    const keys = st.datasets.map(function (d) { return d.key; });
    if (keys.indexOf(S.dataset) < 0) {
      const withRes = st.datasets.filter(function (d) { return d.n_methods > 0; });
      const pref = withRes.filter(function (d) { return d.compare; })[0] || withRes[0] || st.datasets[0];
      S.dataset = pref ? pref.key : null;
    }
    if (!FORM.datasets) FORM.datasets = st.datasets.filter(function (d) { return d.papers.length; }).map(function (d) { return d.key; });
  });
}
function loadMatrix() {
  if (!S.dataset || !hasResults()) { MX = null; return Promise.resolve(); }
  return api("/api/matrix", { profile: S.profile, dataset: S.dataset }).then(function (m) {
    MX = m;
    const avail = m.methods.map(function (x) { return x.key; });
    let sel = S.sel.filter(function (k) { return avail.indexOf(k) >= 0; });
    if (!S.userSel || !sel.length) sel = m.defaults.slice();
    if (!sel.length) sel = avail.slice(0, 4);
    S.sel = avail.filter(function (k) { return sel.indexOf(k) >= 0; });
    if (S.who !== "any" && S.sel.indexOf(S.who) < 0) S.who = "any";
  }).catch(function (e) { MX = null; S.loadError = e.message; });
}
function reloadAll() { S.loadError = null; return loadProfiles().then(loadState).then(loadMatrix).then(render).catch(fatal); }
function fatal(e) {
  const app = clear($("app"));
  app.appendChild(emptyBox("Não deu para carregar", String(e && e.message || e),
    EMB ? null : h("button", { class: "btn", type: "button", onclick: reloadAll }, "Tentar de novo")));
}

// ============================================================================ topo, abas, filtros
const TABS = [["testagem", "1", "Testagem"], ["falhas", "2", "Falhas"], ["comum", "3", "Falhas em comum"], ["dados", "", "Dados"], ["relatorios", "", "Relatórios"]];
function renderChrome() {
  const sp = clear($("selProfile"));
  (BOOT.profiles || []).forEach(function (p) {
    sp.appendChild(h("option", { value: p.name, selected: p.name === S.profile },
      p.name + (p.n_files ? " · " + plural(p.n_files, "resultado", "resultados") : " · sem resultados")));
  });
  const sd = clear($("selDataset"));
  (ST ? ST.datasets : []).forEach(function (d) {
    sd.appendChild(h("option", { value: d.key, selected: d.key === S.dataset }, d.name + (d.n_methods ? "" : " · sem resultados")));
  });
  $("topNote").textContent = BOOT.export ? "arquivo exportado em " + (BOOT.exported || "") + " · somente leitura" : "";
  $("btnReload").hidden = !!BOOT.export;
  const tabs = clear($("tabs"));
  TABS.forEach(function (t) {
    tabs.appendChild(h("button", { class: "tab", type: "button", role: "tab", "aria-selected": String(S.tab === t[0]),
      onclick: function () { S.tab = t[0]; S.page = 0; render(); window.scrollTo(0, 0); } }, t[1] ? h("span", { class: "n" }, t[1] + " ·") : null, t[2]));
  });
  renderFilters();
  $("foot").textContent = ST ? "Pasta: " + (BOOT.root || "") + " · perfil " + S.profile + " · gerador " + ST.models.generator +
    " · crítico " + ST.models.critic + (ST.backend !== "ollama" ? " · backend " + ST.backend : "") : "";
}
function renderFilters() {
  const box = clear($("filters"));
  const on = MX && ["testagem", "falhas", "comum"].indexOf(S.tab) >= 0;
  box.hidden = !on;
  if (!on) return;
  // --- critério
  const out = h("output", { class: "nowrap", style: "min-width:64px;font-weight:600" }, criterionText());
  const slider = h("input", { type: "range", min: "0.1", max: "1", step: "0.05", value: String(S.thr), "aria-label": "Corte do F1",
    hidden: S.metric !== "f1", oninput: function () { S.thr = parseFloat(slider.value); out.textContent = criterionText(); S.page = 0; renderApp(); } });
  box.appendChild(h("div", { class: "group" },
    h("label", { class: "lbl", title: "Um teste conta como falha quando a resposta fica abaixo deste critério." }, "Falhou quando"),
    seg([["f1", "F1 baixo", "O F1 entre a resposta e o gabarito fica abaixo do corte"], ["em", "EM = 0", "A resposta não é idêntica ao gabarito (depois de normalizar)"],
         ["acc", "sem o gabarito", "O gabarito não aparece dentro da resposta"]], S.metric,
      function (v) { S.metric = v; S.page = 0; renderFilters(); renderApp(); }),
    slider, out));
  // --- métodos
  const chips = h("div", { class: "chips" });
  S.sel.forEach(function (k) {
    const m = minfo(k);
    chips.appendChild(h("span", { class: "chip", title: m.label + (m.desc ? " — " + m.desc : "") }, m.short,
      h("button", { type: "button", "aria-label": "Tirar " + m.label + " da comparação", onclick: function () { toggleMethod(k, false); } }, "×")));
  });
  const btn = h("button", { class: "btn sm", type: "button", onclick: function (ev) { ev.stopPropagation(); openPicker(btn); } }, "Escolher métodos…");
  box.appendChild(h("div", { class: "group", style: "position:relative" }, h("label", { class: "lbl" }, "Comparando"), chips, btn));
}
function toggleMethod(key, on) {
  const set = S.sel.filter(function (k) { return k !== key; });
  if (on) set.push(key);
  setSelection(set);
}
function setSelection(keys) {
  S.sel = MX.methods.map(function (m) { return m.key; }).filter(function (k) { return keys.indexOf(k) >= 0; });
  S.userSel = true; S.page = 0;
  if (S.who !== "any" && S.sel.indexOf(S.who) < 0) S.who = "any";
  renderFilters(); renderApp();
}
let pickerEl = null;
function closePicker() { if (pickerEl) { pickerEl.remove(); pickerEl = null; document.removeEventListener("click", outsidePicker); } }
function outsidePicker(ev) { if (pickerEl && !pickerEl.contains(ev.target)) closePicker(); }
function openPicker(anchor) {
  if (pickerEl) { closePicker(); return; }
  const pop = h("div", { class: "pop", role: "dialog", "aria-label": "Escolher métodos" });
  function fill() {
    clear(pop);
    pop.appendChild(h("div", { class: "row" },
      h("b", null, "Métodos na comparação"), h("span", { class: "grow" }),
      h("button", { class: "btn sm", type: "button", onclick: function () { S.userSel = false; setSelection(MX.defaults.length ? MX.defaults : S.sel); S.userSel = false; fill(); } }, "Padrão"),
      h("button", { class: "btn sm", type: "button", onclick: closePicker }, "Fechar")));
    pop.appendChild(h("p", { class: "small muted", style: "margin:4px 0 0" },
      "Padrão: o RAG-Critic, o DynaKRAG e a base simples (Standard RAG)."));
    PAPERS.forEach(function (paper) {
      const ms = MX.methods.filter(function (m) { return m.paper === paper; });
      if (!ms.length) return;
      const keys = ms.map(function (m) { return m.key; });
      pop.appendChild(h("h4", null, PAPER_LABEL[paper] + "  ",
        h("button", { class: "btn ghost sm", type: "button", onclick: function () { setSelection(S.sel.concat(keys)); fill(); } }, "todos"),
        h("button", { class: "btn ghost sm", type: "button", onclick: function () { setSelection(S.sel.filter(function (k) { return keys.indexOf(k) < 0; })); fill(); } }, "nenhum")));
      const grid = h("div", { class: "mgrid" });
      FAMILY_ORDER.forEach(function (fam) {
        const fs = ms.filter(function (m) { return m.family === fam; });
        if (!fs.length) return;
        grid.appendChild(h("div", { class: "fam" }, FAMILY_LABEL[fam]));
        fs.forEach(function (m) {
          const cb = h("input", { type: "checkbox", checked: S.sel.indexOf(m.key) >= 0, onchange: function () { toggleMethod(m.key, cb.checked); } });
          grid.appendChild(h("label", { title: m.desc || "" }, cb, h("span", null, m.label)));
        });
      });
      pop.appendChild(grid);
    });
  }
  fill();
  anchor.parentNode.appendChild(pop);
  pop.style.top = (anchor.offsetTop + anchor.offsetHeight + 6) + "px";
  pop.style.left = "0px";
  pickerEl = pop;
  setTimeout(function () { document.addEventListener("click", outsidePicker); }, 0);
}

function render() { closePicker(); renderChrome(); renderApp(); saveHash(); }
function renderApp() {
  hideTip();
  const app = clear($("app"));
  if (S.loadError) app.appendChild(h("div", { class: "note" }, "Não deu para ler os resultados deste dataset: " + S.loadError));
  ({ testagem: viewTestagem, falhas: viewFalhas, comum: viewComum, dados: viewDados, relatorios: viewRelatorios })[S.tab](app);
}
function needResults(app) {
  if (MX) return false;
  app.appendChild(emptyBox("Ainda não há resultados para este dataset neste perfil",
    BOOT.readonly ? "Este arquivo não traz resultados para ele." : "Rode os testes na aba 1 · Testagem (ou escolha outro perfil ou dataset no topo).",
    BOOT.readonly ? null : h("button", { class: "btn", type: "button", onclick: function () { S.tab = "testagem"; render(); } }, "Ir para Testagem")));
  return true;
}

// ============================================================================ 1 · Testagem
let RUNUI = null;
function viewTestagem(app) {
  if (!BOOT.readonly) { app.appendChild(runCard()); refreshPlan(); pollRun(true); }
  if (!MX) {
    app.appendChild(emptyBox("Sem resultados para " + (dsName() || "este dataset") + " no perfil " + S.profile,
      BOOT.readonly ? "" : "Confira o plano acima e clique em “Rodar os testes”. Dá para parar e continuar depois: nada é perdido."));
    return;
  }
  app.appendChild(resultsCard());
}
function check(label, checked, onchange, small) {
  const cb = h("input", { type: "checkbox", checked: checked, onchange: function () { onchange(cb.checked); } });
  return h("label", { class: "chk" }, cb, h("span", null, label, small ? h("small", null, " " + small) : null));
}
function runCard() {
  const card = h("section", { class: "card runcard" }, h("h2", null, "Rodar os testes"),
    h("p", { class: "sub" }, "Todos os métodos respondem às mesmas perguntas, com o mesmo modelo. A rodada é retomável: se parar no meio, " +
      "rodar de novo continua de onde parou, e as chamadas ao modelo ficam em cache."));
  const form = h("div", { class: "runform" });
  const fsD = h("fieldset", null, h("legend", null, "Datasets"));
  ST.datasets.filter(function (d) { return d.papers.length; }).forEach(function (d) {
    fsD.appendChild(check(d.name, FORM.datasets.indexOf(d.key) >= 0, function (on) {
      FORM.datasets = FORM.datasets.filter(function (x) { return x !== d.key; }); if (on) FORM.datasets.push(d.key); refreshPlan();
    }, d.prepared ? (d.compare ? "· os dois artigos" : "· só " + d.papers.map(function (p) { return PAPER_LABEL[p]; }).join(", ")) : "· dados ainda não baixados"));
  });
  const fsP = h("fieldset", null, h("legend", null, "Artigos"));
  PAPERS.forEach(function (p) {
    fsP.appendChild(check(PAPER_LABEL[p], FORM.papers.indexOf(p) >= 0, function (on) {
      FORM.papers = FORM.papers.filter(function (x) { return x !== p; }); if (on) FORM.papers.push(p); refreshPlan();
    }, "· " + ST.n_eval[p] + " perguntas"));
  });
  const fsT = h("fieldset", null, h("legend", null, "O que rodar"));
  [[1, "Métodos principais e bases simples", "· o necessário para as abas 2 e 3"], [2, "Ablações", "· o que cada peça do método contribui"],
   [3, "Baselines dos artigos", "· Self-Refine, FLARE, IRCoT, CRAG…"]].forEach(function (t) {
    fsT.appendChild(check(t[1], FORM.tiers.indexOf(t[0]) >= 0, function (on) {
      FORM.tiers = FORM.tiers.filter(function (x) { return x !== t[0]; }); if (on) FORM.tiers.push(t[0]); refreshPlan();
    }, t[2]));
  });
  add(form, [fsD, fsP, fsT]);
  card.appendChild(form);
  RUNUI = {
    plan: h("div", { class: "info" }, "calculando o plano…"),
    state: h("span", { class: "state" }, "—"), detail: h("span", { class: "ink2 small" }),
    jobBar: h("div", { class: "progress", hidden: true }, h("div")), jobLab: h("div", { class: "small ink2" }),
    itBar: h("div", { class: "progress", hidden: true }, h("div")), itLab: h("div", { class: "small ink2 mono" }),
    log: h("pre", { class: "log", tabIndex: 0, "aria-label": "Saída da rodada" }), logBox: h("details", null),
    bRun: h("button", { class: "btn primary", type: "button", onclick: function () { startRun("bench"); } }, "Rodar os testes"),
    bPrep: h("button", { class: "btn", type: "button", onclick: function () { startRun("prepare"); }, title: "Baixa os datasets e monta os índices de busca" }, "Preparar os dados"),
    bDoc: h("button", { class: "btn", type: "button", onclick: function () { startRun("doctor"); }, title: "Confere o Ollama, os modelos e a GPU" }, "Verificar o ambiente"),
    bRep: h("button", { class: "btn", type: "button", onclick: function () { startRun("report"); }, title: "Gera de novo REPORT.md, FALHAS.md e as planilhas" }, "Gerar os relatórios"),
    bStop: h("button", { class: "btn danger", type: "button", hidden: true, onclick: stopRun }, "Parar"),
    seq: -1,
  };
  card.appendChild(RUNUI.plan);
  card.appendChild(h("div", { class: "row" }, RUNUI.bRun, RUNUI.bStop, RUNUI.bPrep, RUNUI.bDoc, RUNUI.bRep, h("span", { class: "grow" }), RUNUI.state, RUNUI.detail));
  add(card, [RUNUI.jobLab, RUNUI.jobBar, RUNUI.itLab, RUNUI.itBar]);
  add(RUNUI.logBox, [h("summary", null, "Saída da rodada"), RUNUI.log]);
  card.appendChild(RUNUI.logBox);
  updateRunUI();
  return card;
}
function refreshPlan() {
  if (!RUNUI) return;
  const ui = RUNUI;
  api("/api/plan", { profile: S.profile, datasets: FORM.datasets.join(","), papers: FORM.papers.join(","), tiers: FORM.tiers.join(",") }).then(function (p) {
    PLAN = p;
    if (ui !== RUNUI) return;
    clear(ui.plan);
    if (!FORM.datasets.length || !FORM.papers.length || !FORM.tiers.length) { ui.plan.textContent = "Marque pelo menos um dataset, um artigo e um grupo de métodos."; return; }
    const done = p.n - p.n_todo;
    add(ui.plan, [h("b", null, plural(p.n, "etapa", "etapas")), done ? " (" + done + " já prontas)" : "", " · cerca de " + fint(p.calls) + " chamadas ao modelo",
      p.calls ? " · ≈ " + (p.hours < 0.1 ? "poucos minutos" : fnum(p.hours, 1) + " h") + " nesta máquina (" +
        (p.measured ? "a partir do que já rodou aqui: " + fnum(p.secs_per_call, 1) + " s por chamada simples" : "sem medição ainda: supondo 3 s por chamada simples") + ")" : "",
      ". É uma estimativa grosseira, que pode errar por um fator de 2; o que já rodou não roda de novo."]);
    if (p.n_stale) ui.plan.appendChild(h("div", { class: "note" },
      plural(p.n_stale, "etapa", "etapas") + " do DynaKRAG " + (p.n_stale === 1 ? "foi gerada" : "foram geradas") +
      " com a regra de parada antiga, que não reproduz o artigo, e " + (p.n_stale === 1 ? "roda" : "rodam") +
      " de novo. Os resultados antigos ficam guardados em ragbench_results/logs/controle_anterior."));
    if (p.missing_data.length) ui.plan.appendChild(h("div", { class: "note" },
      "Faltam os dados de " + p.missing_data.join(", ") + ". Clique em “Preparar os dados” antes de rodar (baixa o dataset e monta o índice de busca)."));
    else if (!done) ui.plan.appendChild(h("div", { class: "small", style: "margin-top:4px" },
      "Primeira vez neste perfil? Clique antes em “Verificar o ambiente”: ele confere o Ollama, os modelos e mede a velocidade desta máquina."));
  }).catch(function (e) { if (ui === RUNUI) ui.plan.textContent = "Não deu para calcular o plano: " + e.message; });
}
function startRun(kind) {
  const body = { kind: kind, profile: S.profile, datasets: FORM.datasets, papers: FORM.papers, tiers: FORM.tiers };
  post("/api/run", body).then(function (r) { RUN = r; if (RUNUI) RUNUI.logBox.open = true; updateRunUI(); pollRun(true); })
    .catch(function (e) { if (RUNUI) { RUNUI.detail.textContent = e.message; } });
}
function stopRun() { post("/api/stop", {}).then(function (r) { RUN = r; updateRunUI(); }).catch(function () {}); }
function pollRun(now) {
  if (BOOT.readonly) return;
  clearTimeout(pollTimer);
  const go = function () {
    api("/api/run", {}).then(function (r) {
      const was = lastRunState;
      RUN = r; lastRunState = r.state;
      updateRunUI();
      const active = r.state === "running" || r.state === "stopping";
      if (!active && (was === "running" || was === "stopping")) { reloadAll(); return; }      // terminou: relê os resultados
      if (active) pollTimer = setTimeout(go, 1500);
      else if (S.tab === "testagem") pollTimer = setTimeout(go, 8000);
    }).catch(function () { pollTimer = setTimeout(go, 5000); });
  };
  if (now) go(); else pollTimer = setTimeout(go, 1500);
}
function updateRunUI() {
  const ui = RUNUI;
  if (!ui || !ui.state.isConnected && !ui.plan.isConnected) return;
  const r = RUN || { state: "idle" };
  const active = r.state === "running" || r.state === "stopping";
  const names = { idle: "Parado", running: "Rodando", stopping: "Parando…", done: "Terminou", failed: "Terminou com erro", stopped: "Interrompido" };
  ui.state.className = "state" + (active ? " running" : "");
  ui.state.textContent = names[r.state] || r.state;
  ui.detail.textContent = r.label ? "· " + r.label + (r.started ? " · " + elapsed(r) : "") : "";
  [ui.bRun, ui.bPrep, ui.bDoc, ui.bRep].forEach(function (b) { b.disabled = active; });
  ui.bStop.hidden = !active;
  const job = active ? r.job : null;
  ui.jobBar.hidden = !job; ui.jobLab.textContent = job ? "Etapa " + job.i + " de " + job.n + " · " + job.key : "";
  if (job) ui.jobBar.firstChild.style.width = Math.round(100 * (job.i - 1) / job.n) + "%";
  const bar = active ? r.bar : null;
  ui.itBar.hidden = !bar;
  ui.itLab.textContent = bar ? bar.desc + " · " + bar.i + "/" + bar.n + " · falta " + bar.eta.trim() : (active && r.bar_raw ? r.bar_raw : "");
  if (bar) ui.itBar.firstChild.style.width = bar.pct + "%";
  if (r.seq !== ui.seq && r.log) {
    const atEnd = ui.log.scrollTop + ui.log.clientHeight >= ui.log.scrollHeight - 30;
    ui.log.textContent = r.log.join("\n");
    if (atEnd) ui.log.scrollTop = ui.log.scrollHeight;
    ui.seq = r.seq;
    if (r.state === "failed") ui.logBox.open = true;
  }
}
function elapsed(r) {
  const s = Math.max(0, Math.round(((r.ended || Date.now() / 1000) - r.started)));
  return s < 90 ? s + " s" : s < 5400 ? Math.round(s / 60) + " min" : fnum(s / 3600, 1) + " h";
}
function resultsCard() {
  const keys = MX.methods.map(function (m) { return m.key; });
  const st = compute(MX, keys, S.metric, S.thr);
  const card = h("section", { class: "card" }, h("h2", null, "Testagem padrão — " + MX.name),
    h("p", { class: "sub" }, plural(MX.questions.length, "pergunta", "perguntas") + ", " + plural(keys.length, "método", "métodos") +
      ". Falha = " + criterionText() + ". Marque os métodos que entram nas abas 2 e 3."));
  const rows = [];
  const isMain = function (m) { return m.family === "base" || m.family === "principal" || S.sel.indexOf(m.key) >= 0; };
  const hidden = MX.methods.filter(function (m) { return !isMain(m); }).length;
  PAPERS.forEach(function (paper) {
    const ms = MX.methods.filter(function (m) { return m.paper === paper && (S.allMethods || isMain(m)); });
    if (!ms.length) return;
    rows.push(h("tr", { class: "grp" }, h("td", { colspan: "10" }, PAPER_LABEL[paper] +
      (ST.retrieval[paper] ? " · busca: " + (ST.retrieval[paper].kind === "bm25" ? "BM25" : ST.retrieval[paper].model) : ""))));
    ms.forEach(function (m) {
      const s = st.methods[m.key];
      const cb = h("input", { type: "checkbox", checked: S.sel.indexOf(m.key) >= 0, "aria-label": "Comparar " + m.label, onchange: function () { toggleMethod(m.key, cb.checked); } });
      const base = s.base ? minfo(s.base).short : null;
      rows.push(h("tr", null, h("td", null, cb),
        h("td", { title: m.desc || "" }, m.label, " ", m.n_errors ? h("span", { class: "tag", title: "Perguntas em que o método deu erro de execução (ficam fora da análise e são tentadas de novo na próxima rodada)" }, m.n_errors + " com erro") : null),
        h("td", { class: "num" }, s.n), h("td", { class: "num" }, fpct(s.em)),
        h("td", null, h("div", { class: "inbar" }, h("div", { class: "track" }, h("div", { class: "fill", style: "width:" + (100 * (s.f1 || 0)).toFixed(1) + "%" })), h("span", { class: "v" }, fpct(s.f1)))),
        h("td", { class: "num" }, s.fails + " (" + fpct(s.rate, 0) + "%)"),
        h("td", { class: "num", title: "busca / raciocínio / formato" + (s.stages.indefinido ? " / indefinido" : "") },
          s.stages.busca + " / " + s.stages.raciocinio + " / " + s.stages.formato + (s.stages.indefinido ? " / " + s.stages.indefinido : "")),
        h("td", { class: "num", title: base ? "Em relação a " + base + ": quebra = a base acerta e o método falha; conserto = o contrário" : "" }, base ? s.breaks + " / " + s.fixes : "—"),
        h("td", { class: "num" }, fint(s.tokens)), h("td", { class: "num" }, fnum(s.calls, 1))));
    });
  });
  card.appendChild(table(["", "Método", { label: "Testes", cls: "num" }, { label: "EM %", cls: "num" }, "F1 %", { label: "Falhas", cls: "num" },
    { label: "Busca / Racioc. / Formato", cls: "num", title: "Em que etapa as falhas aparecem" }, { label: "Quebras / Consertos", cls: "num", title: "Em relação ao ponto de partida do próprio método" },
    { label: "Tokens", cls: "num", title: "Tokens por pergunta (prompt + resposta), em média" }, { label: "Chamadas", cls: "num", title: "Chamadas ao modelo por pergunta, em média" }], rows));
  if (hidden || S.allMethods) card.appendChild(h("div", { class: "row", style: "margin-top:10px" },
    h("button", { class: "btn sm", type: "button", onclick: function () { S.allMethods = !S.allMethods; renderApp(); } },
      S.allMethods ? "Mostrar só os principais e os escolhidos" : "Mostrar também " + (hidden === 1 ? "o outro método" : "os outros " + hidden + " métodos") + " (ablações, variantes e baselines dos artigos)")));
  return card;
}

// ============================================================================ 2 · Falhas
function viewFalhas(app) {
  if (needResults(app)) return;
  if (!S.sel.length) { app.appendChild(emptyBox("Nenhum método escolhido", "Use “Escolher métodos…” acima.")); return; }
  const st = compute(MX, S.sel, S.metric, S.thr);
  app.appendChild(stagesCard(st));
  app.appendChild(mapCard(st));
  app.appendChild(testsCard(st, false));
}
function stagesCard(st) {
  const card = h("section", { class: "card" }, h("h2", null, "Onde cada método falha — " + MX.name),
    h("p", { class: "sub" }, "Cada barra são todos os testes do método. A parte colorida são as falhas (" + criterionText() + "), separadas pela etapa em que aparecem, medida contra o gabarito."));
  const present = STAGES.filter(function (s) { return st.keys.some(function (k) { return st.methods[k].stages[s]; }); });
  card.appendChild(legend(["ok"].concat(present.length ? present : ["busca", "raciocinio", "formato"])));
  const fig = h("div", { class: "sbars", role: "img", "aria-label": "Barras empilhadas: acertos e falhas por etapa, para cada método" });
  st.keys.forEach(function (k) {
    const s = st.methods[k], m = minfo(k);
    fig.appendChild(h("div", { class: "name", title: m.label }, m.short));
    const bar = h("div", { class: "sbar" });
    [["ok", s.n - s.fails]].concat(STAGES.map(function (x) { return [x, s.stages[x]]; })).forEach(function (p) {
      if (!p[1]) return;
      const share = p[1] / s.n;
      const sg = h("div", { class: "segm " + p[0], style: "flex:" + p[1] + " 1 0", tabIndex: 0 }, share >= 0.07 ? String(p[1]) : "");
      withTip(sg, function () { return [h("b", null, p[1] + " de " + s.n + " (" + fpct(share, 0) + "%)"), h("div", { class: "t2" }, (p[0] === "ok" ? "acertou" : "falha de " + STAGE_LABEL[p[0]]) + " · " + m.label)]; });
      bar.appendChild(sg);
    });
    fig.appendChild(bar);
    fig.appendChild(h("div", { class: "tot" }, s.fails + " falhas de " + s.n + " (" + fpct(s.rate, 0) + "%)"));
  });
  card.appendChild(fig);
  // a mesma informação em tabela
  const tbl = h("details", { style: "margin-top:10px" }, h("summary", null, "Ver como tabela"));
  tbl.appendChild(table(["Método", { label: "Testes", cls: "num" }, { label: "Acertos", cls: "num" }, { label: "Busca", cls: "num" }, { label: "Raciocínio", cls: "num" },
    { label: "Formato", cls: "num" }, { label: "Indefinido", cls: "num" }, { label: "Sem resposta", cls: "num", title: "Falhas em que o modelo não respondeu (vazio, “unknown”, “the documents do not mention…”)" },
    { label: "Quebras", cls: "num" }, { label: "Consertos", cls: "num" }, "Em relação a"],
    st.keys.map(function (k) {
      const s = st.methods[k];
      return h("tr", null, h("td", null, minfo(k).label), h("td", { class: "num" }, s.n), h("td", { class: "num" }, s.n - s.fails),
        STAGES.map(function (x) { return h("td", { class: "num" }, s.stages[x]); }), h("td", { class: "num" }, s.abst),
        h("td", { class: "num" }, s.base ? s.breaks : "—"), h("td", { class: "num" }, s.base ? s.fixes : "—"), h("td", null, s.base ? minfo(s.base).short : "—"));
    })));
  card.appendChild(tbl);
  if (!MX.has_support) card.appendChild(h("div", { class: "info" }, "Este dataset não anota os parágrafos de suporte: a etapa usa só o texto do gabarito (se aparece ou não nos documentos recuperados)."));
  return card;
}
function orderQuestions(st, mode) {
  const cells = MX.cells, idx = MX.questions.map(function (_, i) { return i; }).filter(function (i) { return st.keys.some(function (k) { return cells[k][i]; }); });
  const nf = {};
  idx.forEach(function (i) { nf[i] = st.keys.filter(function (k) { return cells[k][i] && isFail(cells[k][i], S.metric, S.thr); }).length; });
  if (mode === "dif") idx.sort(function (a, b) { return nf[b] - nf[a] || a - b; });
  else if (mode === "tipo") idx.sort(function (a, b) { return (MX.questions[a].type || "").localeCompare(MX.questions[b].type || "") || nf[b] - nf[a] || a - b; });
  return { idx: idx, nf: nf };
}
function cellTip(k, i) {
  const q = MX.questions[i], c = MX.cells[k][i];
  if (!c) return [h("b", null, "sem resultado"), h("div", { class: "t2" }, minfo(k).label)];
  const stg = stageOf(c, S.metric, S.thr);
  return [h("b", null, stg === "ok" ? "acertou" : "falha de " + STAGE_LABEL[stg]), " · F1 " + fpct(c.f, 0),
    h("div", { class: "t2" }, minfo(k).label), h("div", null, clipText(q.q, 110)),
    h("div", { class: "t2" }, "gabarito: " + clipText(q.gold.join(" / "), 60)), h("div", { class: "t2" }, "resposta: " + clipText(c.p || "(vazia)", 60))];
}
function mapCard(st) {
  const card = h("section", { class: "card" }, h("h2", null, "Mapa de falhas"),
    h("p", { class: "sub" }, "Uma coluna por teste, uma linha por método. Colunas coloridas de cima a baixo são testes em que todos falham. Clique num quadrado para abrir o teste."));
  const body = h("div");
  const tools = h("div", { class: "row", style: "margin-bottom:8px" }, h("span", { class: "small ink2" }, "Ordenar os testes por"));
  function draw() {
    clear(body);
    const o = orderQuestions(st, S.sort), n = o.idx.length;
    const cw = Math.max(5, Math.min(22, Math.floor((Math.min(window.innerWidth, 1320) - 290) / Math.max(1, n)) - 2));
    const map = h("div", { class: "map" });
    const rows = {};
    st.keys.forEach(function (k) {
      const cellsEl = h("div", { class: "cells" });
      rows[k] = cellsEl;
      o.idx.forEach(function (i, col) {
        const c = MX.cells[k][i], stg = c ? stageOf(c, S.metric, S.thr) : "vazio";
        const b = h("button", { class: "cell " + (stg === "ok" ? "" : stg), type: "button", style: "width:" + cw + "px", disabled: !c,
          "aria-label": minfo(k).short + ", teste " + (col + 1) + ": " + (stg === "vazio" ? "sem resultado" : stg === "ok" ? "acertou" : "falha de " + STAGE_LABEL[stg]),
          onclick: function () { openTest(MX.questions[i].id, o.idx.map(function (j) { return MX.questions[j].id; })); } });
        b.addEventListener("pointerenter", function () { st.keys.forEach(function (kk) { const e = rows[kk].children[col]; if (e) e.classList.add("hl"); }); });
        b.addEventListener("pointerleave", function () { st.keys.forEach(function (kk) { const e = rows[kk].children[col]; if (e) e.classList.remove("hl"); }); });
        withTip(b, function () { return cellTip(k, i); });
        cellsEl.appendChild(b);
      });
      map.appendChild(h("div", { class: "maprow" }, h("div", { class: "name", title: minfo(k).label }, minfo(k).short), cellsEl));
    });
    body.appendChild(map);
    if (S.sort === "dif") body.appendChild(h("div", { class: "mapfoot", style: "width:" + (n * (cw + 2)) + "px" }, h("span", null, "← falham em mais métodos"), h("span", null, "em menos →")));
  }
  tools.appendChild(seg([["dif", "dificuldade", "Primeiro os testes em que mais métodos falham"], ["ordem", "ordem do dataset"], ["tipo", "tipo de pergunta"]], S.sort,
    function (v) { S.sort = v; renderApp(); }));
  card.appendChild(tools);
  const present = STAGES.filter(function (x) { return st.keys.some(function (k) { return st.methods[k].stages[x]; }); });
  card.appendChild(legend(["ok"].concat(present)));
  card.appendChild(body);
  draw();
  return card;
}
function testsCard(st, onlyCommon) {
  const cells = MX.cells, qs = MX.questions;
  const card = h("section", { class: "card" }, h("h2", null, onlyCommon ? "Os testes em que todos falham" : "Os testes"),
    h("p", { class: "sub" }, onlyCommon ? "Clique numa linha para ver a pergunta, o gabarito, os documentos e o passo a passo de cada método."
      : "Escolha o que listar e clique numa linha para abrir o teste: pergunta, gabarito, parágrafos de suporte, resposta e passo a passo de cada método."));
  const body = h("div");
  const types = Array.from(new Set(qs.map(function (q) { return q.type || ""; }))).filter(Boolean).sort();
  const search = h("input", { type: "search", placeholder: "buscar na pergunta, no gabarito ou nas respostas", value: S.q, style: "min-width:260px",
    "aria-label": "Buscar nos testes", oninput: function () { S.q = search.value; S.page = 0; draw(); } });
  const tools = h("div", { class: "row", style: "margin-bottom:10px" });
  if (!onlyCommon) {
    tools.appendChild(seg([["falhas", "falhas"], ["quebras", "quebras", "A base do método acerta e o método falha"], ["consertos", "consertos", "A base falha e o método acerta"], ["todos", "todos"]],
      S.show, function (v) { S.show = v; S.page = 0; renderApp(); }));
    const who = h("select", { "aria-label": "De qual método", onchange: function () { S.who = who.value; S.page = 0; draw(); } },
      h("option", { value: "any", selected: S.who === "any" }, "de qualquer método escolhido"),
      st.keys.map(function (k) { return h("option", { value: k, selected: S.who === k }, "de " + minfo(k).short); }));
    tools.appendChild(who);
    const stg = h("select", { "aria-label": "Etapa", onchange: function () { S.stage = stg.value; S.page = 0; draw(); } },
      h("option", { value: "", selected: !S.stage }, "qualquer etapa"), STAGES.map(function (x) { return h("option", { value: x, selected: S.stage === x }, "etapa: " + STAGE_LABEL[x]); }));
    tools.appendChild(stg);
  }
  if (types.length > 1) {
    const ty = h("select", { "aria-label": "Tipo de pergunta", onchange: function () { S.type = ty.value; S.page = 0; draw(); } },
      h("option", { value: "", selected: !S.type }, "qualquer tipo"), types.map(function (x) { return h("option", { value: x, selected: S.type === x }, x); }));
    tools.appendChild(ty);
  }
  tools.appendChild(search);
  const count = h("span", { class: "small ink2" });
  tools.appendChild(h("span", { class: "grow" })); tools.appendChild(count);
  const csvBtn = h("button", { class: "btn sm", type: "button", title: "Baixar esta lista como planilha (CSV)" }, "Baixar CSV");
  tools.appendChild(csvBtn);
  card.appendChild(tools); card.appendChild(body);

  function matches(i) {
    const q = qs[i];
    if (S.type && (q.type || "") !== S.type) return false;
    const who = onlyCommon || S.who === "any" ? st.keys : [S.who];
    const present = who.filter(function (k) { return cells[k][i]; });
    if (!present.length) return false;
    if (onlyCommon) { if (st.all_fail.idx.indexOf(i) < 0) return false; }
    else if (S.show !== "todos") {
      const ok = present.some(function (k) {
        const c = cells[k][i], f = isFail(c, S.metric, S.thr);
        if (S.show === "falhas") return f && (!S.stage || stageOf(c, S.metric, S.thr) === S.stage);
        const b = minfo(k).base, cb = b && cells[b] ? cells[b][i] : null;
        if (!cb) return false;
        const fb = isFail(cb, S.metric, S.thr);
        return (S.show === "quebras" ? (!fb && f) : (fb && !f)) && (!S.stage || !f || stageOf(c, S.metric, S.thr) === S.stage);
      });
      if (!ok) return false;
    }
    if (S.q) {
      const t = S.q.toLowerCase();
      const hay = [q.q, q.gold.join(" "), q.id].concat(st.keys.map(function (k) { return cells[k][i] ? cells[k][i].p : ""; })).join(" ").toLowerCase();
      if (hay.indexOf(t) < 0) return false;
    }
    return true;
  }
  function draw() {
    clear(body);
    const o = orderQuestions(st, "dif");
    const idx = o.idx.filter(matches), per = 25;
    count.textContent = plural(idx.length, "teste", "testes");
    csvBtn.onclick = function () { downloadCsv(idx, st.keys); };
    if (!idx.length) {
      const why = S.show === "quebras" || S.show === "consertos" ? " Quebras e consertos só existem para métodos que partem de uma base (ex.: RAG-Critic sobre o Standard RAG)." : "";
      body.appendChild(h("div", { class: "empty" }, h("b", null, "Nenhum teste com esses filtros"), "Mude os filtros acima." + why));
      return;
    }
    const pageIdx = idx.slice(S.page * per, S.page * per + per);
    const ids = idx.map(function (i) { return qs[i].id; });
    const rows = pageIdx.map(function (i) {
      const q = qs[i];
      const tr = h("tr", { class: "click", tabIndex: 0, onclick: function () { openTest(q.id, ids); },
        onkeydown: function (ev) { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); openTest(q.id, ids); } } },
        h("td", { class: "num" }, o.nf[i] + "/" + st.keys.filter(function (k) { return cells[k][i]; }).length),
        h("td", { class: "q" }, clipText(q.q, 170), q.type ? h("div", null, h("span", { class: "tag" }, q.type)) : null),
        h("td", { class: "ans" }, h("span", { class: "gold" }, clipText(q.gold.slice(0, 2).join(" / "), 70))));
      st.keys.forEach(function (k) {
        const c = cells[k][i];
        if (!c) { tr.appendChild(h("td", { class: "ans muted" }, "—")); return; }
        tr.appendChild(h("td", { class: "ans" }, h("span", { class: "p", title: c.p }, c.p || "(vazia)"), stageChip(stageOf(c, S.metric, S.thr))));
      });
      return tr;
    });
    body.appendChild(table([{ label: "Falharam", cls: "num", title: "Em quantos dos métodos escolhidos este teste falha" }, "Pergunta", "Gabarito"]
      .concat(st.keys.map(function (k) { return { label: minfo(k).short, title: minfo(k).label }; })), rows));
    add(body, pager(S.page, idx.length, per, function (p) { S.page = p; draw(); }));
  }
  draw();
  return card;
}
function downloadCsv(idx, keys) {
  const esc = function (v) { v = String(v == null ? "" : v); return /[";\n\r]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; };
  const head = ["dataset", "id", "tipo", "pergunta", "gabarito"];
  keys.forEach(function (k) { const n = minfo(k).short; head.push(n + " · resposta", n + " · F1", n + " · etapa"); });
  const lines = [head.map(esc).join(";")];
  idx.forEach(function (i) {
    const q = MX.questions[i], row = [MX.dataset, q.id, q.type, q.q, q.gold.join(" | ")];
    keys.forEach(function (k) {
      const c = MX.cells[k][i];
      if (!c) row.push("", "", "sem resultado"); else row.push(c.p, String(c.f).replace(".", ","), stageOf(c, S.metric, S.thr));
    });
    lines.push(row.map(esc).join(";"));
  });
  const blob = new Blob(["﻿" + lines.join("\r\n")], { type: "text/csv;charset=utf-8" });
  const a = h("a", { href: URL.createObjectURL(blob) });
  a.setAttribute("download", "testes_" + MX.dataset + "_" + S.profile + ".csv");
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(function () { URL.revokeObjectURL(a.href); }, 2000);
}

// ============================================================================ 3 · Falhas em comum
function viewComum(app) {
  if (needResults(app)) return;
  const cross = S.sel.filter(function (k) { return S.bases || minfo(k).family !== "base"; });
  const hasBase = S.sel.some(function (k) { return minfo(k).family === "base"; });
  const head = h("section", { class: "card" }, h("h2", null, "Os métodos falham nos mesmos testes? — " + MX.name));
  const chips = h("div", { class: "chips", style: "margin:6px 0" }, h("span", { class: "small ink2" }, "Cruzando:"));
  cross.forEach(function (k, j) { if (j) chips.appendChild(h("span", { class: "ink2" }, "×")); chips.appendChild(h("span", { class: "chip", title: minfo(k).label }, minfo(k).short)); });
  if (hasBase) chips.appendChild(check("incluir as bases simples", S.bases, function (on) { S.bases = on; renderApp(); }));
  head.appendChild(chips);
  app.appendChild(head);
  if (cross.length < 2) {
    const others = MX.methods.filter(function (m) { return m.family !== "base" && cross.indexOf(m.key) < 0; }).length;
    head.appendChild(h("div", { class: "empty" }, h("b", null, "Escolha pelo menos dois métodos para cruzar"),
      others ? "Use “Escolher métodos…” acima" + (hasBase && !S.bases ? " ou marque “incluir as bases simples”." : ".")
             : "Neste dataset só " + (cross.length ? minfo(cross[0]).short : "as bases simples") + " rodou até agora. Rode os outros métodos na aba 1 · Testagem" +
               (hasBase && !S.bases ? ", ou marque “incluir as bases simples” para cruzar com a base." : ".")));
    return;
  }
  const st = compute(MX, cross, S.metric, S.thr), af = st.all_fail;
  const nested = cross.filter(function (k) { return cross.indexOf(minfo(k).base) >= 0; });
  if (nested.length) head.appendChild(h("div", { class: "note" }, minfo(nested[0]).short + " é construído sobre " + minfo(minfo(nested[0]).base).short +
    ": parte das falhas em comum entre os dois é por construção. Para comparar, cruze métodos independentes."));
  if (!st.n) { head.appendChild(h("div", { class: "empty" }, h("b", null, "Esses métodos ainda não responderam às mesmas perguntas"), "Rode os que faltam na aba 1.")); return; }
  head.appendChild(h("div", { class: "tiles" },
    h("div", { class: "tile hero" }, h("div", { class: "lab" }, "Falham em todos"), h("div", { class: "val" }, af.n), h("div", { class: "det" }, "de " + st.n + " testes (" + fpct(af.n / st.n, 0) + "%)")),
    h("div", { class: "tile" }, h("div", { class: "lab" }, "Nenhum falha"), h("div", { class: "val" }, st.none_fail), h("div", { class: "det" }, "testes que todos acertam"))));
  if (af.n === 0) head.appendChild(h("p", { class: "sub", style: "margin-top:6px" }, "Não há nenhum teste em que todos falhem."));
  // --- em quantos métodos cada teste falha + mesma etapa
  const cols = h("div", { class: "cols" });
  const c1 = h("section", { class: "card", style: "margin:0" }, h("h2", null, "Em quantos métodos cada teste falha"), h("p", { class: "sub" }, "Número de testes por quantidade de métodos que falharam nele."));
  const mx = Math.max.apply(null, st.hist.concat([1]));
  const chart = h("div", { class: "colchart", role: "img", "aria-label": "Colunas: número de testes por quantidade de métodos que falharam" });
  const axis = h("div", { class: "colaxis" });
  st.hist.forEach(function (v, k) {
    const bar = h("div", { class: "bar", style: "height:" + Math.round(112 * v / mx) + "px", tabIndex: 0 });
    withTip(bar, function () { return [h("b", null, plural(v, "teste", "testes")), h("div", { class: "t2" }, k === 0 ? "nenhum método falha" : k === cross.length ? "todos falham" : "falham " + k + " de " + cross.length)]; });
    chart.appendChild(h("div", { class: "col" }, h("div", { class: "v" }, v), bar));
    axis.appendChild(h("span", null, k === 0 ? "nenhum" : k === cross.length ? "todos" : String(k)));
  });
  add(c1, [chart, axis]);
  cols.appendChild(c1);
  const c2 = h("section", { class: "card", style: "margin:0" }, h("h2", null, "Falham no mesmo elemento?"),
    h("p", { class: "sub" }, "Nos testes em que todos falham: a falha aparece na mesma etapa em todos os métodos?"));
  if (!af.n) c2.appendChild(h("div", { class: "empty" }, "Sem falhas em comum para analisar."));
  else {
    const order = STAGES.concat(["misto"]).filter(function (s) { return st.same_stage[s]; });
    c2.appendChild(legend(order));
    const bar = h("div", { class: "sbar", style: "max-width:520px" });
    order.forEach(function (s) {
      const v = st.same_stage[s], sg = h("div", { class: "segm " + s, style: "flex:" + v + " 1 0", tabIndex: 0 }, v / af.n >= 0.07 ? String(v) : "");
      withTip(sg, function () { return [h("b", null, v + " de " + af.n), h("div", { class: "t2" }, s === "misto" ? "a etapa muda de um método para outro" : "todos falham na etapa: " + STAGE_LABEL[s])]; });
      bar.appendChild(sg);
    });
    c2.appendChild(bar);
    const sameN = af.n - (st.same_stage.misto || 0);
    c2.appendChild(h("p", { style: "margin-top:10px" }, h("b", null, sameN + " de " + af.n), " falhas em comum acontecem na mesma etapa em todos os métodos" +
      (st.same_stage.misto ? "; em " + st.same_stage.misto + " a etapa muda de um método para outro." : ".")));
    const lst = h("ul", { class: "small ink2", style: "margin:6px 0 0;padding-left:18px" });
    order.forEach(function (s) { lst.appendChild(h("li", null, st.same_stage[s] + " — " + (s === "misto" ? "etapas diferentes" : "todos na " + STAGE_LABEL[s] + ": " + STAGE_HELP[s]))); });
    c2.appendChild(lst);
  }
  cols.appendChild(c2);
  app.appendChild(cols);
  // --- pares
  const pc = h("section", { class: "card" }, h("h2", null, "Par a par"), h("p", { class: "sub" },
    "Jaccard = testes em que os dois falham ÷ testes em que pelo menos um falha (1 = falham exatamente nos mesmos)."));
  pc.appendChild(table(["Par", { label: "Falhas A", cls: "num" }, { label: "Falhas B", cls: "num" }, { label: "Falham os dois", cls: "num" },
    { label: "Jaccard", cls: "num" }],
    st.pairs.map(function (p) {
      return h("tr", null, h("td", null, minfo(p.a).short + " × " + minfo(p.b).short), h("td", { class: "num" }, p.fa), h("td", { class: "num" }, p.fb),
        h("td", { class: "num" }, h("b", null, p.both)), h("td", { class: "num" }, fnum(p.jaccard, 2)));
    })));
  app.appendChild(pc);
  const types = Object.keys(st.by_type);
  if (types.length > 1) {
    const tc = h("section", { class: "card" }, h("h2", null, "Por tipo de pergunta"), h("p", { class: "sub" }, "Quantos testes de cada tipo cada método falha."));
    types.sort(function (a, b) { return st.by_type[b].n - st.by_type[a].n; });
    tc.appendChild(table(["Tipo", { label: "Testes", cls: "num" }].concat(cross.map(function (k) { return { label: minfo(k).short, cls: "num" }; })),
      types.map(function (t) {
        const v = st.by_type[t];
        return h("tr", null, h("td", null, t), h("td", { class: "num" }, v.n), cross.map(function (k) { return h("td", { class: "num" }, v[k] + " (" + fpct(v[k] / v.n, 0) + "%)"); }));
      })));
    app.appendChild(tc);
  }
  app.appendChild(testsCard(st, true));
}

// ============================================================================ gaveta: um teste, todos os métodos
let DR = null;
function openTest(id, list) {
  DR = { id: id, list: list && list.length ? list : [id], all: DR ? DR.all : false, back: DR ? DR.back : document.activeElement };
  renderDrawer();
}
function closeDrawer() {
  const back = DR && DR.back;
  DR = null; clear($("drawerHost"));
  if (back && back.focus && back.isConnected) back.focus();
}
function stepDrawer(d) {
  if (!DR) return;
  const p = DR.list.indexOf(DR.id) + d;
  if (p >= 0 && p < DR.list.length) { DR.id = DR.list[p]; renderDrawer(); }
}
function renderDrawer() {
  hideTip();
  const host = clear($("drawerHost")), pos = DR.list.indexOf(DR.id);
  const body = h("div", { class: "dbody" }, h("p", { class: "muted", style: "padding:16px 0" }, "carregando…"));
  const title = h("div", { class: "grow" }, h("div", { class: "small ink2" }, (MX ? MX.name : S.dataset) + " · teste " + (pos + 1) + " de " + DR.list.length));
  const bClose = h("button", { class: "btn", type: "button", onclick: closeDrawer, "aria-label": "Fechar" }, "Fechar");
  const head = h("div", { class: "dhead" }, title,
    h("button", { class: "btn", type: "button", disabled: pos <= 0, onclick: function () { stepDrawer(-1); }, title: "Teste anterior (←)" }, "‹"),
    h("button", { class: "btn", type: "button", disabled: pos >= DR.list.length - 1, onclick: function () { stepDrawer(1); }, title: "Próximo teste (→)" }, "›"), bClose);
  add(host, [h("div", { class: "scrim", onclick: closeDrawer }), h("aside", { class: "drawer", role: "dialog", "aria-modal": "true", "aria-label": "Detalhe do teste" }, head, body)]);
  bClose.focus();
  const id = DR.id;
  api("/api/test", { profile: S.profile, dataset: S.dataset, id: id }).then(function (d) {
    if (!DR || DR.id !== id) return;
    title.appendChild(h("h2", null, d.question));
    fillDrawer(clear(body), d);
  }).catch(function (e) { clear(body).appendChild(h("div", { class: "empty" }, h("b", null, "Não deu para abrir este teste"), e.message)); });
}
function docItem(d, key) {
  const doc = d.docs[key];
  if (!doc) return null;
  const marks = h("span", { class: "marks" }, doc.sup ? h("span", { class: "tag", title: "Parágrafo de suporte anotado no gabarito" }, "suporte") : null,
    doc.ans ? h("span", { class: "tag", title: "O texto do gabarito aparece neste parágrafo" }, "contém o gabarito") : null);
  return h("li", null, h("details", { class: "doc" }, h("summary", null, doc.title || "(sem título)", marks), h("div", null, doc.text)));
}
function fillDrawer(body, d) {
  const info = h("section", { class: "card" });
  info.appendChild(h("dl", { class: "kv" },
    h("dt", null, "Gabarito"), h("dd", { class: "gold" }, d.answers.length ? d.answers.join("  ·  ") : "(sem gabarito)"),
    d.type ? [h("dt", null, "Tipo"), h("dd", null, d.type + (d.hops ? " · " + d.hops + " saltos" : ""))] : null,
    h("dt", null, "Identificador"), h("dd", { class: "mono" }, d.id)));
  if (d.support.length) {
    const sup = h("details", { open: true, style: "margin-top:10px" }, h("summary", null, "Parágrafos de suporte do gabarito (" + d.support.length + ")"));
    if (d.support_docs.length) sup.appendChild(h("ul", { class: "doclist" }, d.support_docs.map(function (k) { return docItem(d, k); })));
    else sup.appendChild(h("p", { class: "small ink2" }, d.support.join(" · ") + (d.has_corpus ? "" : " (o corpus não está nesta pasta: só os títulos)")));
    info.appendChild(sup);
  } else info.appendChild(h("p", { class: "small ink2", style: "margin-top:8px" }, "Este dataset não anota parágrafos de suporte."));
  body.appendChild(info);
  const have = Object.keys(d.methods);
  if (!have.length) { body.appendChild(h("div", { class: "empty" }, "Nenhum método respondeu a esta pergunta ainda.")); return; }
  const order = MX ? MX.methods.map(function (m) { return m.key; }).filter(function (k) { return have.indexOf(k) >= 0; }) : have;
  const chosen = order.filter(function (k) { return S.sel.indexOf(k) >= 0; });
  const show = DR.all || !chosen.length ? order : chosen;
  if (order.length > chosen.length && chosen.length) body.appendChild(h("div", { class: "row", style: "margin-top:12px" },
    seg([["sel", "métodos escolhidos (" + chosen.length + ")"], ["all", "todos (" + order.length + ")"]], DR.all ? "all" : "sel",
      function (v) { DR.all = v === "all"; fillDrawer(clear(body), d); })));
  show.forEach(function (k) { body.appendChild(methodCard(d, k)); });
  if (BOOT.export && MX && S.sel.some(function (k) { return have.indexOf(k) < 0 && MX.cells[k]; }))
    body.appendChild(h("div", { class: "info" }, "O detalhe de alguns métodos escolhidos não foi incluído neste arquivo exportado. No painel com servidor (python -m ragbench ui) aparecem todos."));
}
function methodCard(d, key) {
  const m = d.methods[key], inf = m.info || minfo(key), stg = stageOf(m, S.metric, S.thr);
  const card = h("article", { class: "mcard" });
  card.appendChild(h("header", null, h("h3", null, inf.label), failMark(stg !== "ok"), stg !== "ok" ? stageChip(stg) : null,
    h("span", { class: "grow" }), h("span", { class: "small ink2 nowrap" }, "EM " + fnum(m.e, 0) + " · F1 " + fpct(m.f, 0) +
      Object.keys(m.meta || {}).map(function (x) { return " · " + x + " " + fnum(Number(m.meta[x]), Number(m.meta[x]) % 1 ? 1 : 0); }).join(""))));
  const b = h("div", { class: "mbody" });
  b.appendChild(h("div", { class: "small ink2" }, "Resposta"));
  b.appendChild(h("div", { class: "answer" }, m.p || "(vazia)"));
  const facts = [];
  if (m.s != null) facts.push("Parágrafos de suporte recuperados: " + fpct(m.s, 0) + "%");
  if (m.i != null) facts.push("O texto do gabarito " + (m.i ? "aparece" : "não aparece") + " nos documentos recuperados");
  if (m.b) facts.push("O modelo não respondeu");
  if (stg !== "ok") facts.push("Etapa da falha: " + STAGE_LABEL[stg] + " — " + STAGE_HELP[stg]);
  if (facts.length) b.appendChild(h("ul", { class: "small ink2", style: "margin:0 0 6px;padding-left:18px" }, facts.map(function (f) { return h("li", null, f); })));
  if (inf.desc) b.appendChild(h("p", { class: "small ink2", style: "margin-bottom:6px" }, inf.desc));
  if (m.docs && m.docs.length) b.appendChild(h("details", null, h("summary", null, "Documentos recuperados (" + m.docs.length + ")"),
    h("ul", { class: "doclist" }, m.docs.map(function (k) { return docItem(d, k); }))));
  if (m.steps && m.steps.length) {
    const ol = h("ol", { class: "steps" });
    m.steps.forEach(function (s) {
      const li = h("li", { class: s.k }, h("div", { class: "st" }, h("span", { class: "kind" }, KIND_LABEL[s.k] || s.k), s.t));
      if (s.x) li.appendChild(h("pre", { tabIndex: 0 }, s.x));
      if (s.d && s.d.length) li.appendChild(h("div", { class: "chips", style: "margin-top:4px" }, s.d.map(function (k) {
        const doc = d.docs[k];
        return doc ? h("span", { class: "chip", title: clipText(doc.text, 300) }, clipText(doc.title, 40), doc.sup ? h("span", { class: "tag" }, "suporte") : null) : null;
      })));
      ol.appendChild(li);
    });
    b.appendChild(h("details", { open: true }, h("summary", null, "Passo a passo (" + m.steps.length + ")"), ol));
  }
  card.appendChild(b);
  return card;
}

// ============================================================================ Dados
function viewDados(app) {
  app.appendChild(h("div", { style: "padding:14px 0 0" }, seg([["perguntas", "Perguntas e gabaritos"], ["corpus", "Corpus (parágrafos)"],
    ["errorbench", "RAG-Error Bench (teste do crítico)"], ["sobre", "Sobre os dados"]], S.dataTab, function (v) { S.dataTab = v; renderApp(); })));
  ({ perguntas: dataQuestions, corpus: dataCorpus, errorbench: dataErrorBench, sobre: dataAbout })[S.dataTab](app);
}
function serverOnly(app, what) {
  app.appendChild(emptyBox(what + " não cabe no arquivo exportado", "Abra o painel com servidor na pasta do projeto: python -m ragbench ui"));
}
function dataQuestions(app) {
  const card = h("section", { class: "card" }, h("h2", null, "Perguntas e gabaritos — " + (dsName())),
    h("p", { class: "sub" }, "As perguntas de avaliação são as dos testes; as de treino só servem para treinar o controlador do DynaKRAG. Clique numa pergunta para ver os parágrafos de suporte e o que cada método respondeu."));
  const body = h("div"), count = h("span", { class: "small ink2" });
  const search = h("input", { type: "search", placeholder: "buscar na pergunta, no gabarito ou no id", value: S.dq, style: "min-width:280px", "aria-label": "Buscar perguntas",
    oninput: function () { S.dq = search.value; S.dpage = 0; draw(); } });
  card.appendChild(h("div", { class: "row", style: "margin-bottom:10px" },
    BOOT.export ? null : seg([["eval", "avaliação"], ["train", "treino"]], S.split, function (v) { S.split = v; S.dpage = 0; renderApp(); }), search, h("span", { class: "grow" }), count));
  card.appendChild(body);
  app.appendChild(card);
  let data = null;
  function draw() {
    clear(body);
    if (!data) return;
    const t = S.dq.trim().toLowerCase();
    const items = data.items.filter(function (r) { return !t || (r.id + " " + r.q + " " + r.gold.join(" ") + " " + r.type).toLowerCase().indexOf(t) >= 0; });
    const nUsed = Math.max.apply(null, PAPERS.map(function (p) { return data.used[p] || 0; }));
    count.textContent = plural(items.length, "pergunta", "perguntas") + " de " + data.total + (S.split === "eval" ? " · as " + Math.min(nUsed, data.total) + " primeiras entram nos testes deste perfil" : "");
    if (!data.total) { body.appendChild(h("div", { class: "empty" }, h("b", null, "Os dados deste dataset não estão nesta pasta"), BOOT.readonly ? "" : "Use “Preparar os dados” na aba 1 · Testagem.")); return; }
    const per = 30, ids = items.map(function (r) { return r.id; });
    const rows = items.slice(S.dpage * per, S.dpage * per + per).map(function (r) {
      const click = S.split === "eval";
      return h("tr", { class: click ? "click" : null, tabIndex: click ? 0 : null, onclick: click ? function () { openTest(r.id, ids); } : null,
        onkeydown: click ? function (ev) { if (ev.key === "Enter") openTest(r.id, ids); } : null },
        h("td", { class: "num" }, r.n), h("td", { class: "q" }, r.q), h("td", { class: "ans" }, h("span", { class: "gold" }, clipText(r.gold.join(" / "), 90))),
        h("td", null, r.type ? h("span", { class: "tag" }, r.type) : ""), h("td", { class: "small ink2" }, clipText(r.sup.join(" · "), 110) || "—"),
        h("td", null, r.used ? h("span", { class: "mark ok" }, h("i", null, "✓"), "nos testes") : h("span", { class: "muted small" }, "fora deste perfil")));
    });
    body.appendChild(table([{ label: "Nº", cls: "num" }, "Pergunta", "Gabarito (rótulo)", "Tipo", "Parágrafos de suporte", "Uso"], rows));
    add(body, pager(S.dpage, items.length, per, function (p) { S.dpage = p; draw(); }));
  }
  body.appendChild(h("p", { class: "muted" }, "carregando…"));
  api("/api/questions", { profile: S.profile, dataset: S.dataset, split: S.split }).then(function (d) { data = d; draw(); })
    .catch(function (e) { clear(body).appendChild(h("div", { class: "empty" }, e.message)); });
}
function dsName() { const d = ST && ST.datasets.filter(function (x) { return x.key === S.dataset; })[0]; return d ? d.name : (S.dataset || ""); }
let corpusTimer = null;
function dataCorpus(app) {
  if (BOOT.export) return serverOnly(app, "O corpus");
  const card = h("section", { class: "card" }, h("h2", null, "Corpus — " + dsName()),
    h("p", { class: "sub" }, "Os parágrafos em que os métodos buscam: os de suporte e os distratores das perguntas de avaliação, de treino e de um grupo extra. Todos os métodos buscam no mesmo corpus."));
  const body = h("div"), count = h("span", { class: "small ink2" });
  const search = h("input", { type: "search", placeholder: "palavras do título ou do texto", value: S.cq, style: "min-width:320px", "aria-label": "Buscar no corpus",
    oninput: function () { S.cq = search.value; S.cpage = 0; clearTimeout(corpusTimer); corpusTimer = setTimeout(load, 250); } });
  card.appendChild(h("div", { class: "row", style: "margin-bottom:10px" }, search, h("span", { class: "grow" }), count));
  card.appendChild(body); app.appendChild(card);
  function load() {
    api("/api/corpus", { profile: S.profile, dataset: S.dataset, q: S.cq, offset: S.cpage * 30, limit: 30 }).then(function (d) {
      clear(body);
      if (d.missing) { body.appendChild(h("div", { class: "empty" }, h("b", null, "O corpus deste dataset não está nesta pasta"), "Use “Preparar os dados” na aba 1 · Testagem.")); count.textContent = ""; return; }
      count.textContent = plural(d.matched, "parágrafo", "parágrafos") + " de " + d.total;
      if (!d.items.length) { body.appendChild(h("div", { class: "empty" }, "Nenhum parágrafo com essas palavras.")); return; }
      body.appendChild(h("ul", { class: "doclist flat" }, d.items.map(function (x) {
        return h("li", { style: "padding:8px 0;border-bottom:1px solid var(--grid)" }, h("div", null, h("b", null, x.title || "(sem título)"), " ",
          x.sup ? h("span", { class: "tag", title: "Parágrafo de suporte de " + plural(x.sup, "pergunta de avaliação", "perguntas de avaliação") }, "suporte de " + x.sup) : null,
          " ", h("span", { class: "mono muted" }, x.id)), h("div", { class: "ink2" }, x.text));
      })));
      add(body, pager(S.cpage, d.matched, 30, function (p) { S.cpage = p; load(); }));
    }).catch(function (e) { clear(body).appendChild(h("div", { class: "empty" }, e.message)); });
  }
  load();
}
function dataErrorBench(app) {
  if (BOOT.export) return serverOnly(app, "O RAG-Error Bench");
  const card = h("section", { class: "card" }, h("h2", null, "RAG-Error Bench — o teste do crítico do RAG-Critic"),
    h("p", { class: "sub" }, "Cada exemplo é uma pergunta, 5 passagens e a resposta de um modelo; o rótulo diz se a resposta está correta e, se não, o tipo de erro. O crítico é julgado duas vezes: com o prompt do arquivo oficial e com o recuo do prompt trocado (20 ↔ 16 espaços), sem mudar nenhuma palavra."));
  const body = h("div");
  card.appendChild(body); app.appendChild(card);
  function load() {
    api("/api/errorbench", { profile: S.profile, tag: S.ebTag, offset: S.ebPage * 20, limit: 20, label: S.ebLabel, show: S.ebShow }).then(function (d) {
      clear(body);
      if (!d.tags.length) { body.appendChild(h("div", { class: "empty" }, h("b", null, "O RAG-Error Bench ainda não rodou neste perfil"), "Ele faz parte dos métodos principais do RAG-Critic (aba 1 · Testagem).")); return; }
      const s = d.summary;
      body.appendChild(h("div", { class: "tiles" },
        h("div", { class: "tile" }, h("div", { class: "lab" }, "Exemplos avaliados"), h("div", { class: "val" }, s.n), h("div", { class: "det" }, "modelo: " + d.tag)),
        h("div", { class: "tile" }, h("div", { class: "lab" }, "Acerto com o prompt oficial"), h("div", { class: "val" }, fpct(s.n ? s.ok_original / s.n : NaN, 0) + "%"), h("div", { class: "det" }, s.ok_original + " de " + s.n)),
        h("div", { class: "tile" }, h("div", { class: "lab" }, "Acerto com o recuo trocado"), h("div", { class: "val" }, s.n_swapped ? fpct(s.ok_swapped / s.n_swapped, 0) + "%" : "—"), h("div", { class: "det" }, s.n_swapped ? s.ok_swapped + " de " + s.n_swapped : "não rodou")),
        h("div", { class: "tile" }, h("div", { class: "lab" }, "Julgamentos que mudaram"), h("div", { class: "val" }, s.n_swapped ? s.flipped : "—"), h("div", { class: "det" }, s.n_swapped ? "de " + s.n_swapped + ", só por trocar o recuo" : ""))));
      const bi = Object.keys(s.by_indent).sort();
      if (bi.length) body.appendChild(h("p", { class: "small ink2", style: "margin:0 0 10px" }, "Recuo do prompt → julgamento do crítico: " +
        bi.map(function (k) { return k.replace("→", " espaços → ") + " (" + s.by_indent[k] + ")"; }).join(" · ")));
      const tools = h("div", { class: "row", style: "margin-bottom:10px" });
      if (d.tags.length > 1) {
        const sel = h("select", { "aria-label": "Modelo", onchange: function () { S.ebTag = sel.value; S.ebPage = 0; load(); } }, d.tags.map(function (t) { return h("option", { value: t, selected: t === d.tag }, t); }));
        tools.appendChild(sel);
      }
      tools.appendChild(seg([["", "todos"], ["Correct", "rótulo: correta"], ["Error", "rótulo: com erro"]], S.ebLabel, function (v) { S.ebLabel = v; S.ebPage = 0; load(); }));
      tools.appendChild(seg([["", "todos"], ["errou", "o crítico errou"], ["mudou", "mudou ao trocar o recuo"]], S.ebShow, function (v) { S.ebShow = v; S.ebPage = 0; load(); }));
      tools.appendChild(h("span", { class: "grow" })); tools.appendChild(h("span", { class: "small ink2" }, plural(d.matched, "exemplo", "exemplos")));
      body.appendChild(tools);
      if (!d.has_items) body.appendChild(h("div", { class: "info" }, "O arquivo do benchmark não está nesta pasta: aparecem só os julgamentos, sem a pergunta e as passagens."));
      d.items.forEach(function (x) {
        const verdict = function (r, name) { return r ? h("span", { class: "nowrap" }, name + " (" + r.indent + " espaços): ", h("b", null, r.judgement), " ", failMark(!r.ok)) : null; };
        const det = h("details", { class: "item" },
          h("summary", null, h("span", { class: "mono" }, x.id), "  ·  rótulo: ", h("b", null, x.label === "Correct" ? "correta" : "com erro"), "  ·  ",
            verdict(x.original, "oficial"), x.swapped ? "  ·  " : null, verdict(x.swapped, "trocado")));
        det.appendChild(h("dl", { class: "kv", style: "margin-top:8px" },
          h("dt", null, "Pergunta"), h("dd", null, x.question || "—"), h("dt", null, "Resposta do modelo"), h("dd", null, x.prediction || "—"),
          x.gold.tag1 ? [h("dt", null, "Tipo de erro (rótulo)"), h("dd", null, [x.gold.tag1, x.gold.tag2, x.gold.tag3].filter(Boolean).map(function (t) { return [].concat(t).join(", "); }).join("  ›  "))] : null,
          x.gold.Error_analysis ? [h("dt", null, "Análise (rótulo)"), h("dd", null, x.gold.Error_analysis)] : null,
          x.original && x.original.tag1 && x.original.tag1.length ? [h("dt", null, "Tags do crítico"), h("dd", null, x.original.tag1.join(", ") + (x.original.tag2 && x.original.tag2.length ? "  ›  " + x.original.tag2.join(", ") : ""))] : null));
        if (x.passages.length) det.appendChild(h("details", null, h("summary", null, "Passagens (" + x.passages.length + ")"), h("ul", { class: "doclist flat" }, x.passages.map(function (p) { return h("li", { class: "ink2 small", style: "padding:4px 0" }, p); }))));
        if (x.original && x.original.raw) det.appendChild(h("details", null, h("summary", null, "Saída do crítico"), h("ol", { class: "steps" },
          h("li", null, h("div", { class: "st" }, "prompt oficial"), h("pre", null, x.original.raw)), x.swapped && x.swapped.raw ? h("li", null, h("div", { class: "st" }, "recuo trocado"), h("pre", null, x.swapped.raw)) : null)));
        body.appendChild(det);
      });
      add(body, pager(S.ebPage, d.matched, 20, function (p) { S.ebPage = p; load(); }));
    }).catch(function (e) { clear(body).appendChild(h("div", { class: "empty" }, e.message)); });
  }
  body.appendChild(h("p", { class: "muted" }, "carregando…"));
  load();
}
function dataAbout(app) {
  const d = ST.datasets.filter(function (x) { return x.key === S.dataset; })[0];
  const card = h("section", { class: "card" }, h("h2", null, "Sobre os dados — " + dsName()));
  if (!d || !d.prepared) card.appendChild(h("div", { class: "empty" }, h("b", null, "Dataset ainda não preparado nesta pasta"), BOOT.readonly ? "" : "Use “Preparar os dados” na aba 1 · Testagem."));
  else {
    const m = d.meta || {};
    card.appendChild(h("dl", { class: "kv" },
      h("dt", null, "Origem das perguntas de avaliação"), h("dd", { class: "mono" }, m.source_eval || "—"),
      h("dt", null, "Origem das perguntas de treino"), h("dd", { class: "mono" }, m.source_train || "—"),
      h("dt", null, "Perguntas"), h("dd", null, fint(m.n_eval) + " de avaliação · " + fint(m.n_train) + " de treino · " + fint(m.n_extra) + " extras (só para acrescentar distratores)"),
      h("dt", null, "Corpus"), h("dd", null, fint(m.n_docs) + " parágrafos (a união dos parágrafos de todas essas perguntas)"),
      h("dt", null, "Semente e data"), h("dd", null, "semente " + m.seed + " · preparado em " + (m.created || "—")),
      h("dt", null, "Índices de busca prontos"), h("dd", { class: "mono" }, d.indexes.length ? d.indexes.join(" · ") : "nenhum ainda"),
      h("dt", null, "Usado por"), h("dd", null, d.papers.map(function (p) { return PAPER_LABEL[p] + " (" + ST.n_eval[p] + " perguntas neste perfil)"; }).join(" · ") || "—")));
  }
  app.appendChild(card);
  const c2 = h("section", { class: "card" }, h("h2", null, "Modelos e busca deste perfil"),
    h("p", { class: "sub" }, "Todos os métodos usam o mesmo gerador. Cada artigo usa o retriever do próprio artigo."));
  const ROLE = { generator: "gerador (responde às perguntas em todos os métodos)", critic: "crítico do RAG-Critic", planner: "planner do RAG-Critic",
    judge: "juiz (sistema de erros do RAG-Critic)", summarizer: "resumidor (sistema de erros do RAG-Critic)" };
  c2.appendChild(h("dl", { class: "kv" }, Object.keys(ST.models).map(function (k) { return [h("dt", null, ROLE[k] || k), h("dd", { class: "mono" }, ST.models[k])]; }),
    PAPERS.map(function (p) { const r = ST.retrieval[p] || {}; return [h("dt", null, "busca · " + PAPER_LABEL[p]), h("dd", { class: "mono" }, r.kind === "bm25" ? "BM25" : (r.model || "—"))]; })));
  app.appendChild(c2);
}

// ============================================================================ Relatórios (markdown simples → DOM, sem innerHTML)
function viewRelatorios(app) {
  const card = h("section", { class: "card" }, h("h2", null, "Relatórios — perfil " + S.profile),
    h("p", { class: "sub" }, "Os mesmos arquivos .md gravados na pasta de resultados" + (ST ? " (" + ST.results + ")" : "") + ". Lá também ficam as planilhas .csv."));
  const nav = h("div", { class: "row", style: "margin-bottom:12px" }), body = h("div", { class: "md" });
  add(card, [nav, body]); app.appendChild(card);
  api("/api/reports", { profile: S.profile }).then(function (list) {
    if (!list.length) { body.appendChild(h("div", { class: "empty" }, h("b", null, "Nenhum relatório ainda"), BOOT.readonly ? "" : "Eles são gerados no fim de cada rodada, ou pelo botão “Gerar os relatórios” na aba 1.")); return; }
    if (!S.report || !list.some(function (r) { return r.path === S.report; })) S.report = (list.filter(function (r) { return r.path === "FALHAS.md"; })[0] || list[0]).path;
    nav.appendChild(seg(list.map(function (r) { return [r.path, r.title, r.path]; }), S.report, function (v) { S.report = v; renderApp(); }));
    return api("/api/report", { profile: S.profile, path: S.report }).then(function (r) {
      renderMarkdown(body, r.text, r.path, list.map(function (x) { return x.path; }));
    });
  }).catch(function (e) { body.appendChild(h("div", { class: "empty" }, e.message)); });
}
function resolvePath(base, rel) {
  const parts = base.split("/").slice(0, -1);
  rel.split("/").forEach(function (p) { if (p === "..") parts.pop(); else if (p && p !== ".") parts.push(p); });
  return parts.join("/");
}
function mdInline(parent, s, ctx) {
  const re = /(`[^`]+`)|(\*\*[^*]+\*\*)|(!\[[^\]]*\]\([^)\s]+\))|(\[[^\]]+\]\([^)\s]+\))|(\*[^*\s][^*]*\*)|(_[^_\s][^_]*_)/g;
  let last = 0, m;
  while ((m = re.exec(s))) {
    const t = m[0];
    if (m[6]) {      // _itálico_ só vale com separador dos dois lados (não em nomes_com_sublinhado)
      const before = s[m.index - 1], after = s[m.index + t.length];
      if ((before && /[\w]/.test(before)) || (after && /[\w]/.test(after))) continue;
    }
    if (m.index > last) parent.appendChild(document.createTextNode(s.slice(last, m.index)));
    last = m.index + t.length;
    if (m[1]) parent.appendChild(h("code", null, t.slice(1, -1)));
    else if (m[2]) { const b = h("b"); mdInline(b, t.slice(2, -2), ctx); parent.appendChild(b); }
    else if (m[3] || m[4]) {
      const mm = /^!?\[([^\]]*)\]\(([^)\s]+)\)$/.exec(t), label = mm[1], url = mm[2];
      if (m[3]) {
        if (BOOT.export || /^[a-z]+:/i.test(url)) parent.appendChild(h("span", { class: "muted" }, "[figura: " + (label || url) + "]"));
        else parent.appendChild(h("img", { alt: label, loading: "lazy", src: "/api/file?profile=" + encodeURIComponent(S.profile) + "&path=" + encodeURIComponent(resolvePath(ctx.path, url)) }));
      } else if (/^https?:\/\//i.test(url)) parent.appendChild(h("a", { href: url, target: "_blank", rel: "noopener noreferrer" }, label));
      else {
        const target = resolvePath(ctx.path, url.split("#")[0]);
        if (ctx.known.indexOf(target) >= 0) parent.appendChild(h("a", { href: "#", onclick: function (ev) { ev.preventDefault(); S.report = target; renderApp(); window.scrollTo(0, 0); } }, label));
        else parent.appendChild(document.createTextNode(label));
      }
    } else { const i = h("em"); mdInline(i, t.slice(1, -1), ctx); parent.appendChild(i); }
  }
  if (last < s.length) parent.appendChild(document.createTextNode(s.slice(last)));
}
function renderMarkdown(root, text, path, known) {
  clear(root);
  const ctx = { path: path, known: known }, lines = String(text || "").replace(/\r/g, "").split("\n");
  const stack = [root];
  const cur = function () { return stack[stack.length - 1]; };
  const cellsOf = function (ln) { return ln.replace(/^\s*\|/, "").replace(/\|\s*$/, "").split("|").map(function (c) { return c.trim(); }); };
  let i = 0;
  while (i < lines.length) {
    let ln = lines[i];
    if (/^```/.test(ln)) {
      const buf = []; i++;
      while (i < lines.length && !/^```/.test(lines[i])) buf.push(lines[i++]);
      i++; cur().appendChild(h("pre", null, h("code", null, buf.join("\n")))); continue;
    }
    const dm = /^\s*<details>\s*(?:<summary>(.*?)<\/summary>)?\s*$/.exec(ln);
    if (dm) { const d = h("details"); if (dm[1] != null) { const sm = h("summary"); mdInline(sm, dm[1], ctx); d.appendChild(sm); } cur().appendChild(d); stack.push(d); i++; continue; }
    const sm2 = /^\s*<summary>(.*?)<\/summary>\s*$/.exec(ln);
    if (sm2 && stack.length > 1) { const sm = h("summary"); mdInline(sm, sm2[1], ctx); cur().appendChild(sm); i++; continue; }
    if (/^\s*<\/details>\s*$/.test(ln)) { if (stack.length > 1) stack.pop(); i++; continue; }
    if (!ln.trim()) { i++; continue; }
    const hm = /^(#{1,6})\s+(.*)$/.exec(ln);
    if (hm) { const el = h("h" + Math.min(4, hm[1].length)); mdInline(el, hm[2], ctx); cur().appendChild(el); i++; continue; }
    if (/^\s*(-{3,}|\*{3,})\s*$/.test(ln)) { cur().appendChild(h("hr")); i++; continue; }
    if (/^\s*\|/.test(ln) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
      const head = cellsOf(ln), align = cellsOf(lines[i + 1]).map(function (c) { return /:$/.test(c) && !/^:/.test(c); });
      i += 2;
      const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) {
        rows.push(h("tr", null, cellsOf(lines[i]).map(function (c, j) { const td = h("td", { class: align[j] ? "num" : null }); mdInline(td, c, ctx); return td; })));
        i++;
      }
      const ths = head.map(function (c, j) { const th = h("th", { class: align[j] ? "num" : null }); mdInline(th, c, ctx); return th; });
      cur().appendChild(h("div", { class: "tablewrap" }, h("table", null, h("thead", null, h("tr", null, ths)), h("tbody", null, rows))));
      continue;
    }
    const lm = /^\s*([*-]|\d+\.)\s+(.*)$/.exec(ln);
    if (lm) {
      const list = h(/\d/.test(lm[1]) ? "ol" : "ul");
      while (i < lines.length) {
        const m2 = /^\s*([*-]|\d+\.)\s+(.*)$/.exec(lines[i]);
        if (!m2) break;
        let item = m2[2]; i++;
        while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*([*-]|\d+\.)\s+/.test(lines[i])) item += " " + lines[i++].trim();
        const li = h("li"); mdInline(li, item, ctx); list.appendChild(li);
      }
      cur().appendChild(list); continue;
    }
    if (/^>\s?/.test(ln)) {
      const bq = h("blockquote"), buf = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) buf.push(lines[i++].replace(/^>\s?/, ""));
      mdInline(bq, buf.join(" "), ctx); cur().appendChild(bq); continue;
    }
    const buf = [];
    while (i < lines.length && lines[i].trim() && !/^(#{1,6}\s|```|\s*\||\s*([*-]|\d+\.)\s+|>\s?|\s*<\/?details>|\s*<summary>)/.test(lines[i])) buf.push(lines[i++].trim());
    if (!buf.length) { buf.push(lines[i++].trim()); }
    const p = h("p"); mdInline(p, buf.join(" "), ctx); cur().appendChild(p);
  }
}

// ============================================================================ início
function applyTheme(t) {
  if (t === "light" || t === "dark") document.documentElement.setAttribute("data-theme", t); else document.documentElement.removeAttribute("data-theme");
  $("btnTheme").textContent = "Tema: " + (t === "light" ? "claro" : t === "dark" ? "escuro" : "automático");
}
function saveHash() { try { history.replaceState(null, "", "#" + [S.tab, S.dataset || "", S.profile || ""].join("/")); } catch (e) { /* file:// em alguns navegadores */ } }
function readHash() {
  const p = decodeURIComponent((location.hash || "").slice(1)).split("/");
  if (TABS.some(function (t) { return t[0] === p[0]; })) S.tab = p[0];
  if (p[1]) S.dataset = p[1];
  if (p[2] && (BOOT.profiles || []).some(function (x) { return x.name === p[2]; })) S.profile = p[2];
}
$("selProfile").addEventListener("change", function (ev) { S.profile = ev.target.value; S.dataset = null; S.userSel = false; S.page = 0; ST = null; FORM.datasets = null; reloadAll(); });
$("selDataset").addEventListener("change", function (ev) { S.dataset = ev.target.value; S.page = 0; S.dpage = 0; S.cpage = 0; S.type = ""; S.loadError = null; loadMatrix().then(render); });
$("btnReload").addEventListener("click", reloadAll);
$("btnTheme").addEventListener("click", function () {
  const curT = store("theme"), next = curT === "light" ? "dark" : curT === "dark" ? "" : "light";
  store("theme", next); applyTheme(next);
});
document.addEventListener("keydown", function (ev) {
  if (ev.key === "Escape") { if (pickerEl) closePicker(); else if (DR) closeDrawer(); return; }
  if (!DR || /^(INPUT|SELECT|TEXTAREA|PRE)$/.test((ev.target.tagName || ""))) return;
  if (ev.key === "ArrowLeft") stepDrawer(-1);
  if (ev.key === "ArrowRight") stepDrawer(1);
});
let resizeTimer = null;
window.addEventListener("resize", function () { clearTimeout(resizeTimer); resizeTimer = setTimeout(function () { if (S.tab === "falhas" && !DR) renderApp(); }, 250); });
applyTheme(store("theme"));
readHash();
reloadAll();

})();
