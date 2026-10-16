"use strict";

const LABELS = {
  baseline_left_corner: "פינת קו הסיום – שמאל",
  corner3_baseline_left: "קו השלוש פוגש את קו הסיום – שמאל",
  lane_baseline_left: "פינת הצבע על קו הסיום – שמאל",
  lane_baseline_right: "פינת הצבע על קו הסיום – ימין",
  corner3_baseline_right: "קו השלוש פוגש את קו הסיום – ימין",
  baseline_right_corner: "פינת קו הסיום – ימין",
  lane_ft_left: "פינת הצבע בקו העונשין – שמאל",
  lane_ft_right: "פינת הצבע בקו העונשין – ימין",
  three_point_top: "ראש קשת השלוש",
  midcourt_left: "קו האמצע – שמאל",
  midcourt_center: "מרכז המגרש",
  midcourt_right: "קו האמצע – ימין",
  rim: "מרכז הטבעת (באוויר)",
};
const ORDER = Object.keys(LABELS);
const STORAGE_KEY = "courtvision.calibration";
const VERDICTS = {
  good: ["החלטה טובה", ""],
  should_pass: ["עדיף היה למסור", "bad"],
  should_shoot: ["עדיף היה לזרוק", "bad"],
  better_pass: ["הייתה מסירה טובה יותר", "warn"],
};

const $ = (id) => document.getElementById(id);
const state = {
  job: null,
  geometry: null,
  landmarks: {},
  points: {}, // landmark -> [x, y] in video pixels, "rim" included
  current: ORDER[0],
  image: null,
  mouse: null,
  events: [],
  showPreview: false,
  lastDone: null, // last finished job, shown again if a re-run fails
};

// ---------- helpers ----------
async function api(path, options = {}) {
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `שגיאה ${res.status}`);
  return data;
}
const postJSON = (path, body) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
function show(id, visible = true) { $(id).classList.toggle("hidden", !visible); }
function setError(id, message) { $(id).textContent = message || ""; show(id, !!message); }
function setStep(step) {
  const order = ["upload", "calibrate", "results"];
  document.querySelectorAll("#steps li").forEach((li) => {
    const i = order.indexOf(li.dataset.step), cur = order.indexOf(step);
    li.classList.toggle("active", i === cur);
    li.classList.toggle("done", i < cur);
  });
}
function floorPoints() {
  const out = {};
  for (const [k, v] of Object.entries(state.points)) if (k !== "rim") out[k] = v;
  return out;
}
function storageGet() { try { return JSON.parse(localStorage.getItem(STORAGE_KEY)); } catch { return null; } }
function storageSet(v) { try { localStorage.setItem(STORAGE_KEY, JSON.stringify(v)); } catch { /* private mode */ } }

// ---------- upload ----------
function upload(file) {
  setError("upload-error", "");
  const form = new FormData();
  form.append("file", file);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/videos");
  show("upload-bar");
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) $("upload-bar").firstElementChild.style.width = `${(100 * e.loaded) / e.total}%`;
  };
  xhr.onload = () => {
    show("upload-bar", false);
    let data = {};
    try { data = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
    if (xhr.status !== 200) return setError("upload-error", data.detail || "ההעלאה נכשלה");
    startCalibration(data);
  };
  xhr.onerror = () => { show("upload-bar", false); setError("upload-error", "ההעלאה נכשלה – האם השרת רץ?"); };
  xhr.send(form);
}

function startCalibration(job) {
  state.job = job;
  state.points = {};
  state.events = [];
  state.showPreview = false;
  state.current = ORDER[0];
  $("preview-img").removeAttribute("src");
  show("preview-img", false);
  show("frame-canvas");
  const img = new Image();
  img.onload = () => {
    state.image = img;
    show("calibrate-section");
    show("upload-section", false);
    setStep("calibrate");
    const saved = storageGet();
    show("load-calib", !!(saved && saved.width === job.width && saved.height === job.height));
    refreshCalibration();
    $("calibrate-section").scrollIntoView({ behavior: "smooth" });
  };
  img.src = `/files/${job.id}/frame.jpg`;
}

// ---------- calibration ----------
const canvas = $("frame-canvas");
const ctx = canvas.getContext("2d");

function drawFrame() {
  const img = state.image;
  if (!img) return;
  const width = canvas.clientWidth || 800;
  canvas.width = width * devicePixelRatio;
  canvas.height = (width * img.naturalHeight / img.naturalWidth) * devicePixelRatio;
  const s = canvas.width / img.naturalWidth;
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  const r = 7 * devicePixelRatio;
  ctx.font = `bold ${12 * devicePixelRatio}px Heebo, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (const [name, [x, y]] of Object.entries(state.points)) {
    const isRim = name === "rim";
    ctx.beginPath();
    ctx.arc(x * s, y * s, isRim ? r * 1.4 : r, 0, 2 * Math.PI);
    ctx.fillStyle = name === state.current ? "#e8590c" : isRim ? "#fd7e14" : "#2b8a3e";
    ctx.fill();
    ctx.lineWidth = 2 * devicePixelRatio;
    ctx.strokeStyle = "#fff";
    ctx.stroke();
    ctx.fillStyle = "#fff";
    ctx.fillText(isRim ? "ט" : String(ORDER.indexOf(name) + 1), x * s, y * s);
  }
  drawLoupe(s);
}

// A magnifier in the corner so points can be placed precisely.
function drawLoupe(s) {
  if (!state.mouse) return;
  const [mx, my] = state.mouse; // video pixels
  const size = 150 * devicePixelRatio, zoom = 3;
  const src = size / zoom / s;
  const left = state.mouse[0] * s > canvas.width / 2;
  const ox = left ? 10 : canvas.width - size - 10, oy = 10;
  ctx.save();
  ctx.beginPath();
  ctx.rect(ox, oy, size, size);
  ctx.clip();
  ctx.drawImage(state.image, mx - src / 2, my - src / 2, src, src, ox, oy, size, size);
  ctx.restore();
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 2;
  ctx.strokeRect(ox, oy, size, size);
  ctx.strokeStyle = "#e8590c";
  ctx.beginPath();
  ctx.moveTo(ox + size / 2, oy + 8); ctx.lineTo(ox + size / 2, oy + size - 8);
  ctx.moveTo(ox + 8, oy + size / 2); ctx.lineTo(ox + size - 8, oy + size / 2);
  ctx.stroke();
}

function eventToImage(e) {
  const rect = canvas.getBoundingClientRect();
  const s = state.image.naturalWidth / rect.width;
  return [(e.clientX - rect.left) * s, (e.clientY - rect.top) * s];
}
canvas.addEventListener("mousemove", (e) => { state.mouse = eventToImage(e); drawFrame(); });
canvas.addEventListener("mouseleave", () => { state.mouse = null; drawFrame(); });
canvas.addEventListener("click", (e) => {
  const [x, y] = eventToImage(e);
  state.points[state.current] = [Math.round(x * 10) / 10, Math.round(y * 10) / 10];
  const next = ORDER.find((k) => !(k in state.points));
  if (next) state.current = next;
  refreshCalibration();
});

function selectLandmark(name) { state.current = name; refreshCalibration(); }

function renderList() {
  const list = $("landmark-list");
  list.innerHTML = "";
  ORDER.forEach((name, i) => {
    const li = document.createElement("li");
    li.className = (name in state.points ? "set " : "") + (name === state.current ? "current" : "");
    li.innerHTML = `<span class="num">${name === "rim" ? "ט" : i + 1}</span><span>${LABELS[name]}</span>`;
    if (name in state.points) {
      const rm = document.createElement("button");
      rm.className = "remove";
      rm.title = "מחק";
      rm.textContent = "✕";
      rm.onclick = (ev) => { ev.stopPropagation(); delete state.points[name]; state.current = name; refreshCalibration(); };
      li.appendChild(rm);
    }
    li.onclick = () => selectLandmark(name);
    list.appendChild(li);
  });
}

function svg(tag, attrs) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function renderMiniCourt() {
  const g = state.geometry;
  const root = $("mini-court");
  root.innerHTML = "";
  const [bx, by] = g.basket;
  const arc = (r, a0, a1, cx, cy) => {
    const p = (a) => [cx + r * Math.cos(a), cy + r * Math.sin(a)];
    const [x0, y0] = p(a0), [x1, y1] = p(a1);
    return `M ${x0} ${y0} A ${r} ${r} 0 0 1 ${x1} ${y1}`;
  };
  const start = Math.atan2(g.corner_three_y - by, g.width - g.corner_three_x - bx);
  const lines = [
    `M 0 0 H ${g.width} V ${g.half_length} H 0 Z`,
    `M ${bx - g.lane_half_width} 0 V ${g.lane_length} H ${bx + g.lane_half_width} V 0`,
    arc(g.circle_radius, 0, Math.PI, bx, g.lane_length),
    arc(g.circle_radius, Math.PI, 2 * Math.PI, bx, g.half_length),
    `M ${g.width - g.corner_three_x} 0 V ${g.corner_three_y} ` + arc(g.three_radius, start, Math.PI - start, bx, by).replace(/^M [^A]+/, "") + ` V 0`,
  ];
  lines.forEach((d) => root.appendChild(svg("path", { d, class: "court-line" })));
  root.appendChild(svg("circle", { cx: bx, cy: by, r: 0.23, class: "court-line" }));
  const marks = { ...state.landmarks, rim: g.basket };
  ORDER.forEach((name, i) => {
    const [x, y] = marks[name];
    const grp = svg("g", { class: `lm ${name in state.points ? "set" : ""} ${name === state.current ? "current" : ""}` });
    grp.appendChild(svg("circle", { cx: x, cy: y, r: name === "rim" ? 0.42 : 0.36 }));
    const t = svg("text", { x, y });
    t.textContent = name === "rim" ? "ט" : String(i + 1);
    grp.appendChild(t);
    const title = svg("title", {});
    title.textContent = LABELS[name];
    grp.appendChild(title);
    grp.addEventListener("click", () => selectLandmark(name));
    root.appendChild(grp);
  });
}

function refreshCalibration() {
  const n = Object.keys(floorPoints()).length;
  const hasRim = "rim" in state.points;
  $("current-label").textContent = `עכשיו מסמנים: ${LABELS[state.current]}`;
  $("calib-status").textContent =
    `סומנו ${n} נקודות רצפה${n < 4 ? ` (חסרות עוד ${4 - n})` : ""}` + (hasRim ? " + הטבעת" : " · הטבעת עוד לא סומנה");
  $("analyze-btn").disabled = n < 4;
  $("toggle-preview").disabled = n < 4;
  if (n < 4 && state.showPreview) togglePreview(false);
  renderList();
  if (state.geometry) renderMiniCourt();
  drawFrame();
  if (state.showPreview) loadPreview();
}

function calibrationBody() {
  return { points: floorPoints(), rim: state.points.rim || null, rim_radius_px: Number($("rim-radius").value) || 30 };
}

async function loadPreview() {
  try {
    const data = await postJSON(`/api/jobs/${state.job.id}/preview`, calibrationBody());
    $("preview-img").src = `${data.image}?t=${Date.now()}`;
    setError("calib-error", data.error_m > 0.3
      ? `שימו לב: הנקודות לא מתיישבות זו עם זו (סטייה ממוצעת ${data.error_m.toFixed(2)} מ'). בדקו שכל נקודה סומנה במקום הנכון.`
      : "");
  } catch (err) { setError("calib-error", err.message); }
}

function togglePreview(on = !state.showPreview) {
  state.showPreview = on;
  show("preview-img", on);
  show("frame-canvas", !on);
  $("toggle-preview").textContent = on ? "חזרה לסימון" : "הצג קווי מגרש";
  if (on) loadPreview();
  else drawFrame();
}

$("toggle-preview").onclick = () => togglePreview();
$("clear-calib").onclick = () => { state.points = {}; state.current = ORDER[0]; refreshCalibration(); };
$("load-calib").onclick = () => {
  const saved = storageGet();
  if (!saved) return;
  state.points = { ...saved.points };
  if (saved.rim) state.points.rim = saved.rim;
  $("rim-radius").value = saved.rim_radius_px || 30;
  refreshCalibration();
};
window.addEventListener("resize", drawFrame);

// ---------- analysis ----------
async function analyze(events) {
  setError("calib-error", "");
  const body = { ...calibrationBody(), model: $("model").value, stride: Number($("stride").value), events: events || null };
  try {
    await postJSON(`/api/jobs/${state.job.id}/analyze`, body);
  } catch (err) {
    if (!events) return setError("calib-error", err.message);
    show("progress-section");
    $("progress-title").textContent = "לא ניתן להתחיל";
    return setError("progress-error", err.message);
  }
  storageSet({ width: state.job.width, height: state.job.height, ...calibrationBody() });
  watch();
}
$("analyze-btn").onclick = () => analyze(null);

$("demo-btn").onclick = async () => {
  const job = await api("/api/demo", { method: "POST" });
  state.job = job;
  state.events = [];
  show("upload-section", false);
  watch();
};

async function watch() {
  show("calibrate-section", false);
  show("results-section", false);
  show("progress-section");
  setError("progress-error", "");
  $("progress-title").textContent = "מנתח…";
  for (;;) {
    let job;
    try { job = await api(`/api/jobs/${state.job.id}`); }
    catch (err) { return setError("progress-error", err.message); }
    $("progress-fill").style.width = `${Math.round(job.progress * 100)}%`;
    $("progress-stage").textContent = job.stage ? `${job.stage} – ${Math.round(job.progress * 100)}%` : "";
    if (job.state === "done") { state.job = job; state.lastDone = job; return showResults(job); }
    if (job.state === "error") {
      if (state.lastDone && state.lastDone.id === job.id) showResults(state.lastDone); // keep the earlier results
      else if (job.has_video) { show("calibrate-section"); setStep("calibrate"); }
      show("progress-section");
      $("progress-title").textContent = "הניתוח נכשל";
      setError("progress-error", job.error);
      return;
    }
    await new Promise((r) => setTimeout(r, 800));
  }
}

// ---------- results ----------
function showResults(job) {
  const r = job.results;
  show("progress-section", false);
  show("results-section");
  setStep("results");
  const decisions = r.decisions;
  const good = decisions.filter((d) => d.verdict === "good").length;
  const lost = decisions.reduce((sum, d) => sum + (d.verdict !== "good" && d.actual_points != null ? d.best_points - d.actual_points : 0), 0);
  $("stats").innerHTML = [
    [r.players.length, "שחקנים על המפה"],
    [decisions.length, "החלטות שנותחו"],
    [`${good}/${decisions.length}`, "החלטות טובות"],
    [lost.toFixed(2), "נקודות צפויות שהלכו לאיבוד"],
  ].map(([v, l]) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`).join("");
  show("no-events", decisions.length === 0);

  const base = `/files/${job.id}/output`;
  $("court-video").src = r.video ? `${base}/${r.video}?t=${Date.now()}` : "";
  show("court-video", !!r.video);

  const list = $("decisions");
  list.innerHTML = "";
  decisions.forEach((d) => {
    const [label, cls] = VERDICTS[d.verdict];
    const kind = d.event_kind === "shot" ? "זריקה" : "מסירה";
    const card = document.createElement("article");
    card.className = `decision ${cls}`;
    card.innerHTML = `
      <img src="${base}/${d.image}?t=${Date.now()}" alt="מפת המגרש ברגע ה${kind}">
      <div>
        <h4>${kind} של #${d.player} · ${d.time.toFixed(1)} שנ' <span class="badge">${label}</span></h4>
        <pre></pre>
      </div>`;
    card.querySelector("pre").textContent = d.text;
    list.appendChild(card);
  });

  $("downloads").innerHTML = ["report.md", "report.json", "tracks.csv"]
    .map((f) => `<a href="${base}/${f}" download>${f}</a>`).join("");

  show("events-editor", job.has_video);
  if (job.has_video) {
    if (!state.events.length) state.events = r.events.map((e) => ({ time: +e.time.toFixed(2), type: e.kind, player: e.player, target: e.target }));
    const v = $("source-video");
    show("source-video-error", false);
    v.onerror = () => show("source-video-error");
    v.src = r.source_video ? `${base}/${r.source_video}?t=${Date.now()}` : job.video_url;
    renderEvents();
  }
}

// ---------- manual events ----------
function renderEvents() {
  const body = $("events-body");
  body.innerHTML = "";
  state.events.sort((a, b) => a.time - b.time);
  state.events.forEach((ev, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><input type="number" step="0.1" min="0" value="${ev.time}"></td>
      <td><select><option value="shot">זריקה</option><option value="pass">מסירה</option></select></td>
      <td><input type="number" min="0" placeholder="אוטומטי" value="${ev.player ?? ""}"></td>
      <td><input type="number" min="0" placeholder="—" value="${ev.target ?? ""}"></td>
      <td><button class="ghost" title="מחק">✕</button></td>`;
    const [time, type, player, target] = tr.querySelectorAll("input, select");
    type.value = ev.type;
    target.disabled = ev.type !== "pass";
    time.onchange = () => { ev.time = Number(time.value); };
    type.onchange = () => { ev.type = type.value; if (ev.type !== "pass") ev.target = null; renderEvents(); };
    player.onchange = () => { ev.player = player.value === "" ? null : Number(player.value); };
    target.onchange = () => { ev.target = target.value === "" ? null : Number(target.value); };
    tr.querySelector("button").onclick = () => { state.events.splice(i, 1); renderEvents(); };
    body.appendChild(tr);
  });
  $("rerun").disabled = state.events.length === 0;
}
function addEvent(type) {
  const t = $("source-video").currentTime || 0;
  state.events.push({ time: Math.round(t * 100) / 100, type, player: null, target: null });
  renderEvents();
}
$("add-shot").onclick = () => addEvent("shot");
$("add-pass").onclick = () => addEvent("pass");
$("rerun").onclick = () => {
  const events = state.events.map((e) => {
    const out = { time: e.time, type: e.type };
    if (e.player != null) out.player = e.player;
    if (e.type === "pass" && e.target != null) out.target = e.target;
    return out;
  });
  analyze(events);
};
$("recalibrate").onclick = () => {
  show("results-section", false);
  show("calibrate-section");
  setStep("calibrate");
  drawFrame();
};
$("new-video").onclick = () => {
  state.job = null;
  show("results-section", false);
  show("upload-section");
  setStep("upload");
  $("file").value = "";
};

// ---------- upload wiring ----------
const drop = $("drop");
$("file").addEventListener("change", (e) => e.target.files[0] && upload(e.target.files[0]));
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && upload(e.dataTransfer.files[0]));

api("/api/landmarks").then((data) => {
  state.landmarks = data.landmarks;
  state.geometry = data.court;
  renderMiniCourt();
});
