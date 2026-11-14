// Pipeline sama dengan cry_utils.load_wave: mono, 8 kHz, maks 20 dtk, diulang sampai >= 5 dtk.
// Normalisasi puncak, STFT, mel, log dan CNN ada di dalam model.onnx.
const SR = 8000, CLIP = 40000, MAX_SEC = 20, LOW_CONF = 0.45;

// urutan WAJIB sama dengan CLASSES di cry_utils.py
const CLASSES = [
  { key: "hungry", name: "Lapar", cry: true,
    desc: "Tangisan berirama dan berulang, sering diawali rengekan lalu makin keras. Bayi biasanya juga mengisap tangan, menoleh mencari puting, atau mengecap bibir.",
    tips: ["Tawarkan ASI atau susu, meski jadwal menyusu belum tiba.", "Perhatikan tanda lapar awal (mengisap tangan, menoleh) agar tidak menunggu sampai menangis keras.", "Catat jarak antarmenyusu untuk melihat polanya."] },
  { key: "belly_pain", name: "Sakit perut", cry: true,
    desc: "Tangisan melengking dan datang tiba-tiba, wajah memerah, kaki sering ditekuk ke arah perut, sulit ditenangkan.",
    tips: ["Gendong tegak atau telungkupkan bayi di lengan Anda sambil diayun pelan.", "Pijat perut searah jarum jam atau gerakkan kakinya seperti mengayuh sepeda.", "Hubungi dokter bila disertai demam, muntah, perut keras, atau tangis tak berhenti berjam-jam."] },
  { key: "burping", name: "Ingin sendawa", cry: true,
    desc: "Rengekan pendek dan gelisah, biasanya saat atau sesudah menyusu karena ada udara yang tertelan.",
    tips: ["Sandarkan bayi tegak di bahu, lalu tepuk atau usap punggungnya pelan.", "Beri jeda untuk sendawa di tengah sesi menyusu.", "Periksa pelekatan menyusu atau ukuran dot agar udara yang tertelan berkurang."] },
  { key: "cold_hot", name: "Kedinginan atau kepanasan", cry: true,
    desc: "Rengekan yang menetap disertai tubuh gelisah. Bayi belum bisa mengatur suhu tubuhnya sebaik orang dewasa.",
    tips: ["Raba tengkuk atau dada bayi, bukan tangan atau kakinya.", "Tambah atau kurangi selapis pakaian; berkeringat berarti terlalu hangat.", "Jaga ruangan sejuk, berventilasi, dan jauh dari embusan AC atau kipas langsung."] },
  { key: "discomfort", name: "Tidak nyaman", cry: true,
    desc: "Tangisan rewel yang naik-turun, sering karena popok basah, pakaian ketat atau gatal, atau posisi yang kurang enak.",
    tips: ["Periksa dan ganti popok.", "Longgarkan pakaian, periksa label baju atau karet yang menekan kulit.", "Ubah posisi gendong atau posisi berbaring."] },
  { key: "tired", name: "Lelah atau mengantuk", cry: true,
    desc: "Rengekan yang makin lama makin kuat, disertai menguap, menggosok mata, atau memalingkan muka dari keramaian.",
    tips: ["Redupkan lampu dan kurangi suara di sekitar bayi.", "Bedong, ayun pelan, atau perdengarkan suara desis (white noise).", "Jaga rutinitas tidur yang sama setiap hari."] },
  { key: "laugh", name: "Tertawa", cry: false,
    desc: "Bukan tangisan. Suara ini terdengar seperti tawa atau celoteh senang.",
    tips: ["Si kecil sedang senang. Ajak bermain atau bicara untuk merespons tawanya."] },
  { key: "noise", name: "Suara bising", cry: false,
    desc: "Suara yang terdengar lebih mirip bising lingkungan daripada tangisan bayi.",
    tips: ["Rekam lebih dekat ke bayi, sekitar 30 sampai 50 cm.", "Matikan TV, kipas, atau sumber bising lain lalu rekam ulang."] },
  { key: "silence", name: "Hening", cry: false,
    desc: "Tidak terdengar tangisan yang jelas pada rekaman ini.",
    tips: ["Rekam saat bayi sedang menangis.", "Pastikan izin mikrofon aktif dan ponsel tidak tertutup tangan atau kain."] },
];

const $ = (id) => document.getElementById(id);
const status = $("status");
const setStatus = (text, cls) => { status.className = "status " + (cls || ""); status.querySelector("b").textContent = text; };

// ---------- model ----------
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
let session;
const ready = (location.protocol === "file:"
  ? Promise.reject(new Error("Buka halaman ini lewat GitHub Pages atau server lokal (python -m http.server), bukan langsung dari file."))
  : ort.InferenceSession.create("model.onnx"))
  .then((s) => { session = s; setStatus("Model siap", "ready"); })
  .catch((e) => { setStatus("Model gagal dimuat", "fail"); showError(e.message || String(e)); throw e; });

async function toModelInput(arrayBuffer) {
  // decode langsung ke 8 kHz: file 8 kHz tidak di-resample sama sekali, file lain cukup sekali
  let buf;
  try { buf = await new OfflineAudioContext(1, 1, SR).decodeAudioData(arrayBuffer); }
  catch { throw new Error("Format audio tidak dikenali browser. Coba WAV atau MP3."); }
  if (!buf.length) throw new Error("Rekaman kosong. Coba rekam lagi lebih lama.");
  const n = Math.min(buf.length, MAX_SEC * SR), x = new Float32Array(n);
  for (let c = 0; c < buf.numberOfChannels; c++) { // mono = rata-rata kanal, sama seperti x.mean(1)
    const ch = buf.getChannelData(c);
    for (let i = 0; i < n; i++) x[i] += ch[i] / buf.numberOfChannels;
  }
  if (x.length >= CLIP) return x;
  const y = new Float32Array(CLIP); // klip pendek diulang, sama seperti np.tile di Python
  for (let i = 0; i < CLIP; i++) y[i] = x[i % x.length];
  return y;
}

async function predict(arrayBuffer) {
  await ready;
  const x = await toModelInput(arrayBuffer);
  const out = await session.run({ audio: new ort.Tensor("float32", x, [1, x.length]) });
  const z = Array.from(out.logits.data), m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b);
  return e.map((v) => v / s);
}

// ---------- tampilan hasil ----------
const pct = (p) => (p * 100).toLocaleString("id-ID", { maximumFractionDigits: 1 }) + "%";

function showError(msg) { const el = $("error"); el.textContent = msg; el.hidden = !msg; }
function busy(on) {
  $("empty").hidden = !on && !$("result").hidden;
  $("empty").classList.toggle("busy", on);
  $("empty").querySelector("p").textContent = on ? "Menganalisis suara" : "Hasil analisis akan muncul di sini.";
  if (on) $("result").hidden = true;
}

function render(p) {
  const order = p.map((v, i) => i).sort((a, b) => p[b] - p[a]);
  const top = CLASSES[order[0]], conf = p[order[0]];
  $("result").dataset.key = top.key;
  $("resKind").textContent = top.cry ? "Kemungkinan besar bayi sedang" : "Hasil";
  $("resLabel").textContent = top.name;
  $("confText").textContent = pct(conf) + " yakin";
  $("resDesc").textContent = top.desc;
  $("resTips").replaceChildren(...top.tips.map((t) => Object.assign(document.createElement("li"), { textContent: t })));
  let note = "";
  if (!top.cry) note = "Model tidak mendengar tangisan pada rekaman ini.";
  else if (conf < LOW_CONF) {
    const alt = CLASSES[order[1]];
    note = `Keyakinan rendah. Kemungkinan lain: ${alt.name.toLowerCase()} (${pct(p[order[1]])}). Coba rekam ulang lebih dekat dan lebih lama.`;
  }
  $("notice").textContent = note; $("notice").hidden = !note;
  $("bars").replaceChildren(...order.map((i) => {
    const li = document.createElement("li");
    li.innerHTML = `<span></span><span class="track"><i style="width:0"></i></span><span class="pct"></span>`;
    li.children[0].textContent = CLASSES[i].name; li.children[2].textContent = pct(p[i]);
    return li;
  }));
  $("empty").hidden = true; $("result").hidden = false;
  $("confFill").style.width = "0";
  requestAnimationFrame(() => requestAnimationFrame(() => {
    $("confFill").style.width = conf * 100 + "%";
    $("bars").querySelectorAll(".track i").forEach((el, k) => (el.style.width = p[order[k]] * 100 + "%"));
  }));
}

async function analyze(blob) {
  showError("");
  const player = $("player");
  URL.revokeObjectURL(player.src);
  player.src = URL.createObjectURL(blob); player.hidden = false;
  busy(true);
  try { render(await predict(await blob.arrayBuffer())); }
  catch (e) { busy(false); showError(e.message || String(e)); }
}

// ---------- tab ----------
function selectTab(which) {
  for (const t of ["rec", "file"]) {
    $("tab-" + t).setAttribute("aria-selected", t === which);
    $("pane-" + t).hidden = t !== which;
  }
}
$("tab-rec").onclick = () => selectTab("rec");
$("tab-file").onclick = () => selectTab("file");
document.querySelectorAll("[data-go]").forEach((a) => a.addEventListener("click", () => selectTab(a.dataset.go)));

// ---------- unggah ----------
$("fileIn").addEventListener("change", (e) => e.target.files[0] && analyze(e.target.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("over")));
drop.addEventListener("drop", (e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) analyze(f); });

// ---------- rekam ----------
const recBtn = $("recBtn"), meter = $("meter");
let rec = null;

async function startRec() {
  showError("");
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder)
    return showError("Browser ini tidak mendukung perekaman. Gunakan tab Unggah file, atau buka di Chrome, Edge, Firefox, atau Safari terbaru.");
  let stream;
  try {
    // peredam bising & AGC dimatikan agar suara tangisan tetap asli seperti data latih
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
  } catch (e) {
    return showError(e.name === "NotAllowedError"
      ? "Izin mikrofon ditolak. Aktifkan izin mikrofon untuk situs ini di pengaturan browser, atau gunakan tab Unggah file."
      : "Mikrofon tidak bisa dibuka: " + e.message);
  }
  const chunks = [], mr = new MediaRecorder(stream);
  const ctx = new AudioContext(), an = ctx.createAnalyser();
  an.fftSize = 2048; ctx.createMediaStreamSource(stream).connect(an);
  const t0 = performance.now();
  rec = { mr, stream, ctx, an, t0, raf: 0 };
  mr.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  mr.onstop = () => {
    stream.getTracks().forEach((t) => t.stop()); ctx.close(); cancelAnimationFrame(rec.raf); rec = null;
    recBtn.setAttribute("aria-pressed", "false"); recBtn.querySelector(".rec-label").textContent = "Rekam ulang";
    analyze(new Blob(chunks, { type: mr.mimeType }));
  };
  mr.start();
  recBtn.setAttribute("aria-pressed", "true"); recBtn.querySelector(".rec-label").textContent = "Berhenti & analisis";
  const data = new Float32Array(an.fftSize);
  const tick = () => {
    const sec = (performance.now() - t0) / 1000;
    $("timer").textContent = sec.toLocaleString("id-ID", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
    if (sec >= MAX_SEC) return stopRec();
    an.getFloatTimeDomainData(data); drawWave(meter, data, "#ffb38a", 3);
    rec.raf = requestAnimationFrame(tick);
  };
  tick();
}
function stopRec() { if (rec && rec.mr.state === "recording") rec.mr.stop(); }
recBtn.onclick = () => (rec ? stopRec() : startRec());

function drawWave(cv, data, color, gain = 1) {
  const w = (cv.width = cv.clientWidth * devicePixelRatio), h = (cv.height = cv.clientHeight * devicePixelRatio);
  const g = cv.getContext("2d"), step = Math.max(1, Math.floor(data.length / w));
  g.clearRect(0, 0, w, h); g.strokeStyle = color; g.lineWidth = 2 * devicePixelRatio; g.beginPath();
  for (let x = 0; x < w; x++) g.lineTo(x, h / 2 + Math.max(-1, Math.min(1, data[x * step] * gain)) * h * 0.45);
  g.stroke();
}
drawWave(meter, new Float32Array(2048), "rgba(244,239,233,.25)");

// ---------- dekorasi: gelombang hero, bento, reveal ----------
function sigPath(seed, n = 48) { // pola gelombang tetap per kelas (bukan data asli)
  let r = seed, d = "M0 17";
  for (let i = 1; i <= n; i++) { r = (r * 16807) % 2147483647; d += ` L${(i * 200) / n} ${17 + ((r / 2147483647) - 0.5) * 30 * Math.sin((i / n) * Math.PI)}`; }
  return d;
}
$("bento").replaceChildren(...CLASSES.map((c, i) => {
  const el = document.createElement("article");
  el.className = "card reveal" + (i === 0 ? " big" : "");
  el.innerHTML = `<svg class="sig" viewBox="0 0 200 34" preserveAspectRatio="none" aria-hidden="true"><path d="${sigPath(i + 7)}" fill="none" stroke="currentColor" stroke-width="1.6" vector-effect="non-scaling-stroke"/></svg><h3></h3><p></p>`;
  el.querySelector("h3").textContent = c.name; el.querySelector("p").textContent = c.desc;
  return el;
}));

const scrub = $("scrub");
scrub.innerHTML = scrub.textContent.split(" ").map((w) => `<span>${w}</span> `).join("");
const words = [...scrub.children];

const io = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && (e.target.classList.add("in"), io.unobserve(e.target))), { threshold: 0.15 });
document.querySelectorAll(".reveal").forEach((el) => io.observe(el));

const hero = $("heroWave"), wave = new Float32Array(1024);
let tHero = 0;
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
function heroLoop() {
  tHero += 0.012;
  for (let i = 0; i < wave.length; i++) {
    const u = i / wave.length;
    wave[i] = Math.sin(u * 18 + tHero * 3) * Math.sin(u * Math.PI) * (0.25 + 0.2 * Math.sin(u * 5 - tHero)) * 0.9;
  }
  drawWave(hero, wave, "rgba(169,155,255,.55)");
  if (!reduce) requestAnimationFrame(heroLoop);
}
heroLoop();

addEventListener("scroll", () => { // kata-kata menyala sesuai posisi scroll
  const r = scrub.getBoundingClientRect(), k = (innerHeight * 0.85 - r.top) / (r.height + innerHeight * 0.35);
  const on = Math.round(Math.min(1, Math.max(0, k)) * words.length);
  words.forEach((w, i) => w.classList.toggle("on", i < on));
}, { passive: true });
