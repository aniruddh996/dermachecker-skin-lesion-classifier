// ---- Hero ambient background: sparse drifting dot-grid (echoes the logo mark) ----
(function heroCanvas() {
  const canvas = document.getElementById('hero-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let W, H, nodes = [];
  const COUNT = 46;

  function resize() {
    const rect = canvas.parentElement.getBoundingClientRect();
    W = canvas.width = rect.width * devicePixelRatio;
    H = canvas.height = rect.height * devicePixelRatio;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
  }
  function init() {
    nodes = [];
    for (let i = 0; i < COUNT; i++) {
      nodes.push({
        x: Math.random() * W,
        y: Math.random() * H,
        vx: (Math.random() - 0.5) * 0.12 * devicePixelRatio,
        vy: (Math.random() - 0.5) * 0.12 * devicePixelRatio,
        r: (Math.random() * 1.8 + 1) * devicePixelRatio,
        hue: Math.random() < 0.5 ? '#6D3FE0' : '#E23F87',
      });
    }
  }
  function draw() {
    ctx.clearRect(0, 0, W, H);
    const maxDist = 120 * devicePixelRatio;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < maxDist) {
          ctx.strokeStyle = 'rgba(109,63,224,' + (0.12 * (1 - dist / maxDist)) + ')';
          ctx.lineWidth = devicePixelRatio;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }
    for (const n of nodes) {
      ctx.beginPath();
      ctx.fillStyle = n.hue + '55';
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  function step() {
    for (const n of nodes) {
      n.x += n.vx; n.y += n.vy;
      if (n.x < 0 || n.x > W) n.vx *= -1;
      if (n.y < 0 || n.y > H) n.vy *= -1;
    }
    draw();
    if (!reduce) requestAnimationFrame(step);
  }
  try {
    resize(); init(); draw();
    if (!reduce) requestAnimationFrame(step);
    window.addEventListener('resize', () => { resize(); init(); draw(); });
  } catch (e) { /* non-critical */ }
})();

// ---- Test & Result panel ----
(function testPanel() {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const previewWrap = document.getElementById('preview-wrap');
  const previewImg = document.getElementById('preview-img');
  const clearBtn = document.getElementById('preview-clear');
  const analyzeBtn = document.getElementById('analyze-btn');
  const resultCard = document.getElementById('result-card');
  const resultPlaceholder = document.getElementById('result-placeholder');
  const resultLabel = document.getElementById('result-label');
  const resultConf = document.getElementById('result-conf');
  const barRow = document.getElementById('bar-row');
  const apiWarning = document.getElementById('api-warning');

  if (!dropzone) return;

  let currentFile = null;

  function showPreview(file) {
    currentFile = file;
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    dropzone.style.display = 'none';
    previewWrap.style.display = 'block';
    analyzeBtn.disabled = false;
    resultCard.classList.remove('show');
    resultPlaceholder.style.display = 'block';
    resultPlaceholder.textContent = 'Ready — click Analyze to classify this image.';
  }

  function resetPanel() {
    currentFile = null;
    dropzone.style.display = 'flex';
    previewWrap.style.display = 'none';
    analyzeBtn.disabled = true;
    resultCard.classList.remove('show');
    resultPlaceholder.style.display = 'block';
    resultPlaceholder.textContent = 'Upload a photo to see a prediction here.';
  }

  dropzone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) showPreview(e.target.files[0]);
  });
  ['dragenter', 'dragover'].forEach(evt => {
    dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add('drag'); });
  });
  ['dragleave', 'drop'].forEach(evt => {
    dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove('drag'); });
  });
  dropzone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) showPreview(file);
  });
  clearBtn.addEventListener('click', (e) => { e.stopPropagation(); resetPanel(); });

  analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<span class="spinner"></span>Analyzing…';
    apiWarning.classList.remove('show');

    const form = new FormData();
    form.append('file', currentFile);

    try {
      const res = await fetch('/predict', { method: 'POST', body: form });
      if (!res.ok) throw new Error('API error ' + res.status);
      const data = await res.json();
      renderResult(data);
    } catch (err) {
      apiWarning.classList.add('show');
      apiWarning.textContent = "Couldn't reach the model API (" + err.message + "). Make sure server.py is running.";
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = 'Analyze another';
    }
  });

  function renderResult(data) {
    resultPlaceholder.style.display = 'none';
    resultCard.classList.add('show');
    resultLabel.textContent = data.label;
    resultConf.textContent = Math.round(data.confidence * 100) + '% confidence';
    barRow.innerHTML = '';
    data.top3.forEach(item => {
      const wrap = document.createElement('div');
      wrap.className = 'bar-item';
      const pct = Math.round(item.probability * 100);
      wrap.innerHTML = `
        <div class="bar-label"><span>${item.label}</span><span>${pct}%</span></div>
        <div class="bar-track"><div class="bar-fill" style="width:0%"></div></div>
      `;
      barRow.appendChild(wrap);
      requestAnimationFrame(() => {
        wrap.querySelector('.bar-fill').style.width = pct + '%';
      });
    });
  }
})();
