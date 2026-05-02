const API = window.location.origin;

// ── DOM refs ──
const dropZone       = document.getElementById('drop-zone');
const fileInput      = document.getElementById('file-input');
const previewSection = document.getElementById('preview-section');
const previewImage   = document.getElementById('preview-image');
const fileName       = document.getElementById('file-name');
const uploadSection  = document.getElementById('upload-section');
const analyzeBtn     = document.getElementById('analyze-btn');
const changeBtn      = document.getElementById('change-btn');
const loadingSection = document.getElementById('loading-section');
const resultsSection = document.getElementById('results-section');
const errorSection   = document.getElementById('error-section');
const errorMessage   = document.getElementById('error-message');
const retryBtn       = document.getElementById('retry-btn');
const newAnalysisBtn = document.getElementById('new-analysis-btn');

let currentFile = null;

// ── Helpers ──
function showOnly(section) {
  [uploadSection, previewSection, loadingSection, resultsSection, errorSection]
    .forEach(s => s.classList.add('hidden'));
  section.classList.remove('hidden');
}

function formatClassName(name) {
  return name.replace(/_+/g, ' ').replace(/\s+/g, ' ').trim();
}

// ── Drag & Drop ──
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// ── File handling ──
function handleFile(file) {
  const allowed = ['.jpg','.jpeg','.png','.bmp','.webp'];
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (!allowed.includes(ext)) {
    showError('Invalid file type. Please upload JPG, PNG, BMP, or WebP.');
    return;
  }
  currentFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImage.src = e.target.result;
    fileName.textContent = file.name;
    showOnly(previewSection);
  };
  reader.readAsDataURL(file);
}

// ── Buttons ──
changeBtn.addEventListener('click', () => {
  currentFile = null;
  fileInput.value = '';
  showOnly(uploadSection);
});

retryBtn.addEventListener('click', () => {
  if (currentFile) {
    showOnly(previewSection);
  } else {
    showOnly(uploadSection);
  }
});

newAnalysisBtn.addEventListener('click', () => {
  currentFile = null;
  fileInput.value = '';
  showOnly(uploadSection);
});

// ── Analysis ──
analyzeBtn.addEventListener('click', async () => {
  if (!currentFile) return;
  showOnly(loadingSection);

  const formData = new FormData();
  formData.append('file', currentFile);

  try {
    // Run both requests in parallel
    const [predictRes, explainRes] = await Promise.all([
      fetch(`${API}/predict`, { method: 'POST', body: formData }),
      (() => {
        const fd2 = new FormData();
        fd2.append('file', currentFile);
        return fetch(`${API}/explain`, { method: 'POST', body: fd2 });
      })()
    ]);

    if (!predictRes.ok) throw new Error(`Prediction failed (${predictRes.status})`);
    if (!explainRes.ok) throw new Error(`Explanation failed (${explainRes.status})`);

    const predictData = await predictRes.json();
    const explainData = await explainRes.json();

    renderResults(predictData, explainData);
  } catch (err) {
    showError(err.message || 'Something went wrong. Please try again.');
  }
});

// ── Error display ──
function showError(msg) {
  errorMessage.textContent = msg;
  showOnly(errorSection);
}

// ── Render results ──
function renderResults(predictData, explainData) {
  // Original image
  const resultOriginal = document.getElementById('result-original');
  resultOriginal.src = previewImage.src;

  // GradCAM image
  const resultGradcam = document.getElementById('result-gradcam');
  resultGradcam.src = `data:image/png;base64,${explainData.gradcam_image}`;

  // Diagnosis
  const diagName = document.getElementById('diagnosis-name');
  const diagConf = document.getElementById('diagnosis-confidence');
  diagName.textContent = formatClassName(explainData.predicted_class);
  diagConf.textContent = `${(explainData.confidence * 100).toFixed(1)}%`;

  // Predictions list
  const listEl = document.getElementById('predictions-list');
  listEl.innerHTML = '';

  const predictions = predictData.predictions || [];
  predictions.forEach((p, i) => {
    const confNum = parseFloat(p.confidence);
    const item = document.createElement('div');
    item.className = 'pred-item';
    item.innerHTML = `
      <div class="pred-rank ${i === 0 ? 'top' : ''}">${p.rank}</div>
      <div class="pred-info">
        <div class="pred-name">${formatClassName(p.class)}</div>
        <div class="pred-bar-track"><div class="pred-bar-fill" data-width="${confNum}"></div></div>
      </div>
      <div class="pred-conf">${p.confidence}</div>
    `;
    listEl.appendChild(item);
  });

  showOnly(resultsSection);

  // Animate bars after render
  requestAnimationFrame(() => {
    setTimeout(() => {
      document.querySelectorAll('.pred-bar-fill').forEach(bar => {
        bar.style.width = bar.dataset.width + '%';
      });
    }, 100);
  });
}
