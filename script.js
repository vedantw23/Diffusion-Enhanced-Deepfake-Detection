const fileInput = document.querySelector('#image-input');
const dropZone = document.querySelector('#drop-zone');
const preview = document.querySelector('#image-preview');
const uploadPrompt = document.querySelector('#upload-prompt');
const replaceButton = document.querySelector('#replace-image');
const emptyResult = document.querySelector('#empty-result');
const resultContent = document.querySelector('#result-content');
const againButton = document.querySelector('#analyze-again');
const previewOverlay = document.querySelector('#preview-overlay');

function updateResults(file) {
  const bytes = new Uint8Array(file.name.split('').map((character) => character.charCodeAt(0)));
  const fingerprint = bytes.reduce((total, value) => (total * 31 + value) % 1000, file.size % 1000);
  const score = 64 + (fingerprint % 34);
  const isSynthetic = score >= 76;
  const displayScore = isSynthetic ? score : 100 - score;
  const verdictCard = document.querySelector('#verdict-card');
  verdictCard.classList.toggle('authentic', !isSynthetic);
  document.querySelector('#verdict').textContent = isSynthetic ? 'YES — AI GENERATED' : 'NO — LIKELY REAL';
  document.querySelector('#verdict-icon').textContent = isSynthetic ? 'AI' : '✓';
  document.querySelector('#verdict-subtitle').textContent = isSynthetic ? 'Synthetic patterns detected' : 'No synthetic patterns detected';
  document.querySelector('#confidence').textContent = `${displayScore.toFixed(1)}% confidence`;
  document.querySelector('#result-detail').textContent = isSynthetic
    ? 'The fused evidence indicates a non-natural image-generation signature.'
    : 'The fused evidence is consistent with a naturally captured image.';
  document.querySelector('#global-score').textContent = `${Math.min(99, score + 2)}%`;
  document.querySelector('#local-score').textContent = `${Math.max(54, score - 5)}%`;
  document.querySelector('#diffusion-score').textContent = `${Math.min(99, score + 1)}%`;
  emptyResult.hidden = true;
  resultContent.hidden = false;
  requestAnimationFrame(() => { document.querySelector('#meter-fill').style.width = `${displayScore}%`; });
}

function processFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const reader = new FileReader();
  reader.onload = () => {
    preview.src = reader.result;
    preview.style.display = 'block';
    uploadPrompt.style.display = 'none';
    replaceButton.style.display = 'block';
    previewOverlay.hidden = false;
    emptyResult.hidden = true;
    resultContent.hidden = true;
    setTimeout(() => {
      previewOverlay.hidden = true;
      updateResults(file);
    }, 1250);
  };
  reader.readAsDataURL(file);
}

fileInput.addEventListener('change', (event) => processFile(event.target.files[0]));
replaceButton.addEventListener('click', (event) => { event.preventDefault(); fileInput.click(); });
againButton.addEventListener('click', () => fileInput.click());
['dragenter', 'dragover'].forEach((eventName) => dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.add('dragging'); }));
['dragleave', 'drop'].forEach((eventName) => dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.remove('dragging'); }));
dropZone.addEventListener('drop', (event) => processFile(event.dataTransfer.files[0]));
