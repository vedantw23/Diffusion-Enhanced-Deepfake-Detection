const fileInput = document.querySelector('#image-input');
const dropZone = document.querySelector('#drop-zone');
const preview = document.querySelector('#image-preview');
const videoPreview = document.querySelector('#video-preview');
const uploadPrompt = document.querySelector('#upload-prompt');
const replaceButton = document.querySelector('#replace-image');
const emptyResult = document.querySelector('#empty-result');
const resultContent = document.querySelector('#result-content');
const againButton = document.querySelector('#analyze-again');
const previewOverlay = document.querySelector('#preview-overlay');
const overlayStatus = document.querySelector('#overlay-status');
const backendUrl = window.DETECTOR_API_URL || '';
const MAX_UPLOAD_BYTES = 100 * 1024 * 1024;

function updateResults(result) {
  const isSynthetic = result.decision === 'likely_ai_generated';
  const isUncertain = result.decision === 'uncertain';
  const displayScore = result.confidence * 100;
  const verdictCard = document.querySelector('#verdict-card');
  verdictCard.classList.toggle('authentic', !isSynthetic);
  verdictCard.classList.toggle('uncertain', isUncertain);
  document.querySelector('#verdict').textContent = isSynthetic ? 'YES — LIKELY AI GENERATED' : isUncertain ? 'UNCERTAIN — REVIEW NEEDED' : 'NO — LIKELY REAL';
  document.querySelector('#verdict-icon').textContent = isSynthetic ? 'AI' : isUncertain ? '?' : '✓';
  document.querySelector('#verdict-subtitle').textContent = isSynthetic ? 'Model detected generation patterns' : isUncertain ? 'Confidence is below the decision threshold' : 'Model found stronger real-image evidence';
  document.querySelector('#verdict-question').textContent = result.media_type === 'video' ? 'IS THIS VIDEO AI GENERATED?' : 'IS THIS IMAGE AI GENERATED?';
  document.querySelector('#confidence').textContent = `${displayScore.toFixed(1)}% confidence`;
  document.querySelector('#result-detail').textContent = result.explanation;
  document.querySelector('#media-type').textContent = result.media_type.toUpperCase();
  document.querySelector('#frames-analyzed').textContent = result.frames_analyzed;
  document.querySelector('#model-status').textContent = 'TRAINED MODEL';
  document.querySelector('#result-label').textContent = 'MODEL RESULT';
  emptyResult.hidden = true;
  resultContent.hidden = false;
  requestAnimationFrame(() => { document.querySelector('#meter-fill').style.width = `${displayScore}%`; });
}

function showError(message, mediaType = 'image') {
  const verdictCard = document.querySelector('#verdict-card');
  verdictCard.classList.remove('authentic', 'uncertain');
  document.querySelector('#result-label').textContent = 'MODEL UNAVAILABLE';
  document.querySelector('#verdict').textContent = 'ANALYSIS NOT RUN';
  document.querySelector('#verdict-icon').textContent = '!';
  document.querySelector('#verdict-subtitle').textContent = 'No simulated result is shown';
  document.querySelector('#verdict-question').textContent = mediaType === 'video' ? 'IS THIS VIDEO AI GENERATED?' : 'IS THIS IMAGE AI GENERATED?';
  document.querySelector('#confidence').textContent = '—';
  document.querySelector('#meter-fill').style.width = '0%';
  document.querySelector('#result-detail').textContent = message;
  document.querySelector('#media-type').textContent = '—';
  document.querySelector('#frames-analyzed').textContent = '—';
  document.querySelector('#model-status').textContent = 'OFFLINE';
  emptyResult.hidden = true;
  resultContent.hidden = false;
}

async function requestPrediction(file) {
  const formData = new FormData();
  formData.append('file', file);
  let response;
  try {
    response = await fetch(`${backendUrl}/predict`, { method: 'POST', body: formData });
  } catch {
    throw new Error('The detection API cannot be reached. The free server may be waking up; wait one minute, refresh the page, and try again.');
  }
  const isJson = response.headers.get('content-type')?.includes('application/json');
  const data = isJson ? await response.json() : null;
  if (!response.ok) {
    if (data?.detail) throw new Error(data.detail);
    throw new Error('The detection API is not connected to this website. Deploy the FastAPI service with a trained best.pt checkpoint, then set its URL in api-config.js.');
  }
  if (!data) throw new Error('The detection API returned an invalid response.');
  return data;
}

function processFile(file) {
  const isVideo = file?.type.startsWith('video/');
  if (!file || (!file.type.startsWith('image/') && !isVideo)) return;
  if (file.size > MAX_UPLOAD_BYTES) {
    showError('This file is larger than the 100 MB limit. Choose a smaller image or video.', isVideo ? 'video' : 'image');
    return;
  }
  const objectUrl = URL.createObjectURL(file);
  if (isVideo) {
    preview.style.display = 'none';
    videoPreview.src = objectUrl;
    videoPreview.hidden = false;
    videoPreview.style.display = 'block';
  } else {
    videoPreview.pause();
    videoPreview.removeAttribute('src');
    videoPreview.load();
    videoPreview.hidden = true;
    videoPreview.style.display = 'none';
    preview.src = objectUrl;
    preview.style.display = 'block';
  }
  uploadPrompt.style.display = 'none';
  replaceButton.style.display = 'block';
  previewOverlay.hidden = false;
  overlayStatus.textContent = isVideo ? 'ANALYZING VIDEO FRAMES — PLEASE WAIT' : 'ANALYZING IMAGE — PLEASE WAIT';
  emptyResult.hidden = true;
  resultContent.hidden = true;
  requestPrediction(file)
    .then(updateResults)
    .catch((error) => showError(error.message || 'Unable to contact the trained detection model.', isVideo ? 'video' : 'image'))
    .finally(() => { previewOverlay.hidden = true; });
}

fileInput.addEventListener('change', (event) => processFile(event.target.files[0]));
replaceButton.addEventListener('click', (event) => { event.preventDefault(); fileInput.click(); });
againButton.addEventListener('click', () => fileInput.click());
['dragenter', 'dragover'].forEach((eventName) => dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.add('dragging'); }));
['dragleave', 'drop'].forEach((eventName) => dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.remove('dragging'); }));
dropZone.addEventListener('drop', (event) => processFile(event.dataTransfer.files[0]));
