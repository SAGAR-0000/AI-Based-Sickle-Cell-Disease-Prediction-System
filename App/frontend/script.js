/**
 * SickleVision — Frontend interaction logic.
 * Handles image upload (drag-and-drop + click), preview, API calls, and result rendering.
 */

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const API_BASE_URL = "http://localhost:8000";
const PREDICT_ENDPOINT = `${API_BASE_URL}/predict`;
const ALLOWED_TYPES = ["image/jpeg", "image/png"];

// ---------------------------------------------------------------------------
// DOM References
// ---------------------------------------------------------------------------
/** @returns {HTMLElement} */
function getElement(/** @type {string} */ id) {
  const el = document.getElementById(id);
  if (!el) {
    throw new Error(`Element with id "${id}" not found in DOM.`);
  }
  return el;
}

const dropzone = /** @type {HTMLDivElement} */ (getElement("dropzone"));
const dropzoneContent = /** @type {HTMLDivElement} */ (getElement("dropzone-content"));
const fileInput = /** @type {HTMLInputElement} */ (getElement("file-input"));
const previewImage = /** @type {HTMLImageElement} */ (getElement("preview-image"));
const predictBtn = /** @type {HTMLButtonElement} */ (getElement("predict-btn"));
const clearBtn = /** @type {HTMLButtonElement} */ (getElement("clear-btn"));
const btnSpinner = /** @type {HTMLSpanElement} */ (getElement("btn-spinner"));
const resultsPanel = /** @type {HTMLDivElement} */ (getElement("results-panel"));
const resultBadge = /** @type {HTMLSpanElement} */ (getElement("result-badge"));
const resultConfidence = /** @type {HTMLSpanElement} */ (getElement("result-confidence"));
const barNegative = /** @type {HTMLDivElement} */ (getElement("bar-negative"));
const barPositive = /** @type {HTMLDivElement} */ (getElement("bar-positive"));
const valNegative = /** @type {HTMLSpanElement} */ (getElement("val-negative"));
const valPositive = /** @type {HTMLSpanElement} */ (getElement("val-positive"));
const errorPanel = /** @type {HTMLDivElement} */ (getElement("error-panel"));
const errorText = /** @type {HTMLParagraphElement} */ (getElement("error-text"));

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
/** @type {File | null} */
let selectedFile = null;

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/**
 * Validates a file is an allowed image type.
 * @param {File} file
 * @returns {boolean}
 */
function isValidImage(file) {
  return ALLOWED_TYPES.includes(file.type);
}

// ---------------------------------------------------------------------------
// UI Helpers
// ---------------------------------------------------------------------------

/**
 * Shows the image preview and enables action buttons.
 * @param {File} file
 * @returns {void}
 */
function showPreview(file) {
  const objectUrl = URL.createObjectURL(file);
  previewImage.src = objectUrl;
  previewImage.classList.remove("hidden");
  dropzoneContent.classList.add("hidden");
  predictBtn.disabled = false;
  clearBtn.disabled = false;
  hideResults();
  hideError();
}

/**
 * Resets the upload UI back to initial state.
 * @returns {void}
 */
function resetUpload() {
  selectedFile = null;
  fileInput.value = "";
  previewImage.src = "";
  previewImage.classList.add("hidden");
  dropzoneContent.classList.remove("hidden");
  predictBtn.disabled = true;
  clearBtn.disabled = true;
  hideResults();
  hideError();
}

/**
 * Shows the results panel with prediction data.
 * @param {{ prediction: string, confidence: number, positive_prob: number, negative_prob: number }} data
 * @returns {void}
 */
function showResults(data) {
  resultsPanel.classList.remove("hidden");

  const isPositive = data.prediction === "Positive";
  resultBadge.textContent = data.prediction;
  resultBadge.className = `result-badge ${isPositive ? "badge-positive" : "badge-negative"}`;
  resultConfidence.textContent = `${(data.confidence * 100).toFixed(1)}% confidence`;

  // Animate bars via requestAnimationFrame for smooth transition
  requestAnimationFrame(() => {
    barNegative.style.width = `${(data.negative_prob * 100).toFixed(1)}%`;
    barPositive.style.width = `${(data.positive_prob * 100).toFixed(1)}%`;
  });

  valNegative.textContent = `${(data.negative_prob * 100).toFixed(1)}%`;
  valPositive.textContent = `${(data.positive_prob * 100).toFixed(1)}%`;
}

/** Hides the results panel. @returns {void} */
function hideResults() {
  resultsPanel.classList.add("hidden");
  barNegative.style.width = "0%";
  barPositive.style.width = "0%";
}

/**
 * Shows an error message.
 * @param {string} message
 * @returns {void}
 */
function showError(message) {
  errorPanel.classList.remove("hidden");
  errorText.textContent = message;
}

/** Hides the error panel. @returns {void} */
function hideError() {
  errorPanel.classList.add("hidden");
  errorText.textContent = "";
}

/**
 * Sets the predict button to a loading state.
 * @param {boolean} loading
 * @returns {void}
 */
function setLoading(loading) {
  const btnLabel = predictBtn.querySelector(".btn-label");
  if (loading) {
    predictBtn.disabled = true;
    clearBtn.disabled = true;
    if (btnLabel) btnLabel.textContent = "Analyzing…";
    btnSpinner.classList.remove("hidden");
  } else {
    predictBtn.disabled = false;
    clearBtn.disabled = false;
    if (btnLabel) btnLabel.textContent = "Predict";
    btnSpinner.classList.add("hidden");
  }
}

// ---------------------------------------------------------------------------
// File Selection
// ---------------------------------------------------------------------------

/**
 * Handles a selected file from input or drop.
 * @param {File} file
 * @returns {void}
 */
function handleFileSelect(file) {
  if (!isValidImage(file)) {
    showError("Please upload a valid image file (JPG or PNG).");
    return;
  }
  selectedFile = file;
  showPreview(file);
}

// ---------------------------------------------------------------------------
// API Call
// ---------------------------------------------------------------------------

/**
 * Sends the selected file to the prediction API and renders results.
 * @returns {Promise<void>}
 */
async function runPrediction() {
  if (!selectedFile) {
    showError("No image selected.");
    return;
  }

  setLoading(true);
  hideError();
  hideResults();

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch(PREDICT_ENDPOINT, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorBody = await response.json().catch(() => null);
      const detail = errorBody?.detail ?? `Server responded with status ${response.status}`;
      throw new Error(detail);
    }

    /** @type {{ prediction: string, confidence: number, positive_prob: number, negative_prob: number }} */
    const data = await response.json();
    showResults(data);
  } catch (/** @type {unknown} */ err) {
    const message = err instanceof Error ? err.message : "An unexpected error occurred.";
    showError(`Prediction failed: ${message}`);
  } finally {
    setLoading(false);
  }
}

// ---------------------------------------------------------------------------
// Event Listeners
// ---------------------------------------------------------------------------

// Click to upload
dropzone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) handleFileSelect(file);
});

// Drag & Drop
dropzone.addEventListener("dragover", (/** @type {DragEvent} */ e) => {
  e.preventDefault();
  dropzone.classList.add("drag-over");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("drag-over");
});

dropzone.addEventListener("drop", (/** @type {DragEvent} */ e) => {
  e.preventDefault();
  dropzone.classList.remove("drag-over");
  const file = e.dataTransfer?.files[0];
  if (file) handleFileSelect(file);
});

// Predict button
predictBtn.addEventListener("click", () => {
  runPrediction();
});

// Clear button
clearBtn.addEventListener("click", () => {
  resetUpload();
});
