const REVIEW_BATCH_SIZE = 5;

const fallbackCases = [
  {
    image: "ISIC_0015219_downsampled",
    diagnosis: "MEL",
    age_group: "missing",
    sex_clean: "male",
    review_reason: "missing age or sex metadata",
  },
  {
    image: "ISIC_0030858",
    diagnosis: "NV",
    age_group: "<20",
    sex_clean: "female",
    review_reason: "underrepresented age x sex group",
  },
  {
    image: "ISIC_0030680",
    diagnosis: "SCC",
    age_group: "80+",
    sex_clean: "female",
    review_reason: "rare diagnosis within age x sex group",
  },
];

let queue = [];
let selectedIndex = 0;
let completedInitialReviews = 0;
let pendingDisagreement = null;
let rereviewQueue = [];
let rereviewIndex = 0;
let trainingPoll = null;

const reviewWorkspace = document.querySelector("#review-workspace");
const startReviewButton = document.querySelector("#start-review");
const queueList = document.querySelector("#queue-list");
const queueCount = document.querySelector("#queue-count");
const reviewProgress = document.querySelector("#review-progress");
const caseImage = document.querySelector("#case-image");
const caseId = document.querySelector("#case-id");
const caseReason = document.querySelector("#case-reason");
const hiddenLabelText = document.querySelector("#hidden-label-text");
const ageGroupText = document.querySelector("#age-group-text");
const sexText = document.querySelector("#sex-text");
const ageSexText = document.querySelector("#age-sex-text");
const notes = document.querySelector("#review-notes");
const reviewStatus = document.querySelector("#review-status");
const challengePanel = document.querySelector("#challenge-panel");
const challengeCopy = document.querySelector("#challenge-copy");
const challengeNotes = document.querySelector("#challenge-notes");
const serverNote = document.querySelector("#server-note");
const trainingSection = document.querySelector("#training");
const trainingStatus = document.querySelector("#training-status");
const trainingMetrics = document.querySelector("#training-metrics");
const trainingProgress = document.querySelector("#training-progress");
const startTrainingButton = document.querySelector("#start-training");
const interruptTrainingButton = document.querySelector("#interrupt-training");
const rereviewTable = document.querySelector("#rereview-table");
const rereviewCard = document.querySelector("#rereview-card");
const rereviewImage = document.querySelector("#rereview-image");
const rereviewId = document.querySelector("#rereview-id");
const rereviewReason = document.querySelector("#rereview-reason");
const rereviewContext = document.querySelector("#rereview-context");
const rereviewNotes = document.querySelector("#rereview-notes");
const rereviewStatus = document.querySelector("#rereview-status");
const completeSection = document.querySelector("#complete");

function imagePath(imageId) {
  return `ISIC_2019_Training_Input/${imageId}.jpg`;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

function setPhase(activeId) {
  document.querySelectorAll(".phase").forEach((phase) => phase.classList.remove("active"));
  document.querySelector(activeId)?.classList.add("active");
}

function checkedRadioValue(name) {
  return document.querySelector(`input[name="${name}"]:checked`)?.value || "";
}

function clearRadioValue(name) {
  document.querySelectorAll(`input[name="${name}"]`).forEach((input) => {
    input.checked = false;
  });
}

function setRadioValue(name, value) {
  const input = document.querySelector(`input[name="${name}"][value="${value}"]`);
  if (input) input.checked = true;
}

function updateInitialProgress() {
  queueCount.textContent = `${completedInitialReviews} / ${queue.length}`;
  reviewProgress.style.width = `${queue.length ? (completedInitialReviews / queue.length) * 100 : 0}%`;
}

function renderQueue() {
  queueList.innerHTML = "";
  queue.forEach((item, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `queue-item${index === selectedIndex ? " active" : ""}`;
    button.innerHTML = `
      <span>${item.image}</span>
      <strong>${item.review_reason}</strong>
      <small>Diagnosis hidden</small>
    `;
    button.addEventListener("click", () => {
      selectedIndex = index;
      selectInitialCase();
    });
    queueList.appendChild(button);
  });
  updateInitialProgress();
}

function selectInitialCase() {
  const item = queue[selectedIndex];
  if (!item) return;
  document.querySelectorAll(".queue-item").forEach((node, index) => {
    node.classList.toggle("active", index === selectedIndex);
  });
  caseImage.src = imagePath(item.image);
  caseImage.alt = `${item.image} lesion image`;
  caseId.textContent = item.image;
  caseReason.textContent = item.review_reason;
  hiddenLabelText.textContent = "Hidden until submitted";
  ageGroupText.textContent = item.age_group;
  sexText.textContent = item.sex_clean;
  ageSexText.textContent = `${item.age_group} | ${item.sex_clean}`;
  clearRadioValue("diagnosis");
  setRadioValue("confidence", "medium");
  notes.value = "";
  challengeNotes.value = "";
  challengePanel.classList.add("is-hidden");
  pendingDisagreement = null;
  reviewStatus.textContent = "Select a diagnosis, add notes if needed, then submit.";
  reviewStatus.className = "status-line";
}

async function loadSummary() {
  try {
    const summary = await api("/api/summary");
    serverNote.textContent = `Live backend connected. Pending reviews: ${summary.pending_review_count.toLocaleString()}. Post-training rereview: ${summary.post_training_rereview_count.toLocaleString()}.`;
  } catch {
    serverNote.innerHTML = 'Static preview mode. Run <code>python3 local_pipeline_server.py</code> and open <code>http://127.0.0.1:8502</code> to save reviews and train.';
  }
}

async function loadQueue() {
  try {
    const items = await api(`/api/queue?limit=${REVIEW_BATCH_SIZE}&status=pending`);
    queue = items.length ? items.slice(0, REVIEW_BATCH_SIZE) : [];
  } catch {
    queue = fallbackCases.slice(0, REVIEW_BATCH_SIZE);
  }
  if (!queue.length) {
    queue = fallbackCases.slice(0, REVIEW_BATCH_SIZE);
  }
  selectedIndex = 0;
  completedInitialReviews = 0;
  renderQueue();
  selectInitialCase();
}

async function saveReview(payload) {
  await api("/api/review", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

function currentCase() {
  return queue[selectedIndex];
}

function advanceInitialReview() {
  completedInitialReviews += 1;
  updateInitialProgress();
  if (completedInitialReviews >= queue.length) {
    reviewStatus.textContent = "Initial review batch complete. Training is ready.";
    reviewStatus.className = "status-line ok";
    trainingSection.classList.remove("is-hidden");
    setPhase("#phase-training");
    trainingSection.scrollIntoView({ behavior: "smooth", block: "start" });
    return;
  }
  selectedIndex = completedInitialReviews;
  renderQueue();
  selectInitialCase();
}

async function submitInitialReview() {
  const item = currentCase();
  const selectedDiagnosis = checkedRadioValue("diagnosis");
  const confidence = checkedRadioValue("confidence") || "medium";
  if (!item || !selectedDiagnosis) {
    reviewStatus.textContent = "Please select a diagnosis first.";
    reviewStatus.className = "status-line bad";
    return;
  }

  hiddenLabelText.textContent = item.diagnosis;
  if (selectedDiagnosis === item.diagnosis) {
    try {
      await saveReview({
        image: item.image,
        review_status: "confirm labels",
        reviewed_diagnosis: selectedDiagnosis,
        reviewed_age: item.age_group,
        reviewed_sex: item.sex_clean,
        reviewer_confidence: confidence,
        reviewer_notes: notes.value,
      });
      reviewStatus.textContent = "Correct. Moving to the next review.";
      reviewStatus.className = "status-line ok";
      advanceInitialReview();
    } catch {
      reviewStatus.textContent = "Could not save review. Start the local backend first.";
      reviewStatus.className = "status-line bad";
    }
    return;
  }

  pendingDisagreement = item;
  challengeCopy.textContent = `Ground truth says ${item.diagnosis}. You selected ${selectedDiagnosis}. Why do you think the ground truth is not correct?`;
  challengePanel.classList.remove("is-hidden");
  reviewStatus.textContent = "Please add a second review explanation before continuing.";
  reviewStatus.className = "status-line bad";
}

async function submitChallengeReview() {
  if (!pendingDisagreement) return;
  const selectedDiagnosis = checkedRadioValue("diagnosis");
  const confidence = checkedRadioValue("confidence") || "low";
  if (!challengeNotes.value.trim()) {
    reviewStatus.textContent = "Please explain the disagreement before saving.";
    reviewStatus.className = "status-line bad";
    return;
  }
  try {
    await saveReview({
      image: pendingDisagreement.image,
      review_status: "request second review",
      reviewed_diagnosis: selectedDiagnosis,
      reviewed_age: pendingDisagreement.age_group,
      reviewed_sex: pendingDisagreement.sex_clean,
      reviewer_confidence: confidence,
      reviewer_notes: `Disagreement with ground truth ${pendingDisagreement.diagnosis}: ${challengeNotes.value}`,
    });
    reviewStatus.textContent = "Second review saved. Moving to the next case.";
    reviewStatus.className = "status-line ok";
    advanceInitialReview();
  } catch {
    reviewStatus.textContent = "Could not save second review. Start the local backend first.";
    reviewStatus.className = "status-line bad";
  }
}

async function saveSpecialAction(action) {
  const item = currentCase();
  const selectedDiagnosis = checkedRadioValue("diagnosis");
  const confidence = checkedRadioValue("confidence") || "low";
  if (!item) return;
  try {
    await saveReview({
      image: item.image,
      review_status: action,
      reviewed_diagnosis: selectedDiagnosis,
      reviewed_age: item.age_group,
      reviewed_sex: item.sex_clean,
      reviewer_confidence: confidence,
      reviewer_notes: notes.value,
    });
    reviewStatus.textContent = `${action} saved. Moving to the next case.`;
    reviewStatus.className = "status-line ok";
    advanceInitialReview();
  } catch {
    reviewStatus.textContent = "Could not save review. Start the local backend first.";
    reviewStatus.className = "status-line bad";
  }
}

function renderTrainingMetrics(metrics) {
  const values = metrics
    ? [
        `${(metrics.accuracy * 100).toFixed(1)}%`,
        `${(metrics.balanced_accuracy * 100).toFixed(1)}%`,
        metrics.post_training_rereview_count.toLocaleString(),
      ]
    : ["-", "-", "-"];
  [...trainingMetrics.querySelectorAll("strong")].forEach((node, index) => {
    node.textContent = values[index];
  });
}

function setTrainingRunning(isRunning) {
  startTrainingButton.disabled = isRunning;
  interruptTrainingButton.classList.toggle("is-hidden", !isRunning);
}

async function pollTrainingStatus() {
  try {
    const status = await api("/api/training-status");
    trainingStatus.textContent = status.message;
    trainingStatus.className = status.state === "failed" || status.state === "interrupted" ? "status-line bad" : "status-line ok";
    trainingProgress.style.width = `${status.progress || 0}%`;
    renderTrainingMetrics(status.metrics);
    setTrainingRunning(status.state === "running");

    if (status.state === "finished" || status.state === "failed" || status.state === "interrupted") {
      clearInterval(trainingPoll);
      trainingPoll = null;
      if (status.state === "finished") {
        await loadPostTrainingRereview();
        setPhase("#phase-rereview");
      }
    }
  } catch {
    trainingStatus.textContent = "Could not reach the training backend.";
    trainingStatus.className = "status-line bad";
    clearInterval(trainingPoll);
    trainingPoll = null;
  }
}

async function startTraining() {
  const payload = {
    max_train_rows: Number(document.querySelector("#max-train-rows").value),
    max_validation_rows: Number(document.querySelector("#max-validation-rows").value),
  };
  try {
    setTrainingRunning(true);
    await api("/api/train", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    trainingStatus.textContent = "Training started.";
    trainingStatus.className = "status-line ok";
    trainingProgress.style.width = "5%";
    if (trainingPoll) clearInterval(trainingPoll);
    trainingPoll = setInterval(pollTrainingStatus, 1500);
    await pollTrainingStatus();
  } catch {
    setTrainingRunning(false);
    trainingStatus.textContent = "Could not start training. Run python3 local_pipeline_server.py first.";
    trainingStatus.className = "status-line bad";
  }
}

async function interruptTraining() {
  try {
    interruptTrainingButton.disabled = true;
    trainingStatus.textContent = "Interrupt requested. Waiting for a safe stop point.";
    trainingStatus.className = "status-line bad";
    await api("/api/interrupt-training", {
      method: "POST",
      body: JSON.stringify({}),
    });
    await pollTrainingStatus();
  } catch {
    trainingStatus.textContent = "Could not interrupt training, or no run is active.";
    trainingStatus.className = "status-line bad";
  } finally {
    interruptTrainingButton.disabled = false;
  }
}

function renderRereviewSummary() {
  if (!rereviewQueue.length) {
    rereviewTable.innerHTML = '<p class="note">No post-training rereview cases were generated. Review completed successfully.</p>';
    completeWorkflow();
    return;
  }
  rereviewTable.innerHTML = `
    <p class="note">${rereviewQueue.length} post-training cases require rereview. Complete them one by one below.</p>
  `;
  rereviewCard.classList.remove("is-hidden");
  rereviewIndex = 0;
  selectRereviewCase();
}

function selectRereviewCase() {
  const row = rereviewQueue[rereviewIndex];
  if (!row) {
    completeWorkflow();
    return;
  }
  rereviewImage.src = imagePath(row.image);
  rereviewId.textContent = `${row.image} (${rereviewIndex + 1} / ${rereviewQueue.length})`;
  rereviewReason.textContent = row.rereview_reason;
  rereviewContext.textContent = `Model predicted ${row.predicted_diagnosis} with confidence ${Number(row.confidence).toFixed(2)}. Ground truth is hidden until you submit. Group: ${row.age_x_sex}.`;
  clearRadioValue("rereview-diagnosis");
  setRadioValue("rereview-confidence", "medium");
  rereviewNotes.value = "";
  rereviewStatus.textContent = "Select your rereview diagnosis and submit.";
  rereviewStatus.className = "status-line";
}

async function loadPostTrainingRereview() {
  try {
    rereviewQueue = await api("/api/post-training-rereview");
  } catch {
    rereviewQueue = [];
    rereviewTable.innerHTML = '<p class="note">Start the local backend to view post-training rereview cases.</p>';
    return;
  }
  renderRereviewSummary();
}

async function submitRereview() {
  const row = rereviewQueue[rereviewIndex];
  const selectedDiagnosis = checkedRadioValue("rereview-diagnosis");
  const confidence = checkedRadioValue("rereview-confidence") || "medium";
  if (!row || !selectedDiagnosis) {
    rereviewStatus.textContent = "Please select a diagnosis.";
    rereviewStatus.className = "status-line bad";
    return;
  }
  if (selectedDiagnosis !== row.diagnosis && !rereviewNotes.value.trim()) {
    rereviewStatus.textContent = `Ground truth says ${row.diagnosis}. Why do you think this is not correct? Add a comment before continuing.`;
    rereviewStatus.className = "status-line bad";
    return;
  }

  try {
    await saveReview({
      image: row.image,
      review_status: selectedDiagnosis === row.diagnosis ? "confirm labels" : "request second review",
      reviewed_diagnosis: selectedDiagnosis,
      reviewed_age: row.age_group,
      reviewed_sex: row.sex_clean,
      reviewer_confidence: confidence,
      reviewer_notes:
        selectedDiagnosis === row.diagnosis
          ? `Post-training rereview confirmed. ${rereviewNotes.value}`
          : `Post-training disagreement with ground truth ${row.diagnosis}: ${rereviewNotes.value}`,
    });
    rereviewIndex += 1;
    selectRereviewCase();
  } catch {
    rereviewStatus.textContent = "Could not save rereview. Start the local backend first.";
    rereviewStatus.className = "status-line bad";
  }
}

function completeWorkflow() {
  rereviewCard.classList.add("is-hidden");
  completeSection.classList.remove("is-hidden");
  setPhase("#phase-complete");
  completeSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

startReviewButton.addEventListener("click", async () => {
  reviewWorkspace.classList.remove("is-hidden");
  setPhase("#phase-review");
  await loadQueue();
  reviewWorkspace.scrollIntoView({ behavior: "smooth", block: "start" });
});

document.querySelector("#submit-review").addEventListener("click", submitInitialReview);
document.querySelector("#submit-challenge").addEventListener("click", submitChallengeReview);
document.querySelectorAll("[data-review-action]").forEach((button) => {
  button.addEventListener("click", () => saveSpecialAction(button.dataset.reviewAction));
});
startTrainingButton.addEventListener("click", startTraining);
interruptTrainingButton.addEventListener("click", interruptTraining);
document.querySelector("#submit-rereview").addEventListener("click", submitRereview);

loadSummary();
