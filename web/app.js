// app.js - Frontend Logic for Parcel Database Dashboard
// Categories and styling are dynamically driven by categories.json as the single source of truth.

let categoriesList = [];
const categoriesMap = new Map();
let allParcels = [];
let activeReviewCoord = null;

// Initialize on page load
document.addEventListener("DOMContentLoaded", async () => {
  await loadCategoriesConfig();
  bindKeyboardShortcuts();
  await loadAllData();
});

// -----------------------------------------------------------------------------
// CONFIGURATION & CATEGORY TEMPLATES
// -----------------------------------------------------------------------------
async function loadCategoriesConfig() {
  try {
    const res = await fetch("/categories.json");
    if (!res.ok) {
      throw new Error(`Failed to load categories.json: ${res.statusText}`);
    }
    categoriesList = await res.json();

    // Index categories by id and aliases for fast lookup
    categoriesMap.clear();
    categoriesList.forEach((cat) => {
      categoriesMap.set(cat.id, cat);
      if (Array.isArray(cat.aliases)) {
        cat.aliases.forEach((alias) => categoriesMap.set(alias, cat));
      }
    });

    renderDynamicMetricCards();
    renderCategoryFilterOptions();
    renderOperatorDecisionOptions();
  } catch (err) {
    console.error("Error loading categories configuration:", err);
  }
}

function getCategoryInfo(statusName) {
  if (!statusName) return null;
  if (categoriesMap.has(statusName)) {
    return categoriesMap.get(statusName);
  }
  const lower = statusName.toLowerCase();
  for (const [key, cat] of categoriesMap.entries()) {
    if (key.toLowerCase() === lower) {
      return cat;
    }
  }
  return null;
}

function renderDynamicMetricCards() {
  const container = document.getElementById("dynamic-metrics-container");
  const template = document.getElementById("metric-card-template");
  if (!container || !template) return;

  const fragment = document.createDocumentFragment();

  categoriesList.forEach((cat) => {
    const clone = template.content.cloneNode(true);
    const card = clone.querySelector(".dynamic-cat-card");

    if (cat.color) card.style.setProperty("--cat-color", cat.color);
    if (cat.bg) card.style.setProperty("--cat-bg", cat.bg);
    if (cat.borderColor) card.style.setProperty("--cat-border", cat.borderColor);

    card.dataset.categoryId = cat.id;

    const titleEl = clone.querySelector(".metric-title");
    titleEl.textContent = `${cat.icon ? cat.icon + " " : ""}${cat.label || cat.id}`;

    const valueEl = clone.querySelector(".metric-value");
    valueEl.id = `val-cat-${cat.id.toLowerCase().replace(/\s+/g, "-")}`;
    valueEl.textContent = "0";

    const subEl = clone.querySelector(".metric-sub");
    subEl.textContent = cat.description || "";

    fragment.appendChild(clone);
  });

  container.replaceChildren(fragment);
}

function renderCategoryFilterOptions() {
  const select = document.getElementById("filter-status");
  const template = document.getElementById("select-option-template");
  if (!select || !template) return;

  // Preserve 'all' and 'flagged' static options
  const defaultOptions = Array.from(select.querySelectorAll("option[value='all'], option[value='flagged']"));
  select.replaceChildren(...defaultOptions);

  const fragment = document.createDocumentFragment();
  categoriesList.forEach((cat) => {
    const clone = template.content.cloneNode(true);
    const opt = clone.querySelector("option");
    opt.value = cat.id;
    opt.textContent = `${cat.icon ? cat.icon + " " : ""}${cat.label || cat.id}`;
    fragment.appendChild(clone);
  });

  select.appendChild(fragment);
}

function renderOperatorDecisionOptions() {
  const select = document.getElementById("operator-decision");
  const template = document.getElementById("select-option-template");
  const label = document.getElementById("decision-title-label");
  if (!select || !template) return;

  // Preserve disabled placeholder option
  const placeholderOption = select.querySelector("option[value='']");
  select.replaceChildren(placeholderOption || new Option("Choose a decision...", "", true, true));

  const fragment = document.createDocumentFragment();
  categoriesList.forEach((cat) => {
    const clone = template.content.cloneNode(true);
    const opt = clone.querySelector("option");
    opt.value = cat.id;

    const keyPart = cat.key ? `[${cat.key}] ` : "";
    const iconPart = cat.icon ? `${cat.icon} ` : "";
    const notePart = cat.note ? ` ${cat.note}` : "";
    opt.textContent = `${keyPart}${iconPart}${cat.label || cat.id}${notePart}`;

    fragment.appendChild(clone);
  });

  select.appendChild(fragment);

  // Update shortcut range indicator in the label if keys exist
  const keys = categoriesList.filter((c) => c.key).map((c) => c.key);
  if (label && keys.length > 0) {
    label.textContent = `Select Operator Decision [${keys[0]}-${keys[keys.length - 1]}]:`;
  }
}

// -----------------------------------------------------------------------------
// DATA FETCHING & RENDERING
// -----------------------------------------------------------------------------
async function loadAllData() {
  try {
    // 1. Fetch System & DB Status
    const resStatus = await fetch("/api/status");
    const statusData = await resStatus.json();

    updateDbStatusBadge(statusData.db_exists);
    updateMetrics(statusData);

    // 2. Fetch Parcels into In-Memory Cache
    const resParcels = await fetch("/api/parcels");
    const parcelsData = await resParcels.json();
    allParcels = parcelsData.parcels || [];

    // 3. Render Table from cached data
    renderTable();
    updateWarningBanner();
  } catch (err) {
    console.error("Error loading data:", err);
  }
}

function updateDbStatusBadge(exists) {
  const badge = document.getElementById("db-status-badge");
  const text = document.getElementById("db-status-text");
  if (!badge || !text) return;

  if (exists) {
    badge.className = "badge badge-active";
    text.textContent = " Database Active (farm_parcels.db)";
  } else {
    badge.className = "badge badge-neutral";
    text.textContent = "○ Database Not Initialized Yet";
  }
}

function updateMetrics(statusData) {
  const totalEl = document.getElementById("val-total");
  const flaggedEl = document.getElementById("val-flagged");
  if (totalEl) totalEl.textContent = statusData.total_parcels || 0;
  if (flaggedEl) flaggedEl.textContent = statusData.flagged_count || 0;

  const summary = statusData.summary || {};

  // Dynamically update each category card based on categoriesList
  categoriesList.forEach((cat) => {
    const cardValEl = document.getElementById(`val-cat-${cat.id.toLowerCase().replace(/\s+/g, "-")}`);
    if (cardValEl) {
      let count = summary[cat.id] || 0;
      if (Array.isArray(cat.aliases)) {
        cat.aliases.forEach((alias) => {
          count += summary[alias] || 0;
        });
      }
      cardValEl.textContent = count;
    }
  });

  const cardFlagged = document.getElementById("card-flagged");
  if (cardFlagged) {
    if (statusData.flagged_count > 0) {
      cardFlagged.style.borderColor = "rgba(234, 179, 8, 0.6)";
    } else {
      cardFlagged.style.borderColor = "var(--border-color)";
    }
  }
}

function updateWarningBanner() {
  const flagged = allParcels.filter((p) => p.flag);
  const banner = document.getElementById("warning-banner");
  const countSpan = document.getElementById("warning-flagged-count");

  if (!banner) return;

  if (flagged.length > 0) {
    banner.classList.remove("hidden");
    if (countSpan) countSpan.textContent = flagged.length;
  } else {
    banner.classList.add("hidden");
  }
}

// In-memory filter & template-cloned table rendering
function renderTable() {
  const tbody = document.getElementById("parcels-tbody");
  const filterSelect = document.getElementById("filter-status");
  const rowTemplate = document.getElementById("parcel-row-template");
  const emptyTemplate = document.getElementById("empty-state-template");
  const countBadge = document.getElementById("record-count-badge");

  if (!tbody || !rowTemplate || !emptyTemplate) return;

  const filterVal = filterSelect ? filterSelect.value : "all";

  // Filter in-memory data
  let filtered = allParcels;
  if (filterVal === "flagged") {
    filtered = allParcels.filter((p) => p.flag);
  } else if (filterVal !== "all") {
    filtered = allParcels.filter((p) => {
      if (p.status === filterVal) return true;
      const cat = getCategoryInfo(p.status);
      return cat && cat.id === filterVal;
    });
  }

  if (countBadge) {
    countBadge.textContent = `${filtered.length} record(s)`;
  }

  // Handle empty states via template cloning
  if (allParcels.length === 0) {
    const emptyClone = emptyTemplate.content.cloneNode(true);
    emptyClone.querySelector(".empty-icon").textContent = "🗄️";
    emptyClone.querySelector(".empty-title").textContent = "Database not created yet";
    emptyClone.querySelector(".empty-desc").textContent =
      'Click "Classify Input Directory" above to scan images, create farm_parcels.db, and populate records.';
    tbody.replaceChildren(emptyClone);
    return;
  }

  if (filtered.length === 0) {
    const emptyClone = emptyTemplate.content.cloneNode(true);
    emptyClone.querySelector(".empty-icon").textContent = "🔍";
    emptyClone.querySelector(".empty-title").textContent = "No parcels match filter";
    emptyClone.querySelector(".empty-desc").textContent =
      'Try switching the status filter above to "All Statuses".';
    tbody.replaceChildren(emptyClone);
    return;
  }

  // Populate data rows via template cloning
  const fragment = document.createDocumentFragment();

  filtered.forEach((p) => {
    const clone = rowTemplate.content.cloneNode(true);
    const row = clone.querySelector("tr");

    if (p.flag) {
      row.classList.add("table-row-flagged");
    }

    // Coordinate & Google Maps link
    const coordParts = (p.coordinate || "").split(",").map((s) => s.trim());
    const lat = coordParts[0] || "";
    const lon = coordParts[1] || "";
    const coordLink = clone.querySelector(".coord-link");
    coordLink.textContent = `📍 ${p.coordinate}`;
    coordLink.href = `https://www.google.com/maps?q=${encodeURIComponent(lat)},${encodeURIComponent(lon)}`;

    // Filename badge
    const fileBadge = clone.querySelector(".file-badge");
    if (p.filename) {
      fileBadge.textContent = `📁 ${p.filename}`;
    } else {
      fileBadge.remove();
    }

    // Date
    clone.querySelector(".col-date").textContent = p.date || "--";

    // Dynamic Category Status Pill
    const cat = getCategoryInfo(p.status);
    const pill = clone.querySelector(".status-pill");
    const statusIcon = clone.querySelector(".status-icon");
    const statusName = clone.querySelector(".status-name");

    if (cat) {
      if (cat.color) pill.style.setProperty("--cat-color", cat.color);
      if (cat.bg) pill.style.setProperty("--cat-bg", cat.bg);
      if (cat.borderColor) pill.style.setProperty("--cat-border", cat.borderColor);
      statusIcon.textContent = cat.icon || "📦";
      statusName.textContent = cat.label || p.status;
    } else {
      statusIcon.textContent = "📦";
      statusName.textContent = p.status;
    }

    // Confidence
    const confBadge = clone.querySelector(".conf-badge");
    const confVal = p.confidence !== null ? `${(p.confidence * 100).toFixed(1)}%` : "--";
    const isLow = p.confidence !== null && p.confidence < 0.8;
    confBadge.textContent = isLow ? `${confVal} ⚠️` : confVal;
    if (isLow) {
      confBadge.classList.add("conf-low");
    }

    // Flag badge
    const flagBadge = clone.querySelector(".flag-badge");
    if (p.flag) {
      flagBadge.className = "flag-badge flag-yes";
      flagBadge.textContent = "⚠️ Flagged";
    } else {
      flagBadge.className = "flag-badge flag-no";
      flagBadge.textContent = "✓ Verified";
    }

    // Operator action buttons
    const btnInspect = clone.querySelector(".btn-inspect");
    const txtDiscarded = clone.querySelector(".picture-discarded-text");
    const btnDelete = clone.querySelector(".btn-delete");

    if (p.has_picture) {
      txtDiscarded.remove();
      btnInspect.addEventListener("click", () => openReviewModal(p.coordinate));
    } else {
      btnInspect.remove();
      txtDiscarded.style.display = "inline";
    }

    btnDelete.addEventListener("click", () => deleteParcel(p.coordinate));

    fragment.appendChild(clone);
  });

  tbody.replaceChildren(fragment);
}

// -----------------------------------------------------------------------------
// CLASSIFICATION TRIGGER
// -----------------------------------------------------------------------------
async function triggerClassification() {
  const btn = document.getElementById("btn-classify");
  const icon = btn ? btn.querySelector(".btn-icon") : null;
  const label = btn ? btn.querySelector(".btn-label") : null;

  if (btn) btn.disabled = true;
  if (icon) icon.textContent = "⏳";
  if (label) label.textContent = "Classifying Input Directory...";

  try {
    const res = await fetch("/api/classify", { method: "POST" });
    const data = await res.json();

    if (data.success) {
      alert(`🎉 Classification Complete!\n\n${data.message}`);
    } else {
      alert(`Notice: ${data.message || data.error}`);
    }
    await loadAllData();
  } catch (err) {
    alert(`Classification failed: ${err}`);
  } finally {
    if (btn) btn.disabled = false;
    if (icon) icon.textContent = "⚡";
    if (label) label.textContent = "Classify Input Directory";
  }
}

// -----------------------------------------------------------------------------
// OPERATOR REVIEW MODAL & DECISIONS
// -----------------------------------------------------------------------------
function openReviewModal(coord) {
  activeReviewCoord = coord;
  const p = allParcels.find((item) => item.coordinate === coord);
  if (!p) return;

  const decisionSelect = document.getElementById("operator-decision");
  if (decisionSelect) decisionSelect.value = "";
  updateDecisionButton();

  document.getElementById("modal-coord").textContent = coord;
  document.getElementById("modal-date").textContent = p.date || "--";

  const cat = getCategoryInfo(p.status);
  const iconPart = cat && cat.icon ? `${cat.icon} ` : "";
  document.getElementById("modal-status").textContent = `${iconPart}${p.status}`;

  document.getElementById("modal-conf").textContent =
    p.confidence !== null ? `${(p.confidence * 100).toFixed(1)}%` : "--";

  const img = document.getElementById("modal-img");
  if (img) {
    img.src = `/api/picture?coordinate=${encodeURIComponent(coord)}&t=${Date.now()}`;
  }

  document.getElementById("review-modal").classList.remove("hidden");
}

function closeModal() {
  const modal = document.getElementById("review-modal");
  if (modal) modal.classList.add("hidden");
  activeReviewCoord = null;
}

function updateDecisionButton() {
  const decisionSelect = document.getElementById("operator-decision");
  const submitBtn = document.getElementById("btn-submit-decision");
  if (!decisionSelect || !submitBtn) return;

  const decision = decisionSelect.value;
  submitBtn.disabled = !decision;

  if (decision) {
    const cat = getCategoryInfo(decision);
    if (cat) {
      submitBtn.style.setProperty("--decision-submit-bg", cat.hoverBg || cat.color || "");
      submitBtn.style.setProperty("--decision-submit-border", cat.borderColor || "transparent");
    }
  } else {
    submitBtn.style.removeProperty("--decision-submit-bg");
    submitBtn.style.removeProperty("--decision-submit-border");
  }
}

function submitSelectedDecision() {
  const decisionSelect = document.getElementById("operator-decision");
  const decision = decisionSelect ? decisionSelect.value : "";
  if (!decision) return;
  submitDecision(decision);
}

async function submitDecision(decision) {
  if (!activeReviewCoord || !decision) return;

  const coord = activeReviewCoord;
  try {
    const res = await fetch("/api/review", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ coordinate: coord, decision: decision }),
    });

    const data = await res.json();
    if (data.success) {
      closeModal();
      await loadAllData();
    } else {
      alert(`Error updating parcel: ${data.error}`);
    }
  } catch (err) {
    alert(`Review submission failed: ${err}`);
  }
}

function bindKeyboardShortcuts() {
  document.addEventListener("keydown", (e) => {
    const modal = document.getElementById("review-modal");
    if (!modal || modal.classList.contains("hidden")) return;

    if (e.key === "Escape") {
      closeModal();
      return;
    }

    // Dynamic shortcut matching based on categories.json
    const matchedCategory = categoriesList.find((cat) => String(cat.key) === e.key);
    if (matchedCategory) {
      const decisionSelect = document.getElementById("operator-decision");
      if (decisionSelect) {
        decisionSelect.value = matchedCategory.id;
        updateDecisionButton();
      }
      submitDecision(matchedCategory.id);
    }
  });
}

// -----------------------------------------------------------------------------
// PARCEL DELETION
// -----------------------------------------------------------------------------
async function deleteParcel(coord) {
  if (!confirm(`Are you sure you want to permanently delete parcel '${coord}'?`)) {
    return;
  }

  try {
    const res = await fetch("/api/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ coordinate: coord }),
    });
    const data = await res.json();
    if (data.success) {
      if (activeReviewCoord === coord) {
        closeModal();
      }
      await loadAllData();
    } else {
      alert(`Failed to delete parcel: ${data.error || data.message}`);
    }
  } catch (err) {
    alert(`Delete request failed: ${err}`);
  }
}

async function deleteActiveModalParcel() {
  if (!activeReviewCoord) return;
  await deleteParcel(activeReviewCoord);
}

async function clearAllParcels() {
  if (!allParcels.length) {
    alert("Database is already empty.");
    return;
  }

  const ans = confirm(
    `⚠️ DANGER: Are you sure you want to delete ALL ${allParcels.length} records and pictures from the database?\n\nThis cannot be undone.`
  );
  if (!ans) return;

  try {
    const res = await fetch("/api/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ all: true }),
    });
    const data = await res.json();
    if (data.success) {
      closeModal();
      await loadAllData();
      alert("✅ All records and stored images successfully deleted.");
    } else {
      alert(`Clear failed: ${data.error || data.message}`);
    }
  } catch (err) {
    alert(`Clear all request failed: ${err}`);
  }
}
