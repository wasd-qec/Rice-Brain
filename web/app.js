// app.js - Frontend Logic for Parcel Database Dashboard

let allParcels = [];
let activeReviewCoord = null;

// Initialize on load
document.addEventListener("DOMContentLoaded", () => {
  loadAllData();
  bindKeyboardShortcuts();
});

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

    // 2. Fetch Parcels
    const resParcels = await fetch("/api/parcels");
    const parcelsData = await resParcels.json();
    allParcels = parcelsData.parcels || [];

    renderTable();
    updateWarningBanner();
  } catch (err) {
    console.error("Error loading data:", err);
  }
}

function updateDbStatusBadge(exists) {
  const badge = document.getElementById("db-status-badge");
  const text = document.getElementById("db-status-text");

  if (exists) {
    badge.className = "badge badge-active";
    text.textContent = "● Database Active (farm_parcels.db)";
  } else {
    badge.className = "badge badge-neutral";
    text.textContent = "○ Database Not Initialized Yet";
  }
}

function updateMetrics(statusData) {
  document.getElementById("val-total").textContent = statusData.total_parcels || 0;
  document.getElementById("val-flagged").textContent = statusData.flagged_count || 0;

  const s = statusData.summary || {};
  document.getElementById("val-dry").textContent = s["Dry"] || 0;
  document.getElementById("val-flooded").textContent = s["Flooded"] || 0;
  document.getElementById("val-planted").textContent = s["Planted"] || 0;
  document.getElementById("val-others").textContent = s["Others"] || 0;

  const cardFlagged = document.getElementById("card-flagged");
  if (statusData.flagged_count > 0) {
    cardFlagged.style.borderColor = "rgba(234, 179, 8, 0.6)";
  } else {
    cardFlagged.style.borderColor = "var(--border-color)";
  }
}

function updateWarningBanner() {
  const flagged = allParcels.filter(p => p.flag);
  const banner = document.getElementById("warning-banner");
  const text = document.getElementById("warning-banner-text");

  if (flagged.length > 0) {
    banner.classList.remove("hidden");
    text.innerHTML = `⚠️ <strong>${flagged.length} parcel(s) require operator review</strong> (Planted or Low AI Confidence &lt;80%). Their JPEG images are preserved in the database. Please review them below to confirm category.`;
  } else {
    banner.classList.add("hidden");
  }
}

function renderTable() {
  const tbody = document.getElementById("parcels-tbody");
  const filterVal = document.getElementById("filter-status").value;

  let filtered = allParcels;
  if (filterVal === "flagged") {
    filtered = allParcels.filter(p => p.flag);
  } else if (filterVal !== "all") {
    filtered = allParcels.filter(p => p.status === filterVal);
  }

  document.getElementById("record-count-badge").textContent = `${filtered.length} record(s)`;

  if (allParcels.length === 0) {
    tbody.innerHTML = `
      <tr class="empty-row">
        <td colspan="6">
          <div class="empty-state">
            <span class="empty-icon">🗄️</span>
            <h3>Database not created yet</h3>
            <p>Click <strong>"Classify Input Directory"</strong> above to scan images, create <code>farm_parcels.db</code>, and populate records.</p>
          </div>
        </td>
      </tr>
    `;
    return;
  }

  if (filtered.length === 0) {
    tbody.innerHTML = `
      <tr class="empty-row">
        <td colspan="6">
          <div class="empty-state">
            <span class="empty-icon">🔍</span>
            <h3>No parcels match filter</h3>
            <p>Try switching the status filter above to "All Statuses".</p>
          </div>
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = filtered.map(p => {
    const statusClass = `status-${p.status.toLowerCase()}`;
    const statusIcon = getStatusIcon(p.status);
    const confText = p.confidence !== null ? `${(p.confidence * 100).toFixed(1)}%` : "--";
    const isLow = p.confidence !== null && p.confidence < 0.80;

    const flagBadge = p.flag
      ? `<span class="flag-badge flag-yes">⚠️ Flagged</span>`
      : `<span class="flag-badge flag-no">✓ Verified</span>`;

    const rowClass = p.flag ? "table-row-flagged" : "";

    const [lat, lon] = p.coordinate.split(",").map(s => s.trim());
    const mapsLink = `https://www.google.com/maps?q=${lat},${lon}`;

    const reviewBtn = p.has_picture
      ? `<button class="btn btn-secondary btn-sm" onclick="openReviewModal('${p.coordinate}')">
           🔍 Inspect & Review
         </button>`
      : `<span style="color: var(--text-dim); font-size: 11px;">Picture Discarded</span>`;

    return `
      <tr class="${rowClass}">
        <td>
          <a href="${mapsLink}" target="_blank" rel="noopener" class="coord-link mono">
            📍 ${p.coordinate}
          </a>
        </td>
        <td>${p.date || "--"}</td>
        <td>
          <span class="status-pill ${statusClass}">
            ${statusIcon} ${p.status}
          </span>
        </td>
        <td>
          <span style="color: ${isLow ? 'var(--color-danger)' : 'inherit'}; font-weight: ${isLow ? '600' : '400'}">
            ${confText} ${isLow ? '⚠️' : ''}
          </span>
        </td>
        <td>${flagBadge}</td>
        <td>${reviewBtn}</td>
      </tr>
    `;
  }).join("");
}

function getStatusIcon(status) {
  switch (status) {
    case "Dry": return "🏜️";
    case "Flooded": return "💧";
    case "Planted": return "🌿";
    case "Others": return "🌳";
    default: return "📦";
  }
}

// -----------------------------------------------------------------------------
// CLASSIFICATION TRIGGER
// -----------------------------------------------------------------------------
async function triggerClassification() {
  const btn = document.getElementById("btn-classify");
  const origHtml = btn.innerHTML;

  btn.disabled = true;
  btn.innerHTML = `<span>⏳</span><span>Classifying Input Directory...</span>`;

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
    btn.disabled = false;
    btn.innerHTML = origHtml;
  }
}

// -----------------------------------------------------------------------------
// OPERATOR REVIEW MODAL & DECISIONS
// -----------------------------------------------------------------------------
function openReviewModal(coord) {
  activeReviewCoord = coord;
  const p = allParcels.find(item => item.coordinate === coord);
  if (!p) return;

  document.getElementById("modal-coord").textContent = coord;
  document.getElementById("modal-date").textContent = p.date || "--";
  document.getElementById("modal-status").textContent = `${getStatusIcon(p.status)} ${p.status}`;
  document.getElementById("modal-conf").textContent = p.confidence ? `${(p.confidence * 100).toFixed(1)}%` : "--";

  const img = document.getElementById("modal-img");
  img.src = `/api/picture?coordinate=${encodeURIComponent(coord)}&t=${Date.now()}`;

  document.getElementById("review-modal").classList.remove("hidden");
}

function closeModal() {
  document.getElementById("review-modal").classList.add("hidden");
  activeReviewCoord = null;
}

async function submitDecision(decision) {
  if (!activeReviewCoord) return;

  const coord = activeReviewCoord;
  try {
    const res = await fetch("/api/review", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ coordinate: coord, decision: decision })
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
    if (modal && !modal.classList.contains("hidden")) {
      if (e.key === "Escape") {
        closeModal();
      } else if (e.key === "1") {
        submitDecision("Dry");
      } else if (e.key === "2") {
        submitDecision("Flooded");
      } else if (e.key === "3") {
        submitDecision("Planted");
      } else if (e.key === "4") {
        submitDecision("Others");
      }
    }
  });
}
