// wakeup-ai dashboard — metrics charts (Chart.js via CDN, no build step).
// Fetches GET /api/metrics client-side; the page already carries the
// Basic Auth session cookie/credential, so a same-origin fetch just works.

(function () {
  "use strict";

  var root = document.documentElement;
  var css = getComputedStyle(root);
  var COLOR = {
    emeraldBright: css.getPropertyValue("--accent-emerald-bright").trim() || "#2f9e64",
    emerald: css.getPropertyValue("--accent-emerald").trim() || "#1b6b45",
    maroonBright: css.getPropertyValue("--accent-maroon-bright").trim() || "#a8274f",
    maroon: css.getPropertyValue("--accent-maroon").trim() || "#7a1230",
    info: css.getPropertyValue("--state-info-fg").trim() || "#8fb6dd",
    warn: css.getPropertyValue("--state-warn-fg").trim() || "#e0b46a",
    neutral: css.getPropertyValue("--state-neutral-fg").trim() || "#b7b7bc",
    textMuted: css.getPropertyValue("--text-muted").trim() || "#9a9a9e",
    textSecondary: css.getPropertyValue("--text-secondary").trim() || "#c7c7cb",
    borderSubtle: css.getPropertyValue("--border-subtle").trim() || "#232427",
  };

  var REDUCE_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function fmtDate(iso) {
    var d = new Date(iso + "T00:00:00Z");
    if (isNaN(d.getTime())) return iso;
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric", timeZone: "UTC" });
  }

  var RELATIVE_UNITS = [
    ["year", 31536000],
    ["month", 2592000],
    ["week", 604800],
    ["day", 86400],
    ["hour", 3600],
    ["minute", 60],
    ["second", 1],
  ];

  function fmtRelativeTime(epochSeconds) {
    var diffSec = Math.round(Date.now() / 1000 - epochSeconds);
    var rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
    for (var i = 0; i < RELATIVE_UNITS.length; i++) {
      var unit = RELATIVE_UNITS[i][0];
      var secs = RELATIVE_UNITS[i][1];
      if (Math.abs(diffSec) >= secs || unit === "second") {
        return rtf.format(-Math.round(diffSec / secs), unit);
      }
    }
  }

  function setCardState(cardId, state, message) {
    var wrap = document.querySelector('[data-chart-wrap="' + cardId + '"]');
    if (!wrap) return;
    var canvas = wrap.querySelector("canvas");
    var note = wrap.querySelector(".chart-empty, .chart-loading");
    if (state === "ready") {
      if (canvas) canvas.style.display = "block";
      if (note) note.remove();
      return;
    }
    if (canvas) canvas.style.display = "none";
    if (!note) {
      note = document.createElement("div");
      wrap.appendChild(note);
    }
    note.className = state === "error" ? "chart-loading chart-error" : "chart-loading";
    note.textContent = message;
  }

  function baseChartOptions(extra) {
    var opts = {
      responsive: true,
      maintainAspectRatio: false,
      animation: REDUCE_MOTION ? false : { duration: 400 },
      plugins: {
        legend: {
          labels: { color: COLOR.textSecondary, boxWidth: 12, font: { size: 12 } },
        },
        tooltip: {
          backgroundColor: "#1a1b1f",
          borderColor: COLOR.borderSubtle,
          borderWidth: 1,
          titleColor: "#f2f2f0",
          bodyColor: COLOR.textSecondary,
          padding: 10,
        },
      },
      scales: {
        x: {
          ticks: { color: COLOR.textMuted, font: { size: 11 } },
          grid: { color: COLOR.borderSubtle, display: false },
        },
        y: {
          ticks: { color: COLOR.textMuted, font: { size: 11 }, precision: 0 },
          grid: { color: COLOR.borderSubtle },
          beginAtZero: true,
        },
      },
    };
    return Object.assign(opts, extra || {});
  }

  function renderTrend(daily) {
    setCardState("trend", "loading", "Loading trend…");
    if (!daily || daily.length === 0) {
      setCardState("trend", "empty", "No events in this window yet.");
      return;
    }
    var ctx = document.getElementById("chart-trend");
    new Chart(ctx, {
      type: "bar",
      data: {
        labels: daily.map(function (d) { return fmtDate(d.date); }),
        datasets: [
          {
            label: "Real wakes",
            data: daily.map(function (d) { return d.real_wakes; }),
            backgroundColor: COLOR.maroonBright,
            stack: "wakes",
          },
          {
            label: "False wakes",
            data: daily.map(function (d) { return d.false_wakes; }),
            backgroundColor: COLOR.neutral,
            stack: "wakes",
          },
          {
            label: "Non-escalating",
            data: daily.map(function (d) { return d.non_escalating_wakes; }),
            backgroundColor: COLOR.emeraldBright,
            stack: "wakes",
          },
        ],
      },
      options: baseChartOptions({
        scales: {
          x: { stacked: true, ticks: { color: COLOR.textMuted, font: { size: 11 } }, grid: { display: false } },
          y: { stacked: true, ticks: { color: COLOR.textMuted, font: { size: 11 }, precision: 0 }, grid: { color: COLOR.borderSubtle }, beginAtZero: true },
        },
      }),
    });
    setCardState("trend", "ready");
  }

  function renderVisionSource(breakdown) {
    setCardState("vision", "loading", "Loading breakdown…");
    var buckets = ["real_wakes", "false_wakes", "non_escalating_wakes"];
    var labels = ["Real wakes", "False wakes", "Non-escalating"];
    var counts = buckets.map(function (k) {
      var b = breakdown[k];
      return { gemini: b.gemini, failsafe: b.failsafe, none: b.none, total: b.gemini + b.failsafe + b.none };
    });
    var total = counts.reduce(function (sum, c) { return sum + c.total; }, 0);
    if (!breakdown || total === 0) {
      setCardState("vision", "empty", "No events in this window yet.");
      return;
    }
    // Non-escalating events vastly outnumber real/false wakes, so a shared
    // linear count scale would hide the gemini/failsafe split entirely.
    // Normalize each bucket to its own 100% instead — proportions are what
    // this chart is actually for; raw counts are still in the tooltip.
    function pct(count, bucketTotal) {
      return bucketTotal === 0 ? 0 : (count / bucketTotal) * 100;
    }
    var series = ["gemini", "failsafe", "none"];
    var seriesColor = { gemini: COLOR.info, failsafe: COLOR.warn, none: COLOR.neutral };
    var ctx = document.getElementById("chart-vision");
    new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: series.map(function (s) {
          return {
            label: s,
            data: counts.map(function (c) { return pct(c[s], c.total); }),
            rawCounts: counts.map(function (c) { return c[s]; }),
            backgroundColor: seriesColor[s],
          };
        }),
      },
      options: baseChartOptions({
        indexAxis: "y",
        scales: {
          x: {
            stacked: true,
            min: 0,
            max: 100,
            ticks: { color: COLOR.textMuted, font: { size: 11 }, callback: function (v) { return v + "%"; } },
            grid: { color: COLOR.borderSubtle },
          },
          y: {
            stacked: true,
            ticks: { color: COLOR.textMuted, font: { size: 11 } },
            grid: { display: false },
          },
        },
        plugins: {
          legend: {
            labels: { color: COLOR.textSecondary, boxWidth: 12, font: { size: 12 } },
          },
          tooltip: {
            backgroundColor: "#1a1b1f",
            borderColor: COLOR.borderSubtle,
            borderWidth: 1,
            titleColor: "#f2f2f0",
            bodyColor: COLOR.textSecondary,
            padding: 10,
            callbacks: {
              label: function (item) {
                var raw = item.dataset.rawCounts[item.dataIndex];
                return item.dataset.label + ": " + raw + " (" + item.parsed.x.toFixed(0) + "%)";
              },
            },
          },
        },
      }),
    });
    setCardState("vision", "ready");
  }

  function renderDelivery(summary) {
    setCardState("delivery", "loading", "Loading…");
    var totalAlarmish = summary.real_wakes + summary.false_wakes;
    if (totalAlarmish === 0) {
      setCardState("delivery", "empty", "No escalated events in this window yet.");
      return;
    }
    var rate = summary.email_delivery_rate;
    var delivered = Math.round(rate * 100);
    var missed = 100 - delivered;
    var ctx = document.getElementById("chart-delivery");
    new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: ["Delivered", "Not delivered"],
        datasets: [
          {
            data: [delivered, missed],
            backgroundColor: [COLOR.emeraldBright, COLOR.borderSubtle],
            borderWidth: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: REDUCE_MOTION ? false : { duration: 400 },
        cutout: "72%",
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#1a1b1f",
            borderColor: COLOR.borderSubtle,
            borderWidth: 1,
            titleColor: "#f2f2f0",
            bodyColor: COLOR.textSecondary,
          },
        },
      },
      plugins: [
        {
          id: "centerLabel",
          afterDraw: function (chart) {
            var c = chart.ctx;
            var w = chart.width, h = chart.height;
            c.save();
            c.textAlign = "center";
            c.textBaseline = "middle";
            c.fillStyle = "#f2f2f0";
            c.font = "650 1.5rem " + getComputedStyle(document.body).fontFamily;
            c.fillText(delivered + "%", w / 2, h / 2 - 6);
            c.fillStyle = COLOR.textMuted;
            c.font = "500 0.75rem " + getComputedStyle(document.body).fontFamily;
            c.fillText("delivered", w / 2, h / 2 + 16);
            c.restore();
          },
        },
      ],
    });
    setCardState("delivery", "ready");
  }

  function renderLatency(latency) {
    var wrap = document.getElementById("latency-stats");
    if (!wrap) return;
    if (!latency) {
      wrap.innerHTML = '<div class="chart-empty">No latency data yet.</div>';
      return;
    }
    var rows = [
      ["avg", latency.avg],
      ["p50", latency.p50],
      ["p95", latency.p95],
      ["max", latency.max],
    ];
    wrap.innerHTML = rows
      .map(function (r) {
        return (
          '<div class="stat"><div class="stat-value">' +
          r[1].toFixed(1) +
          's</div><div class="stat-label">' +
          r[0] +
          "</div></div>"
        );
      })
      .join("");
  }

  function renderSyncStatus(lastSync) {
    var el = document.getElementById("sync-status");
    if (!el) return;
    var textEl = el.querySelector(".sync-text");
    if (!lastSync) {
      el.classList.add("is-empty");
      el.removeAttribute("title");
      textEl.textContent = "No sync in the last 30 days";
      return;
    }
    el.classList.remove("is-empty");
    textEl.textContent = "Last synced " + fmtRelativeTime(lastSync.received_at);
    el.title = "Event " + lastSync.event_id + " received " + new Date(lastSync.received_at * 1000).toLocaleString();
  }

  function renderTriggerStat(breakdown) {
    var wrap = document.getElementById("trigger-stat");
    if (!wrap) return;
    if (!breakdown) {
      wrap.innerHTML = '<div class="chart-empty">No trigger data yet.</div>';
      return;
    }
    var outcomes = ["real_wakes", "false_wakes", "non_escalating_wakes"];
    var triggered = outcomes.reduce(function (sum, k) { return sum + breakdown.triggered[k]; }, 0);
    var notTriggered = outcomes.reduce(function (sum, k) { return sum + breakdown.not_triggered[k]; }, 0);
    var total = triggered + notTriggered;
    if (total === 0) {
      wrap.innerHTML = '<div class="chart-empty">No events in this window yet.</div>';
      return;
    }
    var isWarning = notTriggered > 0;
    wrap.classList.toggle("warning", isWarning);
    var icon = isWarning ? "⚠" : "✓";
    var headline = isWarning
      ? notTriggered + " of " + total + " wakes did NOT originate from a genuine SNN trigger"
      : triggered + "/" + total + " wakes were genuine SNN triggers";
    var detail = isWarning
      ? "Unexpected non-trigger boot reached the cloud — check the Pi for a manual or dev restart."
      : "Every wake in this window was hardware-latched, as expected.";
    wrap.innerHTML =
      '<div class="trigger-line"><span class="trigger-icon">' +
      icon +
      "</span>" +
      headline +
      '</div><div class="trigger-detail">' +
      detail +
      "</div>";
  }

  function renderReviewAccuracy(reviewAccuracy) {
    var wrap = document.getElementById("review-accuracy");
    if (!wrap) return;
    if (!reviewAccuracy || reviewAccuracy.reviewed_count === 0) {
      wrap.innerHTML = '<div class="chart-empty">Review an event to start tracking accuracy.</div>';
      return;
    }

    function block(title, counts) {
      var accuracyPct = Math.round(counts.accuracy * 100);
      return (
        '<div class="confusion-block"><h4>' +
        title +
        '</h4><div class="confusion-accuracy">' +
        accuracyPct +
        '<span class="unit">% accurate</span></div><div class="confusion-grid">' +
        '<div class="confusion-cell correct"><div class="confusion-count">' +
        counts.tp +
        '</div><div class="confusion-label">True positive</div></div>' +
        '<div class="confusion-cell incorrect"><div class="confusion-count">' +
        counts.fp +
        '</div><div class="confusion-label">False positive</div></div>' +
        '<div class="confusion-cell incorrect"><div class="confusion-count">' +
        counts.fn +
        '</div><div class="confusion-label">False negative</div></div>' +
        '<div class="confusion-cell correct"><div class="confusion-count">' +
        counts.tn +
        '</div><div class="confusion-label">True negative</div></div>' +
        "</div></div>"
      );
    }

    wrap.innerHTML =
      '<div class="review-accuracy-count"><strong>' +
      reviewAccuracy.reviewed_count +
      "</strong> event" +
      (reviewAccuracy.reviewed_count === 1 ? "" : "s") +
      ' manually reviewed</div><div class="confusion-grids">' +
      block("Window broken", reviewAccuracy.window_broken) +
      block("Intrusion", reviewAccuracy.intrusion) +
      "</div>";
  }

  function loadMetrics() {
    ["trend", "vision", "delivery"].forEach(function (id) {
      setCardState(id, "loading", "Loading…");
    });
    fetch("/api/metrics")
      .then(function (res) {
        if (!res.ok) throw new Error("metrics request failed: " + res.status);
        return res.json();
      })
      .then(function (data) {
        renderTrend(data.daily);
        renderVisionSource(data.vision_source_breakdown);
        renderDelivery(data.summary);
        renderLatency(data.latency_s);
        renderSyncStatus(data.last_sync);
        renderTriggerStat(data.trigger_breakdown);
        renderReviewAccuracy(data.review_accuracy);
      })
      .catch(function (err) {
        ["trend", "vision", "delivery"].forEach(function (id) {
          setCardState(id, "error", "Couldn't load chart data.");
        });
        var latencyWrap = document.getElementById("latency-stats");
        if (latencyWrap) latencyWrap.innerHTML = '<div class="chart-empty chart-error">Couldn’t load latency data.</div>';
        var triggerWrap = document.getElementById("trigger-stat");
        if (triggerWrap) triggerWrap.innerHTML = '<div class="chart-empty chart-error">Couldn’t load trigger data.</div>';
        var reviewWrap = document.getElementById("review-accuracy");
        if (reviewWrap) reviewWrap.innerHTML = '<div class="chart-empty chart-error">Couldn’t load review accuracy.</div>';
        var syncEl = document.getElementById("sync-status");
        if (syncEl) {
          var syncText = syncEl.querySelector(".sync-text");
          if (syncText) syncText.textContent = "Couldn't load sync status.";
        }
        // eslint-disable-next-line no-console
        console.error(err);
      });
  }

  function initRowClicks() {
    document.querySelectorAll("tr[data-href]").forEach(function (row) {
      row.setAttribute("tabindex", "0");
      row.addEventListener("click", function () {
        window.location = row.getAttribute("data-href");
      });
      row.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          window.location = row.getAttribute("data-href");
        }
      });
    });
  }

  function parseTriBool(attr) {
    if (attr === "true") return true;
    if (attr === "false") return false;
    return null;
  }

  function initReviewSection() {
    var section = document.getElementById("review-section");
    if (!section) return;

    var eventId = section.getAttribute("data-event-id");
    var predicted = {
      window_broken_confirmed: parseTriBool(section.getAttribute("data-window-broken")),
      intrusion_confirmed: parseTriBool(section.getAttribute("data-is-intrusion")),
    };
    var reviewedAt = section.getAttribute("data-reviewed-at") || null;
    var selections = {
      window_broken_confirmed: parseTriBool(section.getAttribute("data-window-broken-confirmed")),
      intrusion_confirmed: parseTriBool(section.getAttribute("data-intrusion-confirmed")),
    };

    var toggles = section.querySelectorAll(".review-toggle");
    var submitBtn = document.getElementById("review-submit");
    var statusEl = document.getElementById("review-status");
    var comparisonEl = document.getElementById("review-comparison");
    var labels = { window_broken_confirmed: "Window broken", intrusion_confirmed: "Intrusion" };

    function updateToggleUI() {
      toggles.forEach(function (toggle) {
        var field = toggle.getAttribute("data-field");
        var val = selections[field];
        toggle.querySelectorAll(".review-btn").forEach(function (btn) {
          var btnVal = btn.getAttribute("data-value") === "true";
          btn.classList.toggle("active", val !== null && val === btnVal);
        });
      });
      submitBtn.disabled = selections.window_broken_confirmed === null || selections.intrusion_confirmed === null;
    }

    function triText(val) {
      return val === null ? "n/a" : val ? "yes" : "no";
    }

    function renderComparison() {
      if (reviewedAt === null) {
        comparisonEl.style.display = "none";
        comparisonEl.innerHTML = "";
        return;
      }
      var fields = ["window_broken_confirmed", "intrusion_confirmed"];
      var html = fields
        .map(function (field) {
          var predictedVal = predicted[field];
          var confirmedVal = selections[field];
          var match = predictedVal === confirmedVal;
          return (
            '<div class="review-compare-row ' +
            (match ? "match" : "mismatch") +
            '"><span class="review-compare-icon">' +
            (match ? "✓" : "✕") +
            '</span><div class="review-compare-text"><div class="review-compare-label">' +
            labels[field] +
            '</div><div class="review-compare-values">Gemini said <strong>' +
            triText(predictedVal) +
            "</strong> — you confirmed <strong>" +
            triText(confirmedVal) +
            "</strong></div></div></div>"
          );
        })
        .join("");
      html += '<div class="review-compare-meta">Reviewed ' + fmtRelativeTime(parseFloat(reviewedAt)) + "</div>";
      comparisonEl.innerHTML = html;
      comparisonEl.style.display = "";
    }

    toggles.forEach(function (toggle) {
      var field = toggle.getAttribute("data-field");
      toggle.querySelectorAll(".review-btn").forEach(function (btn) {
        btn.addEventListener("click", function () {
          selections[field] = btn.getAttribute("data-value") === "true";
          updateToggleUI();
        });
      });
    });

    submitBtn.addEventListener("click", function () {
      if (selections.window_broken_confirmed === null || selections.intrusion_confirmed === null) return;
      submitBtn.disabled = true;
      statusEl.classList.remove("is-error");
      statusEl.textContent = "Submitting…";
      fetch("/api/events/" + eventId, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(selections),
      })
        .then(function (res) {
          if (!res.ok) throw new Error("review submit failed: " + res.status);
          return res.json();
        })
        .then(function (data) {
          reviewedAt = String(data.reviewed_at);
          statusEl.textContent = "Saved.";
          renderComparison();
        })
        .catch(function (err) {
          statusEl.classList.add("is-error");
          statusEl.textContent = "Couldn't save review — try again.";
          // eslint-disable-next-line no-console
          console.error(err);
        })
        .finally(function () {
          updateToggleUI();
        });
    });

    updateToggleUI();
    renderComparison();
  }

  document.addEventListener("DOMContentLoaded", function () {
    initRowClicks();
    if (document.getElementById("chart-trend")) {
      loadMetrics();
    }
    initReviewSection();
  });
})();
