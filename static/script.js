/**
 * ==============================================================================
 * SkyFare AI - Flight Fare Predictor Frontend Script (v2 - Scaled Up)
 * Handles route swapping, instant preset selection, source-destination validation,
 * async fetch communication with Flask, and animated INR fare counter display.
 * Now supports 30 Tier 1/2/3 cities and new factors:
 *   day_type, season (seasonality), booking_type (refundability).
 * ==============================================================================
 */

document.addEventListener("DOMContentLoaded", () => {
  // ── Core DOM references ─────────────────────────────────────────────────────
  const form         = document.getElementById("fare-form");
  const submitBtn    = document.getElementById("btn-submit");
  const resetBtn     = document.getElementById("btn-reset");
  const swapBtn      = document.getElementById("btn-swap-cities");

  const cityAlert       = document.getElementById("city-match-alert");
  const placeholderView = document.getElementById("result-placeholder");
  const boardingPassView = document.getElementById("boarding-pass");
  const errorBanner     = document.getElementById("error-banner");
  const errorMsg        = document.getElementById("error-msg");

  // ── Form Inputs ─────────────────────────────────────────────────────────────
  const sourceCityInput     = document.getElementById("source_city");
  const destCityInput       = document.getElementById("destination_city");
  const airlineInput        = document.getElementById("airline");
  const departureTimeInput  = document.getElementById("departure_time");
  const cabinClassInput     = document.getElementById("cabin_class");
  const stopsInput          = document.getElementById("stops");
  const durationInput       = document.getElementById("duration_hours");
  const daysLeftInput       = document.getElementById("days_left");
  // New scaled-up fields
  const dayTypeInput        = document.getElementById("day_type");
  const seasonInput         = document.getElementById("season");
  const bookingTypeInput    = document.getElementById("booking_type");

  // ── Boarding Pass Output Elements ───────────────────────────────────────────
  const resAirline     = document.getElementById("res-airline");
  const resClass       = document.getElementById("res-class");
  const resOrigin      = document.getElementById("res-origin");
  const resDestination = document.getElementById("res-destination");
  const resStops       = document.getElementById("res-stops");
  const resDuration    = document.getElementById("res-duration");
  const resFare        = document.getElementById("res-fare");
  const resRange       = document.getElementById("res-range");
  const resTime        = document.getElementById("res-time");
  const resDays        = document.getElementById("res-days");
  const resHourly      = document.getElementById("res-hourly");
  const resDemand      = document.getElementById("res-demand");
  const resAdvice      = document.getElementById("res-advice");
  // New result fields
  const resDistance    = document.getElementById("res-distance");
  const resPerKm       = document.getElementById("res-perkm");
  const resDayType     = document.getElementById("res-daytype");
  const resRouteTier   = document.getElementById("res-route-tier");

  // ── Demo Preset Scenarios (updated for 30-city scalability) ─────────────────
  //   Each preset includes the 3 new fields introduced in v2:
  //   day_type, season, booking_type
  const presets = {
    del_bom_business: {
      source_city:      "Delhi",
      destination_city: "Mumbai",
      airline:          "Vistara",
      departure_time:   "Morning",
      cabin_class:      "Business",
      stops:            "0",
      duration_hours:   2.2,
      days_left:        8,
      day_type:         "Weekday (Mon-Thu)",
      season:           "Regular",
      booking_type:     "Non-Refundable (Standard)",
    },
    bom_goa_weekend: {
      source_city:      "Mumbai",
      destination_city: "Goa",
      airline:          "IndiGo",
      departure_time:   "Morning",
      cabin_class:      "Economy",
      stops:            "0",
      duration_hours:   1.3,
      days_left:        5,
      day_type:         "Weekend (Fri-Sun)",
      season:           "Regular",
      booking_type:     "Non-Refundable (Standard)",
    },
    del_ded_regional: {
      source_city:      "Delhi",
      destination_city: "Dehradun",
      airline:          "IndiGo",
      departure_time:   "Morning",
      cabin_class:      "Economy",
      stops:            "0",
      duration_hours:   1.2,
      days_left:        3,
      day_type:         "Weekend (Fri-Sun)",
      season:           "Peak / Festive",
      booking_type:     "Flexible / Free Cancellation",
    },
    vns_cok_cross: {
      source_city:      "Varanasi",
      destination_city: "Kochi",
      airline:          "Air India",
      departure_time:   "Afternoon",
      cabin_class:      "Economy",
      stops:            "1",
      duration_hours:   5.5,
      days_left:        20,
      day_type:         "Weekday (Mon-Thu)",
      season:           "Regular",
      booking_type:     "Non-Refundable (Standard)",
    },
  };

  let currentAnimatedFare = 0;

  // ── Utility: animated fare counter ──────────────────────────────────────────
  /**
   * Smoothly rolls the INR price from startVal → endVal over durationMs ms.
   * Uses an ease-out cubic curve for a natural feel.
   */
  function animateFare(element, startVal, endVal, durationMs = 700) {
    const startTime = performance.now();

    function update(now) {
      const elapsed  = now - startTime;
      const progress = Math.min(elapsed / durationMs, 1);
      const easeOut  = 1 - Math.pow(1 - progress, 3);
      const current  = Math.floor(startVal + (endVal - startVal) * easeOut);

      element.textContent = `₹${current.toLocaleString("en-IN")}`;

      if (progress < 1) {
        requestAnimationFrame(update);
      } else {
        element.textContent = `₹${Math.round(endVal).toLocaleString("en-IN")}`;
      }
    }

    requestAnimationFrame(update);
  }

  // ── Validation: source ≠ destination ────────────────────────────────────────
  function checkCityDistinctness() {
    const src  = (sourceCityInput.value  || "").trim().toLowerCase();
    const dest = (destCityInput.value    || "").trim().toLowerCase();
    const isSame = src && dest && src === dest;

    if (isSame) {
      cityAlert.style.display = "block";
      submitBtn.disabled = true;
      return false;
    } else {
      cityAlert.style.display = "none";
      submitBtn.disabled = false;
      return true;
    }
  }

  sourceCityInput.addEventListener("change", checkCityDistinctness);
  destCityInput.addEventListener("change", checkCityDistinctness);

  // ── Swap cities button ───────────────────────────────────────────────────────
  swapBtn.addEventListener("click", () => {
    const temp = sourceCityInput.value;
    sourceCityInput.value = destCityInput.value;
    destCityInput.value   = temp;
    checkCityDistinctness();
  });

  // ── Build request payload from all form fields ───────────────────────────────
  function getPayload() {
    return {
      source_city:      sourceCityInput.value,
      destination_city: destCityInput.value,
      airline:          airlineInput.value,
      departure_time:   departureTimeInput.value,
      cabin_class:      cabinClassInput.value,
      stops:            parseInt(stopsInput.value, 10),
      duration_hours:   parseFloat(durationInput.value),
      days_left:        parseInt(daysLeftInput.value, 10),
      // New v2 fields
      day_type:         dayTypeInput     ? dayTypeInput.value     : "Weekday (Mon-Thu)",
      season:           seasonInput      ? seasonInput.value      : "Regular",
      booking_type:     bookingTypeInput ? bookingTypeInput.value : "Non-Refundable (Standard)",
    };
  }

  // ── Error display helpers ────────────────────────────────────────────────────
  function showError(message) {
    errorBanner.style.display = "flex";
    errorMsg.textContent = message;
    boardingPassView.style.display = "none";
    placeholderView.style.display  = "none";
  }

  function clearError() {
    errorBanner.style.display = "none";
    errorMsg.textContent = "";
  }

  // ── Safe text setter (skips if element doesn't exist) ────────────────────────
  function setText(el, value) {
    if (el) el.textContent = value;
  }

  // ── Core prediction logic ─────────────────────────────────────────────────────
  /**
   * Sends flight search parameters to Flask /predict,
   * then populates the boarding-pass result card.
   */
  async function submitPrediction() {
    clearError();

    if (!checkCityDistinctness()) {
      showError("Origin and destination cities cannot be identical.");
      return;
    }

    const payload = getPayload();

    submitBtn.classList.add("is-loading");
    submitBtn.disabled = true;

    try {
      const response = await fetch("/predict", {
        method:  "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept":        "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok || data.status !== "success") {
        throw new Error(data.message || "Failed to estimate flight fare.");
      }

      const pred = data.prediction;
      const inp  = data.inputs;

      // Show boarding pass
      placeholderView.style.display   = "none";
      boardingPassView.style.display  = "block";

      // ── Boarding pass header ──────────────────────────────────────────
      setText(resAirline,     inp.airline);
      setText(resClass,       inp.cabin_class);
      setText(resRouteTier,   pred.route_tier || "");

      // ── Route display ─────────────────────────────────────────────────
      setText(resOrigin,      inp.source_city);
      setText(resDestination, inp.destination_city);
      setText(resStops,       pred.stops_label);
      setText(resDuration,    `${inp.duration_hours} hrs`);

      // ── Stats grid ────────────────────────────────────────────────────
      setText(resTime,    inp.departure_time);
      setText(resDays,    `${inp.days_left} Days Out`);
      setText(resHourly,  pred.fare_per_hour);
      setText(resDemand,  pred.demand_level);

      // New v2 stats
      setText(resDistance, pred.distance_km   ? `${pred.distance_km} km` : "--");
      setText(resPerKm,    pred.fare_per_km   ? `₹${pred.fare_per_km}/km` : "--");
      setText(resDayType,  inp.day_type       || "--");

      // ── Fare band & advice ────────────────────────────────────────────
      setText(resRange,  `Market Band: ${pred.fare_range}`);
      setText(resAdvice, pred.booking_advice);

      // Animated fare counter
      animateFare(resFare, currentAnimatedFare, pred.raw_fare, 700);
      currentAnimatedFare = pred.raw_fare;

    } catch (err) {
      showError(err.message || "Connection to flight fare server failed.");
    } finally {
      submitBtn.classList.remove("is-loading");
      submitBtn.disabled = false;
    }
  }

  // ── Apply a preset scenario to the form + run prediction ────────────────────
  function applyPreset(presetKey) {
    const p = presets[presetKey];
    if (!p) return;

    sourceCityInput.value     = p.source_city;
    destCityInput.value       = p.destination_city;
    airlineInput.value        = p.airline;
    departureTimeInput.value  = p.departure_time;
    cabinClassInput.value     = p.cabin_class;
    stopsInput.value          = p.stops;
    durationInput.value       = p.duration_hours;
    daysLeftInput.value       = p.days_left;

    // New fields (safe-set so old browsers still work)
    if (dayTypeInput     && p.day_type)     dayTypeInput.value     = p.day_type;
    if (seasonInput      && p.season)       seasonInput.value      = p.season;
    if (bookingTypeInput && p.booking_type) bookingTypeInput.value = p.booking_type;

    checkCityDistinctness();
    submitPrediction();
  }

  // ── Event listeners ──────────────────────────────────────────────────────────
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    submitPrediction();
  });

  resetBtn.addEventListener("click", () => {
    form.reset();
    clearError();
    boardingPassView.style.display = "none";
    placeholderView.style.display  = "block";
    currentAnimatedFare = 0;
    checkCityDistinctness();
  });

  document.querySelectorAll(".btn-preset").forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.getAttribute("data-preset");
      applyPreset(key);
    });
  });

  // Run an initial prediction with the form's default values on page load
  checkCityDistinctness();
  submitPrediction();
});
