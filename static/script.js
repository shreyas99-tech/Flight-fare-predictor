/**
 * ==============================================================================
 * SkyFare AI - Flight Fare Predictor Frontend Script
 * Handles route swapping, instant preset selection, source-destination validation,
 * async fetch communication with Flask, and animated INR fare counter display.
 * ==============================================================================
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const form = document.getElementById("fare-form");
  const submitBtn = document.getElementById("btn-submit");
  const resetBtn = document.getElementById("btn-reset");
  const swapBtn = document.getElementById("btn-swap-cities");

  const cityAlert = document.getElementById("city-match-alert");
  const placeholderView = document.getElementById("result-placeholder");
  const boardingPassView = document.getElementById("boarding-pass");
  const errorBanner = document.getElementById("error-banner");
  const errorMsg = document.getElementById("error-msg");

  // Inputs
  const sourceCityInput = document.getElementById("source_city");
  const destCityInput = document.getElementById("destination_city");
  const airlineInput = document.getElementById("airline");
  const departureTimeInput = document.getElementById("departure_time");
  const cabinClassInput = document.getElementById("cabin_class");
  const stopsInput = document.getElementById("stops");
  const durationInput = document.getElementById("duration_hours");
  const daysLeftInput = document.getElementById("days_left");

  // Output Elements on Boarding Pass
  const resAirline = document.getElementById("res-airline");
  const resClass = document.getElementById("res-class");
  const resOrigin = document.getElementById("res-origin");
  const resDestination = document.getElementById("res-destination");
  const resStops = document.getElementById("res-stops");
  const resDuration = document.getElementById("res-duration");
  const resFare = document.getElementById("res-fare");
  const resRange = document.getElementById("res-range");
  const resTime = document.getElementById("res-time");
  const resDays = document.getElementById("res-days");
  const resHourly = document.getElementById("res-hourly");
  const resDemand = document.getElementById("res-demand");
  const resAdvice = document.getElementById("res-advice");

  // Demo Scenarios
  const presets = {
    del_bom_business: {
      source_city: "Delhi",
      destination_city: "Mumbai",
      airline: "Vistara",
      departure_time: "Morning",
      cabin_class: "Business",
      stops: "0",
      duration_hours: 2.2,
      days_left: 8,
    },
    blr_del_lastminute: {
      source_city: "Bangalore",
      destination_city: "Delhi",
      airline: "IndiGo",
      departure_time: "Morning",
      cabin_class: "Economy",
      stops: "0",
      duration_hours: 2.8,
      days_left: 2,
    },
    bom_maa_saver: {
      source_city: "Mumbai",
      destination_city: "Chennai",
      airline: "SpiceJet",
      departure_time: "Afternoon",
      cabin_class: "Economy",
      stops: "0",
      duration_hours: 2.0,
      days_left: 32,
    },
    hyd_ccu_direct: {
      source_city: "Hyderabad",
      destination_city: "Kolkata",
      airline: "Air India",
      departure_time: "Evening",
      cabin_class: "Economy",
      stops: "0",
      duration_hours: 2.0,
      days_left: 18,
    },
  };

  let currentAnimatedFare = 0;

  /**
   * Smoothly animates INR currency from startVal to endVal
   */
  function animateFare(element, startVal, endVal, durationMs = 600) {
    const startTime = performance.now();

    function update(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / durationMs, 1);
      // Ease out cubic
      const easeOut = 1 - Math.pow(1 - progress, 3);
      const current = Math.floor(startVal + (endVal - startVal) * easeOut);

      element.textContent = `₹${current.toLocaleString("en-IN")}`;

      if (progress < 1) {
        requestAnimationFrame(update);
      } else {
        element.textContent = `₹${Math.round(endVal).toLocaleString("en-IN")}`;
      }
    }

    requestAnimationFrame(update);
  }

  /**
   * Validates if source and destination are different
   */
  function checkCityDistinctness() {
    const isSame = sourceCityInput.value.toLowerCase() === destCityInput.value.toLowerCase();
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

  // City change listeners
  sourceCityInput.addEventListener("change", checkCityDistinctness);
  destCityInput.addEventListener("change", checkCityDistinctness);

  // Swap button
  swapBtn.addEventListener("click", () => {
    const temp = sourceCityInput.value;
    sourceCityInput.value = destCityInput.value;
    destCityInput.value = temp;
    checkCityDistinctness();
  });

  /**
   * Reads and packages form values
   */
  function getPayload() {
    return {
      source_city: sourceCityInput.value,
      destination_city: destCityInput.value,
      airline: airlineInput.value,
      departure_time: departureTimeInput.value,
      cabin_class: cabinClassInput.value,
      stops: parseInt(stopsInput.value, 10),
      duration_hours: parseFloat(durationInput.value),
      days_left: parseInt(daysLeftInput.value, 10),
    };
  }

  function showError(message) {
    errorBanner.style.display = "flex";
    errorMsg.textContent = message;
    boardingPassView.style.display = "none";
    placeholderView.style.display = "none";
  }

  function clearError() {
    errorBanner.style.display = "none";
    errorMsg.textContent = "";
  }

  /**
   * Sends flight search parameters to Flask backend
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
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok || data.status !== "success") {
        throw new Error(data.message || "Failed to estimate flight fare.");
      }

      const pred = data.prediction;
      const inp = data.inputs;

      // Update boarding pass UI
      placeholderView.style.display = "none";
      boardingPassView.style.display = "block";

      resAirline.textContent = inp.airline;
      resClass.textContent = inp.cabin_class;
      resOrigin.textContent = inp.source_city;
      resDestination.textContent = inp.destination_city;
      resStops.textContent = pred.stops_label;
      resDuration.textContent = `${inp.duration_hours} hrs`;
      resRange.textContent = `Market Band: ${pred.fare_range}`;
      resTime.textContent = inp.departure_time;
      resDays.textContent = `${inp.days_left} Days Out`;
      resHourly.textContent = pred.fare_per_hour;
      resDemand.textContent = pred.demand_level;
      resAdvice.textContent = pred.booking_advice;

      // Roll up price counter
      animateFare(resFare, currentAnimatedFare, pred.raw_fare, 700);
      currentAnimatedFare = pred.raw_fare;

    } catch (err) {
      showError(err.message || "Connection to flight fare server failed.");
    } finally {
      submitBtn.classList.remove("is-loading");
      submitBtn.disabled = false;
    }
  }

  /**
   * Applies preset scenario to form inputs
   */
  function applyPreset(presetKey) {
    const p = presets[presetKey];
    if (!p) return;

    sourceCityInput.value = p.source_city;
    destCityInput.value = p.destination_city;
    airlineInput.value = p.airline;
    departureTimeInput.value = p.departure_time;
    cabinClassInput.value = p.cabin_class;
    stopsInput.value = p.stops;
    durationInput.value = p.duration_hours;
    daysLeftInput.value = p.days_left;

    checkCityDistinctness();
    submitPrediction();
  }

  // Event handlers
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    submitPrediction();
  });

  resetBtn.addEventListener("click", () => {
    form.reset();
    clearError();
    boardingPassView.style.display = "none";
    placeholderView.style.display = "block";
    currentAnimatedFare = 0;
    checkCityDistinctness();
  });

  document.querySelectorAll(".btn-preset").forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.getAttribute("data-preset");
      applyPreset(key);
    });
  });

  // Run initial prediction with default form values
  checkCityDistinctness();
  submitPrediction();
});
