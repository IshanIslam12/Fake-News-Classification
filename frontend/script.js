// script.js

// Backend API base URL from Render
const API_BASE = "https://fake-news-classification-1.onrender.com";
const API_URL = `${API_BASE}/predict`;

const titleInput = document.getElementById("title");
const textInput = document.getElementById("text");
const predictBtn = document.getElementById("predictBtn");
const resultBox = document.getElementById("result");

async function predictFakeNews() {
  const title = titleInput.value.trim();
  const text = textInput.value.trim();

  if (!title && !text) {
    resultBox.style.display = "block";
    resultBox.innerHTML = "Please enter a title or some text.";
    return;
  }

  predictBtn.disabled = true;
  predictBtn.textContent = "Predicting...";
  resultBox.style.display = "block";
  resultBox.innerHTML = "Predicting...";

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, text })
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();

    const label = data.label;
    const confidence = data.confidence
      ? (data.confidence * 100).toFixed(2)
      : "N/A";

    resultBox.innerHTML = `
      <strong>Prediction:</strong> ${label}<br>
      <strong>Confidence:</strong> ${confidence}%
    `;
  } catch (err) {
    resultBox.innerHTML = `
      <strong>Error:</strong> ${err.message}<br>
      <span style="font-size: 0.9rem; opacity: 0.8;">
        Check that the backend API is reachable.
      </span>
    `;
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Predict";
  }
}

predictBtn.addEventListener("click", predictFakeNews);
