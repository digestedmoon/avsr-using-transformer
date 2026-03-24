/**
 * Deep AVSR — Frontend Logic
 * Handles file upload, drag-and-drop, fetch to /transcribe, and UI state.
 */

(function () {
    "use strict";

    // --- DOM Elements ---
    const uploadZone = document.getElementById("upload-zone");
    const fileInput = document.getElementById("file-input");
    const filePreview = document.getElementById("file-preview");
    const fileName = document.getElementById("file-name");
    const fileSize = document.getElementById("file-size");
    const fileRemove = document.getElementById("file-remove");
    const videoContainer = document.getElementById("video-container");
    const videoPlayer = document.getElementById("video-player");
    const transcribeBtn = document.getElementById("transcribe-btn");
    const uploadCard = document.getElementById("upload-card");
    const processingCard = document.getElementById("processing-card");
    const resultCard = document.getElementById("result-card");
    const errorCard = document.getElementById("error-card");
    const resultText = document.getElementById("result-text");
    const resultMode = document.getElementById("result-mode");
    const errorText = document.getElementById("error-text");
    const newUploadBtn = document.getElementById("new-upload-btn");
    const errorRetryBtn = document.getElementById("error-retry-btn");
    const badgeStatus = document.getElementById("badge-status");
    const badgeMode = document.getElementById("badge-mode");
    const processingText = document.getElementById("processing-text");

    const steps = [
        document.getElementById("step-1"),
        document.getElementById("step-2"),
        document.getElementById("step-3"),
        document.getElementById("step-4"),
    ];

    let selectedFile = null;

    // --- Health Check ---
    async function checkHealth() {
        try {
            const res = await fetch("/health");
            const data = await res.json();
            if (data.status === "ok" && data.models_loaded) {
                badgeStatus.classList.add("badge--online");
                badgeStatus.innerHTML = '<span class="badge__dot"></span>Online';
                badgeMode.textContent = data.mode + " Mode";
            } else {
                badgeStatus.innerHTML = '<span class="badge__dot"></span>Models loading...';
            }
        } catch {
            badgeStatus.innerHTML = '<span class="badge__dot"></span>Offline';
        }
    }

    // --- File Selection ---
    function handleFile(file) {
        if (!file) return;
        if (!file.name.toLowerCase().endsWith(".mp4")) {
            showError("Please select a .mp4 video file.");
            return;
        }

        selectedFile = file;

        // Show file info
        fileName.textContent = file.name;
        fileSize.textContent = formatBytes(file.size);
        filePreview.style.display = "flex";
        uploadZone.style.display = "none";

        // Show video preview
        const url = URL.createObjectURL(file);
        videoPlayer.src = url;
        videoContainer.style.display = "block";

        // Enable button
        transcribeBtn.disabled = false;
    }

    function clearFile() {
        selectedFile = null;
        fileInput.value = "";
        filePreview.style.display = "none";
        videoContainer.style.display = "none";
        videoPlayer.src = "";
        uploadZone.style.display = "block";
        transcribeBtn.disabled = true;
    }

    function formatBytes(bytes) {
        if (bytes === 0) return "0 B";
        const k = 1024;
        const sizes = ["B", "KB", "MB", "GB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
    }

    // --- UI State Management ---
    function showCard(card) {
        [uploadCard, processingCard, resultCard, errorCard].forEach((c) => {
            c.style.display = "none";
        });
        card.style.display = "block";
        // Re-trigger animation
        card.style.animation = "none";
        card.offsetHeight; // force reflow
        card.style.animation = "";
    }

    function showError(msg) {
        errorText.textContent = msg;
        showCard(errorCard);
    }

    // --- Processing Step Animation ---
    function animateSteps() {
        let current = 0;
        const messages = [
            "Detecting face & cropping video...",
            "Extracting audio & visual features...",
            "Running neural network inference...",
            "Decoding transcription...",
        ];

        // Reset all steps
        steps.forEach((s) => {
            s.className = "step";
        });
        steps[0].classList.add("step--active");
        processingText.textContent = messages[0];

        const interval = setInterval(() => {
            if (current < steps.length) {
                steps[current].classList.remove("step--active");
                steps[current].classList.add("step--done");
            }
            current++;
            if (current < steps.length) {
                steps[current].classList.add("step--active");
                processingText.textContent = messages[current];
            } else {
                clearInterval(interval);
            }
        }, 2500);

        return interval;
    }

    // --- Transcribe ---
    async function transcribe() {
        if (!selectedFile) return;

        showCard(processingCard);
        const stepInterval = animateSteps();

        try {
            const formData = new FormData();
            formData.append("file", selectedFile);

            const res = await fetch("/transcribe", {
                method: "POST",
                body: formData,
            });

            clearInterval(stepInterval);
            const data = await res.json();

            if (!res.ok) {
                showError(data.error || "Server error occurred.");
                return;
            }

            // Show result
            resultText.textContent = data.transcription || "(empty transcription)";
            resultMode.textContent = "Mode: " + (data.mode || "AV");
            showCard(resultCard);
        } catch (err) {
            clearInterval(stepInterval);
            showError("Network error: " + err.message);
        }
    }

    // --- Event Listeners ---

    // Click to upload
    uploadZone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

    // Drag and drop
    uploadZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadZone.classList.add("dragover");
    });

    uploadZone.addEventListener("dragleave", () => {
        uploadZone.classList.remove("dragover");
    });

    uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadZone.classList.remove("dragover");
        const file = e.dataTransfer.files[0];
        handleFile(file);
    });

    // Remove file
    fileRemove.addEventListener("click", clearFile);

    // Transcribe
    transcribeBtn.addEventListener("click", transcribe);

    // New upload / retry
    newUploadBtn.addEventListener("click", () => {
        clearFile();
        showCard(uploadCard);
    });

    errorRetryBtn.addEventListener("click", () => {
        clearFile();
        showCard(uploadCard);
    });

    // --- Init ---
    checkHealth();
    // Re-check health periodically
    setInterval(checkHealth, 15000);
})();
