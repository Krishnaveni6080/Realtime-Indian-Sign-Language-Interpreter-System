document.addEventListener('DOMContentLoaded', () => {
    const currentSignEl = document.getElementById('current-sign');
    const confidenceFillEl = document.getElementById('confidence-fill');
    const confidenceTextEl = document.getElementById('confidence-text');
    const sentenceHistoryEl = document.getElementById('sentence-history');
    
    // UI Elements for Toggle
    const toggleBtn = document.getElementById('camera-toggle-btn');
    const toggleBtnText = document.getElementById('toggle-btn-text');
    const toggleBtnIcon = toggleBtn.querySelector('i');
    
    const videoStreamEl = document.getElementById('video-stream');
    const cameraOfflineEl = document.getElementById('camera-offline');
    const recordingOverlayEl = document.getElementById('recording-overlay');
    const videoWrapperEl = document.getElementById('video-wrapper');
    const headerPulseEl = document.getElementById('header-pulse');

    let cameraActive = false;

    // Toggle Camera API call
    toggleBtn.addEventListener('click', async () => {
        const action = cameraActive ? 'stop' : 'start';
        
        try {
            const response = await fetch('/toggle_camera', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action })
            });
            const result = await response.json();
            
            if (result.status === 'running') {
                cameraActive = true;
                // Switch button to Stop
                toggleBtn.className = 'glow-btn stop-mode';
                toggleBtnText.innerText = 'Stop Camera';
                toggleBtnIcon.className = 'fa-solid fa-stop';
                
                // Show Video
                videoStreamEl.src = "/video_feed?" + new Date().getTime(); // cache bust
                videoStreamEl.style.display = 'block';
                cameraOfflineEl.style.display = 'none';
                recordingOverlayEl.style.display = 'flex';
                videoWrapperEl.classList.add('active');
                headerPulseEl.classList.add('active');
                
            } else if (result.status === 'stopped') {
                cameraActive = false;
                // Switch button to Start
                toggleBtn.className = 'glow-btn start-mode';
                toggleBtnText.innerText = 'Start Camera';
                toggleBtnIcon.className = 'fa-solid fa-camera';
                
                // Show Offline Placeholder
                videoStreamEl.src = "";
                videoStreamEl.style.display = 'none';
                cameraOfflineEl.style.display = 'flex';
                recordingOverlayEl.style.display = 'none';
                videoWrapperEl.classList.remove('active');
                headerPulseEl.classList.remove('active');

                // Reset Stats
                currentSignEl.innerText = "---";
                currentSignEl.className = 'stat-value';
                confidenceFillEl.style.width = `0%`;
                confidenceTextEl.innerText = `0.0%`;
                confidenceFillEl.style.background = 'linear-gradient(90deg, #94a3b8, #94a3b8)';
                confidenceFillEl.style.boxShadow = 'none';
            }
        } catch (error) {
            console.error('Error toggling camera:', error);
            alert("Error toggling camera. Ensure the server is running.");
        }
    });

    // Fetch stats periodically
    setInterval(async () => {
        if (!cameraActive) return; // Don't fetch/update stats if camera is off

        try {
            const response = await fetch('/stats');
            const data = await response.json();
            
            // Re-sync cameraActive state from server just in case
            if (!data.camera_active && cameraActive) {
                // Server camera died remotely
                toggleBtn.click(); // trigger stop locally
                return;
            }

            // Update Confidence
            const confidencePercent = (data.confidence * 100).toFixed(1);
            confidenceFillEl.style.width = `${confidencePercent}%`;
            confidenceTextEl.innerText = `${confidencePercent}%`;
            
            // Adjust confidence color based on threshold (matched to inference1.py)
            const fillStyle = data.confidence > 0.55 
                ? 'linear-gradient(90deg, #10b981, #34d399)' 
                : 'linear-gradient(90deg, #f59e0b, #fbbf24)';
            confidenceFillEl.style.background = fillStyle;
            confidenceFillEl.style.boxShadow = `0 0 15px ${data.confidence > 0.55 ? 'rgba(16, 185, 129, 0.6)' : 'rgba(245, 158, 11, 0.6)'}`;

            // Update Sign
            if (data.sign !== "Waiting...") {
                currentSignEl.innerText = data.sign;
                currentSignEl.className = 'stat-value active';
            } else {
                currentSignEl.innerText = "Waiting...";
                currentSignEl.className = 'stat-value waiting';
            }

            // Update History
            if (data.history.trim() !== '') {
                sentenceHistoryEl.innerText = data.history;
            }

        } catch (error) {
            console.error('Error fetching stats:', error);
        }
    }, 100);
});
