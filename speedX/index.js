// ======================== BOWLMETER PRO MAX - FINAL UPGRADE ========================
// Improvements:
// 1. Euclidean distance (full trajectory) for speed calculation
// 2. Multi-point speed using consecutive tracked points + smoothing
// 3. Direction filtering (reject upward/random motion)
// 4. Enhanced tracking stability (adaptive smoothing, velocity clamping)
// 5. AI detection every 6 frames (optimized)
// 6. Frame skipping for performance (process every 2nd frame)
// 7. Auto color calibration safety (only when ball confidence > 0.6)
// 8. Proper OpenCV Mat deletion (memory leak free)
// 9. Fallback handling for AI/CV failures

// ------------------------- DOM Elements -------------------------
const loadingScreen = document.getElementById('loadingScreen');
const appDiv = document.getElementById('app');
const loadingMsg = document.getElementById('loadingMsg');

const video = document.getElementById('videoFeed');
const canvas = document.getElementById('overlayCanvas');
const ctx = canvas.getContext('2d');

const speedDisplay = document.getElementById('speedDisplay');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const calibrateBtn = document.getElementById('calibrateBtn');
const colorCalibrateBtn = document.getElementById('colorCalibrateBtn');
const pitchDistanceInput = document.getElementById('pitchDistance');
const refDistanceInput = document.getElementById('refDistance');
const hueMinSlider = document.getElementById('hueMin');
const hueMaxSlider = document.getElementById('hueMax');
const satMinSlider = document.getElementById('satMin');
const valMinSlider = document.getElementById('valMin');
const trackingStatusDiv = document.getElementById('trackingStatus');
const calibStatusDiv = document.getElementById('calibStatus');

// ------------------------- Global State -------------------------
let isActive = false;
let animationId = null;
let cvReady = false;
let modelReady = false;
let detectionModel = null;

// Calibration
let calibrated = false;
let pixelsPerMeter = null;
let calibrationPoints = [];
let calibrationMode = false;

// Laser lines (Y coordinates)
let releaseLineY = null;
let pitchLineY = null;

// Tracking
let trackedBall = null;        // {x, y, radius, confidence, timestampSec}
let trajectory = [];           // {x, y, t}
let lastValidPos = null;

// Speed & multi-point calculation
let speedHistory = [];
let lastSpeedKph = 0;
let crossingState = 'idle';    // 'idle', 'start', 'end'
let startTimeSec = null;
let endTimeSec = null;

// HSV (wide default for any ball)
let hMin = 0, hMax = 179, sMin = 30, vMin = 80;

// Performance optimization
let frameSkipCounter = 0;
let aiFrameCounter = 0;
const AI_FRAME_INTERVAL = 6;    // Run AI every 6 frames
const PROCESS_EVERY_N_FRAMES = 2; // Process every 2nd frame

// Direction validation
let previousY = null;
let downwardConsistentCount = 0;

// ------------------------- Helper Functions -------------------------
function updateStatus(msg) {
    trackingStatusDiv.innerHTML = msg;
}

function updateCalibStatus(msg) {
    calibStatusDiv.innerHTML = msg;
}

// ------------------------- Loading & Initialization -------------------------
async function initSystem() {
    loadingMsg.innerText = "Loading OpenCV.js...";
    await new Promise(resolve => {
        if (typeof cv !== 'undefined') {
            cv['onRuntimeInitialized'] = () => {
                cvReady = true;
                resolve();
            };
        } else setTimeout(() => resolve(), 500);
    });
    loadingMsg.innerText = "Loading TensorFlow.js & AI Model...";
    try {
        detectionModel = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
        modelReady = true;
    } catch(e) { console.warn("TF failed", e); modelReady = false; }
    loadingMsg.innerText = "Ready! Launching...";
    setTimeout(() => {
        loadingScreen.style.opacity = '0';
        setTimeout(() => {
            loadingScreen.style.display = 'none';
            appDiv.style.display = 'block';
            updateStatus("✅ System Ready | Calibrate first for accuracy");
        }, 500);
    }, 800);
    canvas.width = 640; canvas.height = 480;
    releaseLineY = canvas.height * 0.3;
    pitchLineY = canvas.height * 0.7;
}

// ------------------------- Camera Module -------------------------
async function initCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "environment" }
        });
        video.srcObject = stream;
        await video.play();
        return true;
    } catch(e) {
        updateStatus("❌ Camera permission denied");
        return false;
    }
}

function stopCamera() {
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
        video.srcObject = null;
    }
}

// ------------------------- Detection (OpenCV + TF) -------------------------
function detectBallOpenCV(frameMat) {
    if (!cvReady) return null;
    let hsv = new cv.Mat();
    let mask = new cv.Mat();
    let result = null;
    try {
        cv.cvtColor(frameMat, hsv, cv.COLOR_RGBA2RGB);
        cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);
        let low = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [hMin, sMin, vMin, 0]);
        let high = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [hMax, 255, 255, 255]);
        cv.inRange(hsv, low, high, mask);
        let kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(5,5));
        cv.morphologyEx(mask, mask, cv.MORPH_OPEN, kernel);
        cv.morphologyEx(mask, mask, cv.MORPH_CLOSE, kernel);
        let contours = new cv.MatVector();
        let hierarchy = new cv.Mat();
        cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
        let best = null;
        for (let i = 0; i < contours.size(); i++) {
            let cnt = contours.get(i);
            let area = cv.contourArea(cnt);
            if (area < 80 || area > 15000) continue;
            let peri = cv.arcLength(cnt, true);
            let circularity = 4 * Math.PI * area / (peri * peri);
            if (circularity < 0.55) continue;
            let center = new cv.Point();
            let radius = new cv.Point();
            cv.minEnclosingCircle(cnt, center, radius);
            let rad = radius.x;
            if (rad > 6 && rad < 80) {
                best = { x: center.x, y: center.y, radius: rad, confidence: Math.min(0.95, circularity) };
                break;
            }
        }
        contours.delete(); hierarchy.delete(); kernel.delete(); low.delete(); high.delete();
        result = best;
    } catch(e) { console.warn(e); }
    finally { hsv.delete(); mask.delete(); }
    return result;
}

async function detectAIBall() {
    if (!modelReady || !video.videoWidth) return null;
    try {
        let preds = await detectionModel.detect(video);
        for (let p of preds) {
            if (p.class === 'sports ball' && p.score > 0.5) {
                let [x, y, w, h] = p.bbox;
                return { x: x + w/2, y: y + h/2, radius: (w+h)/4, confidence: p.score };
            }
        }
    } catch(e) {}
    return null;
}

function fuseDetections(opencv, ai) {
    if (opencv && ai && Math.hypot(opencv.x-ai.x, opencv.y-ai.y) < 50) {
        let conf = Math.max(opencv.confidence, ai.confidence);
        let x = (opencv.x*0.6 + ai.x*0.4);
        let y = (opencv.y*0.6 + ai.y*0.4);
        return { x, y, radius: (opencv.radius+ai.radius)/2, confidence: conf };
    }
    return opencv || ai;
}

// ------------------------- Enhanced Tracking with Direction Filtering & Adaptive Smoothing -------------------------
function updateTracking(detection, timestampSec) {
    if (!detection || detection.confidence < 0.45) return false;
    
    // Direction validation: ball must move downward (Y increases)
    if (trackedBall && detection.y < trackedBall.y - 5) {
        // upward movement – likely false positive
        downwardConsistentCount = 0;
        return false;
    }
    if (detection.y > (trackedBall?.y || 0)) downwardConsistentCount++;
    if (downwardConsistentCount < 2 && trackedBall) return false; // require at least 2 consistent downward movements
    
    if (!trackedBall) {
        trackedBall = { ...detection, timestampSec };
        trajectory = [{ x: detection.x, y: detection.y, t: timestampSec }];
        previousY = detection.y;
        return true;
    }
    
    let dt = timestampSec - trackedBall.timestampSec;
    if (dt > 0.01) {
        let dx = detection.x - trackedBall.x;
        let dy = detection.y - trackedBall.y;
        let speedPxPerSec = Math.hypot(dx, dy) / dt;
        // Reject unrealistic jumps (> 1500 px/sec ~ 15 m/s at 100px/m)
        if (speedPxPerSec > 1500) return false;
    }
    
    // Adaptive smoothing: more smoothing when confidence is low
    let alpha = Math.min(0.6, detection.confidence * 0.7);
    trackedBall.x = trackedBall.x * (1-alpha) + detection.x * alpha;
    trackedBall.y = trackedBall.y * (1-alpha) + detection.y * alpha;
    trackedBall.radius = trackedBall.radius * 0.8 + detection.radius * 0.2;
    trackedBall.confidence = detection.confidence;
    trackedBall.timestampSec = timestampSec;
    
    trajectory.push({ x: trackedBall.x, y: trackedBall.y, t: timestampSec });
    if (trajectory.length > 30) trajectory.shift();
    previousY = trackedBall.y;
    return true;
}

function resetTracking() {
    trackedBall = null;
    trajectory = [];
    downwardConsistentCount = 0;
    previousY = null;
}

// ------------------------- Speed Calculation: Multi-point Euclidean Distance -------------------------
function computeMultiPointSpeed(startIdx, endIdx) {
    if (!calibrated || !pixelsPerMeter) return null;
    if (startIdx < 0 || endIdx >= trajectory.length || startIdx >= endIdx) return null;
    
    let totalPixelDistance = 0;
    let totalTime = 0;
    for (let i = startIdx + 1; i <= endIdx; i++) {
        let p1 = trajectory[i-1];
        let p2 = trajectory[i];
        let distPx = Math.hypot(p2.x - p1.x, p2.y - p1.y);
        totalPixelDistance += distPx;
        totalTime += (p2.t - p1.t);
    }
    if (totalTime <= 0.01) return null;
    let realDistanceM = totalPixelDistance / pixelsPerMeter;
    let speedMs = realDistanceM / totalTime;
    let speedKph = speedMs * 3.6;
    if (speedKph < 40 || speedKph > 200) return null;
    return speedKph;
}

function updateSpeedFromBall(ball, currentTimeSec) {
    if (!ball) return;
    if (crossingState === 'idle') {
        if (ball.y >= releaseLineY - 5 && ball.y <= releaseLineY + 15) {
            crossingState = 'start';
            startTimeSec = currentTimeSec;
            updateStatus("⚡ Release detected");
        }
    } else if (crossingState === 'start') {
        if (ball.y >= pitchLineY - 10 && ball.y <= pitchLineY + 20) {
            crossingState = 'end';
            endTimeSec = currentTimeSec;
            
            // Find indices in trajectory around start and end times
            let startIdx = -1, endIdx = -1;
            for (let i = 0; i < trajectory.length; i++) {
                if (startIdx === -1 && trajectory[i].t >= startTimeSec - 0.03) startIdx = i;
                if (trajectory[i].t >= endTimeSec - 0.03) { endIdx = i; break; }
            }
            if (startIdx !== -1 && endIdx !== -1 && endIdx > startIdx) {
                let speed = computeMultiPointSpeed(startIdx, endIdx);
                if (speed !== null) {
                    speedHistory.push(speed);
                    if (speedHistory.length > 5) speedHistory.shift();
                    let avgSpeed = speedHistory.reduce((a,b)=>a+b,0)/speedHistory.length;
                    lastSpeedKph = avgSpeed;
                    speedDisplay.innerText = avgSpeed.toFixed(1);
                    updateStatus(`🎯 ${avgSpeed.toFixed(1)} km/h`);
                } else {
                    updateStatus("⚠️ Speed calculation failed");
                }
            }
            setTimeout(() => { crossingState = 'idle'; }, 500);
        }
    }
    // Timeout reset (2.5 seconds)
    if (crossingState !== 'idle' && currentTimeSec - (startTimeSec||0) > 2.5) {
        crossingState = 'idle';
        updateStatus("🔄 Laser timeout");
    }
}

// ------------------------- Calibration (Pixel to Meter) -------------------------
function startCalibration() {
    calibrationMode = true;
    calibrationPoints = [];
    updateCalibStatus("📏 Click two points on video, then enter real distance");
    canvas.style.pointerEvents = "auto";
    canvas.addEventListener('click', onCalibrationClick);
}

function onCalibrationClick(e) {
    if (!calibrationMode) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    let x = (e.clientX - rect.left) * scaleX;
    let y = (e.clientY - rect.top) * scaleY;
    x = Math.min(Math.max(0, x), canvas.width);
    y = Math.min(Math.max(0, y), canvas.height);
    calibrationPoints.push({ x, y });
    if (calibrationPoints.length === 2) {
        let refDist = parseFloat(refDistanceInput.value);
        if (isNaN(refDist) || refDist <= 0) refDist = 1.0;
        let pixelDist = Math.hypot(calibrationPoints[0].x - calibrationPoints[1].x, calibrationPoints[0].y - calibrationPoints[1].y);
        if (pixelDist > 5) {
            pixelsPerMeter = pixelDist / refDist;
            calibrated = true;
            updateCalibStatus(`✅ Calibrated: ${pixelsPerMeter.toFixed(2)} px/m`);
        } else {
            updateCalibStatus("❌ Points too close, recalibrate");
        }
        calibrationMode = false;
        canvas.removeEventListener('click', onCalibrationClick);
        canvas.style.pointerEvents = "none";
    } else {
        updateCalibStatus(`Point 1 set. Click second point.`);
    }
}

// ------------------------- Safe Auto Color Calibration (only when ball confident) -------------------------
function autoColorCalibration() {
    if (!trackedBall || trackedBall.confidence < 0.6) {
        updateStatus("❌ No stable ball detected. Bowl or hold ball steady.");
        return;
    }
    if (!cvReady || !video.videoWidth) return;
    let src = cv.imread(video);
    let hsv = new cv.Mat();
    cv.cvtColor(src, hsv, cv.COLOR_RGBA2RGB);
    cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);
    let x = Math.floor(trackedBall.x);
    let y = Math.floor(trackedBall.y);
    let radius = Math.floor(trackedBall.radius);
    let roi = hsv.roi(new cv.Rect(Math.max(0, x-radius), Math.max(0, y-radius), radius*2, radius*2));
    let hValues = [], sValues = [];
    for (let i = 0; i < roi.rows; i++) {
        for (let j = 0; j < roi.cols; j++) {
            let h = roi.ucharPtr(i, j)[0];
            let s = roi.ucharPtr(i, j)[1];
            hValues.push(h); sValues.push(s);
        }
    }
    if (hValues.length > 20) {
        hValues.sort((a,b)=>a-b);
        sValues.sort((a,b)=>a-b);
        let hMinAuto = hValues[Math.floor(hValues.length * 0.05)];
        let hMaxAuto = hValues[Math.floor(hValues.length * 0.95)];
        let sMinAuto = sValues[Math.floor(sValues.length * 0.1)];
        hueMinSlider.value = hMinAuto;
        hueMaxSlider.value = hMaxAuto;
        satMinSlider.value = sMinAuto;
        hMin = hMinAuto; hMax = hMaxAuto; sMin = sMinAuto;
        updateStatus(`✅ Auto color: H ${hMinAuto}-${hMaxAuto}, S >${sMinAuto}`);
    } else {
        updateStatus("⚠️ Not enough color data");
    }
    roi.delete(); hsv.delete(); src.delete();
}

// ------------------------- Drawing Overlay -------------------------
function drawOverlay() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.strokeStyle = "#0ff";
    ctx.lineWidth = 3;
    ctx.setLineDash([12, 16]);
    ctx.moveTo(0, releaseLineY); ctx.lineTo(canvas.width, releaseLineY); ctx.stroke();
    ctx.moveTo(0, pitchLineY); ctx.lineTo(canvas.width, pitchLineY); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#0ff";
    ctx.font = "12px monospace";
    ctx.fillText("RELEASE", 10, releaseLineY-5);
    ctx.fillText("PITCH", 10, pitchLineY-5);
    if (trackedBall) {
        ctx.beginPath();
        ctx.arc(trackedBall.x, trackedBall.y, trackedBall.radius, 0, 2*Math.PI);
        ctx.strokeStyle = "#f0f";
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = "#ff00ff88";
        ctx.fill();
        for (let i=1; i<trajectory.length; i++) {
            ctx.beginPath();
            ctx.moveTo(trajectory[i-1].x, trajectory[i-1].y);
            ctx.lineTo(trajectory[i].x, trajectory[i].y);
            ctx.strokeStyle = `rgba(0,255,200,0.6)`;
            ctx.stroke();
        }
    }
}

// ------------------------- Main Processing Loop (Optimized) -------------------------
async function processFrame() {
    if (!isActive || !video.readyState || video.readyState < 2) return;
    if (!cvReady) return;
    
    // Frame skipping for performance
    frameSkipCounter++;
    if (frameSkipCounter % PROCESS_EVERY_N_FRAMES !== 0) {
        drawOverlay();
        return;
    }
    
    let src = null;
    try {
        src = cv.imread(video);
        let resized = new cv.Mat();
        cv.resize(src, resized, new cv.Size(640, 480));
        
        let opencvDet = detectBallOpenCV(resized);
        let finalDet = opencvDet;
        
        aiFrameCounter++;
        if (aiFrameCounter % AI_FRAME_INTERVAL === 0 && modelReady) {
            let aiDet = await detectAIBall();
            if (aiDet) finalDet = fuseDetections(opencvDet, aiDet);
        }
        
        let currentTime = video.currentTime;
        if (finalDet && finalDet.confidence > 0.4) {
            updateTracking(finalDet, currentTime);
        } else if (trackedBall && (currentTime - trackedBall.timestampSec) > 0.25) {
            resetTracking();
        }
        
        if (trackedBall) {
            updateSpeedFromBall(trackedBall, currentTime);
        }
        drawOverlay();
        resized.delete();
    } catch(e) { console.warn(e); }
    finally { if (src) src.delete(); }
}

function animationLoop() {
    if (!isActive) return;
    processFrame().finally(() => {
        animationId = requestAnimationFrame(animationLoop);
    });
}

// ------------------------- Start / Stop -------------------------
async function startSystem() {
    if (isActive) return;
    let camOk = await initCamera();
    if (!camOk) return;
    isActive = true;
    resetTracking();
    crossingState = 'idle';
    speedHistory = [];
    if (animationId) cancelAnimationFrame(animationId);
    animationLoop();
    updateStatus("🟢 Active - Bowl the ball");
}

function stopSystem() {
    isActive = false;
    if (animationId) cancelAnimationFrame(animationId);
    stopCamera();
    resetTracking();
    updateStatus("⏹️ Stopped");
    speedDisplay.innerText = "0.0";
}

// ------------------------- Event Listeners -------------------------
function attachEvents() {
    startBtn.onclick = startSystem;
    stopBtn.onclick = stopSystem;
    calibrateBtn.onclick = startCalibration;
    colorCalibrateBtn.onclick = autoColorCalibration;
    pitchDistanceInput.onchange = () => { updateCalibStatus("📐 Re-calibrate for accuracy"); };
    hueMinSlider.oninput = () => { hMin = parseInt(hueMinSlider.value); updateStatus("HSV updated"); };
    hueMaxSlider.oninput = () => { hMax = parseInt(hueMaxSlider.value); };
    satMinSlider.oninput = () => { sMin = parseInt(satMinSlider.value); };
    valMinSlider.oninput = () => { vMin = parseInt(valMinSlider.value); };
}

// ------------------------- Initialization -------------------------
window.onload = () => {
    attachEvents();
    initSystem();
};
