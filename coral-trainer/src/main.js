const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;
const { open } = window.__TAURI__.dialog;

// Elements
const viewSetup = document.getElementById('view-setup');
const viewTraining = document.getElementById('view-training');
const viewExport = document.getElementById('view-export');

const inputProjectName = document.getElementById('project-name');
const inputEpochs = document.getElementById('epochs');
const inputBatchSize = document.getElementById('batch-size');
const inputDataPath = document.getElementById('data-path');
const inputOutputDir = document.getElementById('output-dir');

const btnSelectData = document.getElementById('btn-select-data');
const btnSelectOutput = document.getElementById('btn-select-output');
const btnStart = document.getElementById('btn-start');
const btnOpenFolder = document.getElementById('btn-open-folder');
const btnHome = document.getElementById('btn-home');

const progressBar = document.getElementById('progress-bar');
const statusText = document.getElementById('status-text');
const metricEpoch = document.getElementById('metric-epoch');
const metricLoss = document.getElementById('metric-loss');
const metricAcc = document.getElementById('metric-acc');
const logContainer = document.getElementById('log-container');
const finalModelPath = document.getElementById('final-model-path');

let currentOutputDir = "";

// Navigation
function showView(viewId) {
    viewSetup.classList.add('hidden');
    viewTraining.classList.add('hidden');
    viewExport.classList.add('hidden');
    document.getElementById(viewId).classList.remove('hidden');
}

// Logging
function addLog(message, isError = false) {
    const line = document.createElement('div');
    line.textContent = `> ${message}`;
    if (isError) line.classList.add('text-red-400');
    logContainer.appendChild(line);
    logContainer.scrollTop = logContainer.scrollHeight;
}

// Dialogs
btnSelectData.addEventListener('click', async () => {
    try {
        const selected = await open({
            directory: true,
            multiple: false,
            title: "Select Dataset Folder"
        });
        if (selected) {
            inputDataPath.value = selected;
        }
    } catch (err) {
        console.error(err);
    }
});

btnSelectOutput.addEventListener('click', async () => {
    try {
        const selected = await open({
            directory: true,
            multiple: false,
            title: "Select Output Folder"
        });
        if (selected) {
            inputOutputDir.value = selected;
        }
    } catch (err) {
        console.error(err);
    }
});

// Start Training
btnStart.addEventListener('click', async () => {
    const projectName = inputProjectName.value || "project";
    const epochs = parseInt(inputEpochs.value) || 10;
    const batchSize = parseInt(inputBatchSize.value) || 32;
    const dataPath = inputDataPath.value;
    const outputDir = inputOutputDir.value || "./output";

    if (!dataPath) {
        alert("Please select a data path.");
        return;
    }

    currentOutputDir = outputDir;

    // Reset UI
    logContainer.innerHTML = "";
    progressBar.style.width = "0%";
    metricEpoch.textContent = `0/${epochs}`;
    metricLoss.textContent = "--";
    metricAcc.textContent = "--";
    statusText.textContent = "Initializing...";

    showView('view-training');

    try {
        await invoke('start_training', {
            args: {
                projectName,
                epochs,
                batchSize,
                dataPath,
                outputDir
            }
        });
    } catch (e) {
        alert("Failed to start training: " + e);
        showView('view-setup');
    }
});

// Event Listeners for Rust Backend
listen('training-log', (event) => {
    const raw = event.payload.trim();
    // Try parse JSON
    try {
        const data = JSON.parse(raw);
        if (data.status) {
            if (data.message) addLog(data.message);

            if (data.status === 'training') {
                const percent = (data.epoch / data.total_epochs) * 100;
                progressBar.style.width = `${percent}%`;
                metricEpoch.textContent = `${data.epoch}/${data.total_epochs}`;
                if (data.loss !== undefined) metricLoss.textContent = data.loss.toFixed(4);
                if (data.accuracy !== undefined) metricAcc.textContent = data.accuracy.toFixed(4);
                statusText.textContent = "Training...";
            } else if (data.status === 'complete') {
                progressBar.style.width = "100%";
                statusText.textContent = "Complete!";
                if (data.model_path) {
                    finalModelPath.textContent = data.model_path;
                    setTimeout(() => showView('view-export'), 1000);
                }
            } else if (data.status === 'error') {
                statusText.textContent = "Error!";
                statusText.classList.add('text-red-500');
                addLog(data.message, true);
            }
        }
    } catch (e) {
        // Not JSON, just plain log
        addLog(raw);
    }
});

listen('training-error', (event) => {
    addLog(event.payload, true);
});

listen('training-finished', (event) => {
    const code = event.payload;
    if (code !== 0) {
        statusText.textContent = `Process exited with code ${code}`;
        statusText.classList.add('text-red-500');
    }
});


btnOpenFolder.addEventListener('click', () => {
    if (currentOutputDir) {
        invoke('open_folder', { path: currentOutputDir });
    }
});

btnHome.addEventListener('click', () => {
    showView('view-setup');
});
