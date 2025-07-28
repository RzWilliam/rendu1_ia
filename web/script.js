// Configuration du canvas et variables globales
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;
let model = null;

// Configuration du canvas
ctx.lineWidth = 20;
ctx.lineCap = 'round';
ctx.strokeStyle = '#000000';
ctx.fillStyle = '#FFFFFF';
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Chargement du modèle ONNX
async function loadModel() {
    try {
        showStatus('Chargement du modèle...', 'info');
        model = await ort.InferenceSession.create('./mnist_model.onnx');
        showStatus('Modèle chargé avec succès !', 'success');
        document.getElementById('predictBtn').disabled = false;
    } catch (error) {
        console.error('Erreur lors du chargement du modèle:', error);
        showStatus('Erreur: Impossible de charger le modèle IA', 'error');
    }
}

// Affichage des messages de statut
function showStatus(message, type) {
    const statusDiv = document.getElementById('status');
    statusDiv.textContent = message;
    statusDiv.className = `status ${type}`;
    
    if (type === 'info') {
        setTimeout(() => {
            statusDiv.textContent = '';
            statusDiv.className = '';
        }, 3000);
    }
}

// Gestion du dessin sur le canvas
function startDrawing(e) {
    isDrawing = true;
    draw(e);
}

function stopDrawing() {
    isDrawing = false;
    ctx.beginPath();
}

function draw(e) {
    if (!isDrawing) return;
    
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

// Gestion du dessin tactile
function getTouchPos(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    return {
        x: (e.touches[0].clientX - rect.left) * scaleX,
        y: (e.touches[0].clientY - rect.top) * scaleY
    };
}

function startTouchDrawing(e) {
    e.preventDefault();
    isDrawing = true;
    const pos = getTouchPos(e);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
}

function touchDraw(e) {
    e.preventDefault();
    if (!isDrawing) return;
    
    const pos = getTouchPos(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
}

function stopTouchDrawing(e) {
    e.preventDefault();
    isDrawing = false;
    ctx.beginPath();
}

// Événements de dessin
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Événements tactiles
canvas.addEventListener('touchstart', startTouchDrawing);
canvas.addEventListener('touchmove', touchDraw);
canvas.addEventListener('touchend', stopTouchDrawing);

// Effacement du canvas
function clearCanvas() {
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById('prediction').textContent = 'Dessinez un chiffre';
    clearConfidenceBars();
}

// Préparation de l'image pour la prédiction
function preprocessImage() {
    // Redimensionner à 28x28 pixels
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    
    // Dessiner l'image redimensionnée
    tempCtx.fillStyle = '#FFFFFF';
    tempCtx.fillRect(0, 0, 28, 28);
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    
    // Obtenir les données de pixels
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const data = imageData.data;
    
    // Convertir en niveaux de gris et normaliser
    const input = new Float32Array(1 * 1 * 28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
        const pixelIndex = i * 4;
        // Conversion en niveau de gris (inverse car le modèle attend du blanc sur noir)
        const gray = 1.0 - (data[pixelIndex] + data[pixelIndex + 1] + data[pixelIndex + 2]) / (3 * 255);
        // Normalisation MNIST
        input[i] = (gray - 0.1307) / 0.3081;
    }
    
    return input;
}

// Prédiction avec le modèle
async function predict() {
    if (!model) {
        showStatus('Modèle non chargé!', 'error');
        return;
    }
    
    try {
        // Afficher le chargement
        document.getElementById('loading').style.display = 'block';
        document.getElementById('prediction').textContent = '';
        
        // Préprocesser l'image
        const inputData = preprocessImage();
        const inputTensor = new ort.Tensor('float32', inputData, [1, 1, 28, 28]);
        
        // Faire la prédiction
        const feeds = { input: inputTensor };
        const results = await model.run(feeds);
        const output = results.output.data;
        
        // Trouver la classe prédite
        const predicted = output.indexOf(Math.max(...output));
        
        // Calculer les probabilités (softmax)
        const maxLogit = Math.max(...output);
        const exps = output.map(x => Math.exp(x - maxLogit));
        const sumExps = exps.reduce((a, b) => a + b, 0);
        const probabilities = exps.map(x => x / sumExps);
        
        // Afficher le résultat
        document.getElementById('prediction').textContent = predicted.toString();
        updateConfidenceBars(probabilities);
        
        const confidence = (probabilities[predicted] * 100).toFixed(1);
        showStatus(`Prédiction: ${predicted} (confiance: ${confidence}%)`, 'success');
        
    } catch (error) {
        console.error('Erreur lors de la prédiction:', error);
        showStatus('Erreur lors de la prédiction', 'error');
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

// Mise à jour des barres de confiance
function updateConfidenceBars(probabilities) {
    const container = document.getElementById('confidenceBars');
    container.innerHTML = '';
    
    for (let i = 0; i < 10; i++) {
        const confidence = probabilities[i] * 100;
        
        const item = document.createElement('div');
        item.className = 'confidence-item';
        
        item.innerHTML = `
            <div class="digit-label">${i}:</div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${confidence}%"></div>
            </div>
            <div class="confidence-value">${confidence.toFixed(1)}%</div>
        `;
        
        container.appendChild(item);
    }
}

// Effacer les barres de confiance
function clearConfidenceBars() {
    const container = document.getElementById('confidenceBars');
    container.innerHTML = '';
}

// Initialisation
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('predictBtn').disabled = true;
    loadModel();
});

// Gestion du redimensionnement
window.addEventListener('resize', function() {
    // Redessiner si nécessaire
});