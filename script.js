// Paramètres de simulation
let DT = 0.1;
let R = 10;
let MU = [0.156, 0.193, 0.342];
let SIGMA = [0.0118, 0.049, 0.0891];
let BS = [[1, 5/12, 2/3], [1/12, 1], [1]];
const KERNEL_SIZE = 2 * R + 1;
let k = 0;
let a = 1;
let interpolationFactor = 0;
let autoMode = false;

// Éléments DOM
const canvas = document.getElementById('leniaCanvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('startBtn');
const pauseBtn = document.getElementById('pauseBtn');
const resetBtn = document.getElementById('resetBtn');
const stepBtn = document.getElementById('stepBtn');
const simulationType = document.getElementById('simulationType');
const statusText = document.getElementById('status');
const frameCount = document.getElementById('frameCount');
const fpsCounter = document.getElementById('fpsCounter');
const gridWidthInput = document.getElementById('gridWidth');
const gridHeightInput = document.getElementById('gridHeight');
const muControls = document.getElementById('muControls');
const sigmaControls = document.getElementById('sigmaControls');
const bsControls = document.getElementById('bsControls');
const addChannelBtn = document.getElementById('addChannel');
const removeChannelBtn = document.getElementById('removeChannel');
const dtInput = document.getElementById('dtInput');

document.getElementById('kSlider').addEventListener('input', (e) => {
	k = parseFloat(e.target.value);
	document.getElementById('kValue').textContent = k.toFixed(2);
	drawGrid();
});

document.getElementById('aSlider').addEventListener('input', (e) => {
	a = parseFloat(e.target.value);
	document.getElementById('aValue').textContent = a.toFixed(2);
	drawGrid();
});

document.getElementById('autoMode').addEventListener('change', (e) => {
	autoMode = e.target.checked;
	if (autoMode) updateAutoRemap();
	drawGrid();
});

document.getElementById('interpolationSlider').addEventListener('input', (e) => {
    interpolationFactor = parseFloat(e.target.value);
    document.getElementById('interpolationValue').textContent = interpolationFactor.toFixed(1);
    drawGrid();
});

gridWidthInput.addEventListener('change', function() {
	GRID_WIDTH = parseInt(this.value);
	initSimulation();
});

gridHeightInput.addEventListener('change', function() {
	GRID_HEIGHT = parseInt(this.value);
	initSimulation();
});

addChannelBtn.addEventListener('click', () => {
	MU.push((Math.random() * 0.9).toFixed(3));
	SIGMA.push((Math.random() * 0.1).toFixed(3));
	BS.push([1]);
	
	createParameterControls();
	initKernels();
	initSimulation();
});

removeChannelBtn.addEventListener('click', () => {
	if (MU.length > 1) {
		MU.pop();
		SIGMA.pop();
		BS.pop();
		
		createParameterControls();
		initKernels();
		initSimulation();
	}
});

dtInput.addEventListener('change', function() {
	DT = parseFloat(this.value) || 0.2;
});

// Variables d'état
let grid;
let nextGrid;
let animationId;
let currentFrame = 0;
let isRunning = false;
let lastTimestamp = 0;
let frameTimes = [];
let GRID_WIDTH = 70;
let GRID_HEIGHT = 70;

function createParameterControls() {
	muControls.innerHTML = '<h4>MU Values:</h4>';
	sigmaControls.innerHTML = '<h4>SIGMA Values:</h4>';
	bsControls.innerHTML = '<h4>BS Values:</h4>';
	
	MU.forEach((value, i) => {
		const control = document.createElement('div');
		control.className = 'channel-control';
		control.innerHTML = `
			<label>Channel ${i}: 
				<input type="number" class="mu-input" data-index="${i}" 
					   step="0.001" value="${value}">
			</label>
		`;
		muControls.appendChild(control);
	});
	
	SIGMA.forEach((value, i) => {
		const control = document.createElement('div');
		control.className = 'channel-control';
		control.innerHTML = `
			<label>Channel ${i}: 
				<input type="number" class="sigma-input" data-index="${i}" 
					   step="0.001" value="${value}">
			</label>
		`;
		sigmaControls.appendChild(control);
	});
	
	BS.forEach((value, i) => {
		const control = document.createElement('div');
		control.className = 'channel-control';
		control.innerHTML = `
			<label>Channel ${i}: 
				<input type="text" class="bs-input" data-index="${i}" 
					   value="${value.join(',')}">
			</label>
		`;
		bsControls.appendChild(control);
	});
	
	document.querySelectorAll('.mu-input').forEach(input => {
		input.addEventListener('change', function() {
			const index = parseInt(this.dataset.index);
			MU[index] = parseFloat(this.value);
			initSimulation();
		});
	});
	
	document.querySelectorAll('.sigma-input').forEach(input => {
		input.addEventListener('change', function() {
			const index = parseInt(this.dataset.index);
			SIGMA[index] = parseFloat(this.value);
			initSimulation();
		});
	});
	
	document.querySelectorAll('.bs-input').forEach(input => {
		input.addEventListener('change', function() {
			const index = parseInt(this.dataset.index);
			const values = this.value.split(',').map(parseFloat);
			BS[index] = values;
			initKernels();
			initSimulation();
		});
	});

	bsControls.innerHTML = '<h4>BS Values:</h4>';
	BS.forEach((channelValues, i) => {
		const control = document.createElement('div');
		control.className = 'channel-control';
		control.innerHTML = `<label>Channel ${i}</label>`;
		
		const channelDiv = document.createElement('div');
		channelDiv.className = 'bs-channel';
		
		channelValues.forEach((value, j) => {
			const fraction = toFraction(value);
			const input = document.createElement('input');
			input.type = 'text';
			input.value = fraction;
			input.className = 'fraction-input';
			input.dataset.channel = i;
			input.dataset.index = j;
			
			channelDiv.appendChild(input);
		});
		
		control.appendChild(channelDiv);
		bsControls.appendChild(control);
	});
	
	document.querySelectorAll('.fraction-input').forEach(input => {
		input.addEventListener('change', function() {
			const channel = parseInt(this.dataset.channel);
			const index = parseInt(this.dataset.index);
			const value = parseFraction(this.value);
			
			BS[channel][index] = value;
			
			this.value = toFraction(value);
			
			initKernels();
			initSimulation();
		});
	});
}

function toFraction(decimal) {
	const tolerance = 1.0e-6;
	const maxDenominator = 100;
	
	if (Math.abs(decimal) < tolerance) return "0";
	if (Math.abs(1 - decimal) < tolerance) return "1";
	
	let numerator = 1;
	let denominator = 1;
	let error = decimal;
	
	for (let d = 1; d <= maxDenominator; d++) {
		const n = Math.round(decimal * d);
		const currentError = Math.abs(decimal - n/d);
		
		if (currentError < error) {
			error = currentError;
			numerator = n;
			denominator = d;
		}
		
		if (currentError < tolerance) break;
	}
	
	const gcd = (a, b) => b ? gcd(b, a % b) : a;
	const divisor = gcd(numerator, denominator);
	
	numerator /= divisor;
	denominator /= divisor;
	
	if (denominator === 1) return numerator.toString();
	return `${numerator}/${denominator}`;
}

function parseFraction(input) {
	if (input.includes('/')) {
		const [num, den] = input.split('/').map(Number);
		if (den !== 0) return num / den;
	}
	return parseFloat(input) || 0;
}

const cmap = [
	[0, 0.5, 0], [68, 1, 84], [71, 44, 122], [59, 81, 139], [44, 113, 142],
	[33, 144, 141], [39, 173, 129], [92, 200, 99], [170, 220, 50],
	[253, 231, 36], [253, 231, 231]
];

function remap(t, k) {
	const atanK2 = Math.atan(k / 2);
	return 0.5 + (Math.atan(k * (t - 0.5)) / atanK2) * 0.5;
}

function remap2(t, k) {
	const d = t - 0.5;
	const a = Math.pow(2, 2 * k);
	return 0.5 + a * Math.pow(Math.abs(d), 2 * k + 1) * Math.sign(d);
}

function remap3(t, k, a) {
	if (k == 0) {
		return t;
	}
	else {
		return a * remap(t, k) + (1 - a) * remap2(t, k);
	}
}

function viridisColor(t) {
	t = Math.min(1, Math.max(0, t));
	const tRemapped = remap3(t, k, a);
	const scaled = tRemapped * (cmap.length - 1);
	const i = Math.floor(scaled);
	const frac = scaled - i;
	const [r1, g1, b1] = cmap[i];
	const [r2, g2, b2] = cmap[Math.min(i + 1, cmap.length - 1)];
	return `rgb(${Math.round(r1 + frac * (r2 - r1))},${Math.round(g1 + frac * (g2 - g1))},${Math.round(b1 + frac * (b2 - b1))})`;
}

function bicubicInterpolation(x, y) {
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const dx = x - x0;
    const dy = y - y0;
    
    const xi0 = Math.max(0, Math.min(GRID_WIDTH - 1, x0));
    const yi0 = Math.max(0, Math.min(GRID_HEIGHT - 1, y0));
    const xi1 = Math.max(0, Math.min(GRID_WIDTH - 1, x0 + 1));
    const yi1 = Math.max(0, Math.min(GRID_HEIGHT - 1, y0 + 1));
    
    const a = grid[yi0][xi0];
    const b = grid[yi0][xi1];
    const c = grid[yi1][xi0];
    const d = grid[yi1][xi1];
    
    const bilinear = a * (1 - dx) * (1 - dy)
                  + b * dx * (1 - dy)
                  + c * (1 - dx) * dy
                  + d * dx * dy;
    
    if (interpolationFactor > 0.5) {
        const nearest = grid[dy < 0.5 ? yi0 : yi1][dx < 0.5 ? xi0 : xi1];
        return bilinear * (1 - interpolationFactor) + nearest * interpolationFactor;
    } else {
		return bilinear;
	}
}

function updateAutoRemap() {
	let totalWeight = 0;
	let weightedSum = 0;
	let weightedSumSq = 0;

	const eps = 1e-4;

	for (let i = 0; i < GRID_HEIGHT; i++) {
		for (let j = 0; j < GRID_WIDTH; j++) {
			const val = grid[i][j];
			if (val < eps) continue;

			const i0 = (i - 1 + GRID_HEIGHT) % GRID_HEIGHT;
			const i1 = (i + 1) % GRID_HEIGHT;
			const j0 = (j - 1 + GRID_WIDTH) % GRID_WIDTH;
			const j1 = (j + 1) % GRID_WIDTH;

			const dx = (grid[i][j1] - grid[i][j0]) * 0.5;
			const dy = (grid[i1][j] - grid[i0][j]) * 0.5;
			const gradient = Math.sqrt(dx * dx + dy * dy);

			const weight = gradient * gradient + eps;

			weightedSum += weight * val;
			weightedSumSq += weight * val * val;
			totalWeight += weight;
		}
	}

	if (totalWeight === 0) return;

	const mean = weightedSum / totalWeight;
	const variance = weightedSumSq / totalWeight - mean * mean;
	const std = Math.sqrt(variance);

	k = Math.max(0.5, Math.min(10, 1.7 / (std + 1e-4)));
	a = Math.max(0.05, Math.min(0.95, 0.5 + (mean - 0.5) * 1.5));
}

function initKernels() {
	convolveKernels = [];
	
	for (let b of BS) {
		let kernel = Array(KERNEL_SIZE).fill().map(() => Array(KERNEL_SIZE).fill(0));
		let total = 0;

		for (let i = 0; i < KERNEL_SIZE; i++) {
			for (let j = 0; j < KERNEL_SIZE; j++) {
				const dx = i - R;
				const dy = j - R;
				const distance = Math.sqrt(dx * dx + dy * dy) / R;

				const ringIndex = Math.floor(distance * b.length);
				if (ringIndex < b.length) {
					const frac = distance * b.length - ringIndex;
					const value = b[ringIndex] * Math.exp(-Math.pow(frac - 0.5, 2) / (2 * Math.pow(0.15, 2)));
					kernel[i][j] = value;
					total += value;
				}
			}
		}

		for (let i = 0; i < KERNEL_SIZE; i++) {
			for (let j = 0; j < KERNEL_SIZE; j++) {
				kernel[i][j] /= total;
			}
		}
		
		convolveKernels.push(kernel);
	}
}

function gauss(x, mu, sigma) {
	return Math.exp(-Math.pow((x - mu) / sigma, 2) / 2);
}

function growth(x, mu, sigma) {
	return 2 * gauss(x, mu, sigma) - 1;
}

const fish = [
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.06, 0.1, 0.04, 0.02, 0.01, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0.15, 0.37, 0.5, 0.44, 0.19, 0.23, 0.3, 0.23, 0.15, 0.01, 0, 0, 0, 0], 
	[0, 0, 0, 0, 0, 0, 0.32, 0.78, 0.26, 0, 0.11, 0.11, 0.1, 0.08, 0.18, 0.16, 0.17, 0.24, 0.09, 0, 0, 0], 
	[0, 0, 0, 0, 0.45, 0.16, 0, 0, 0, 0, 0, 0.15, 0.15, 0.16, 0.15, 0.1, 0.09, 0.21, 0.24, 0.12, 0, 0], 
	[0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0.17, 0.39, 0.43, 0.34, 0.25, 0.15, 0.16, 0.15, 0.25, 0.03, 0], 
	[0, 0.15, 0.06, 0, 0, 0, 0, 0, 0, 0, 0.24, 0.72, 0.92, 0.85, 0.61, 0.47, 0.39, 0.27, 0.12, 0.18, 0.17, 0], 
	[0, 0.08, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0.73, 0.6, 0.56, 0.31, 0.12, 0.15, 0.24, 0.01], 
	[0, 0.16, 0, 0, 0, 0, 0, 0, 0, 0.76, 1.0, 1.0, 1.0, 1.0, 0.76, 0.72, 0.65, 0.39, 0.1, 0.17, 0.24, 0.05], 
	[0, 0.05, 0, 0, 0, 0, 0, 0, 0.21, 0.83, 1.0, 1.0, 1.0, 1.0, 0.86, 0.85, 0.76, 0.36, 0.17, 0.13, 0.21, 0.07], 
	[0, 0.05, 0, 0, 0.02, 0, 0, 0, 0.4, 0.91, 1.0, 1.0, 1.0, 1.0, 1.0, 0.95, 0.79, 0.36, 0.21, 0.09, 0.18, 0.04], 
	[0.06, 0.08, 0, 0.18, 0.21, 0.1, 0.03, 0.38, 0.92, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.64, 0.31, 0.12, 0.07, 0.25, 0], 
	[0.05, 0.12, 0.27, 0.4, 0.34, 0.42, 0.93, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.97, 0.33, 0.16, 0.05, 0.1, 0.26, 0], 
	[0, 0.25, 0.21, 0.39, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.86, 0.89, 0.94, 0.83, 0.13, 0, 0, 0.04, 0.21, 0.18, 0], 
	[0, 0.06, 0.29, 0.63, 0.84, 0.97, 1.0, 1.0, 1.0, 0.96, 0.46, 0.33, 0.36, 0, 0, 0, 0, 0, 0.03, 0.35, 0, 0], 
	[0, 0, 0.13, 0.22, 0.59, 0.85, 0.99, 1.0, 0.98, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0.34, 0.14, 0, 0], 
	[0, 0, 0, 0, 0.33, 0.7, 0.95, 0.8, 0.33, 0.11, 0, 0, 0, 0, 0, 0, 0, 0.11, 0.26, 0, 0, 0], 
	[0, 0, 0, 0, 0.16, 0.56, 0.52, 0.51, 0.4, 0.18, 0.01, 0, 0, 0, 0, 0, 0, 0.42, 0, 0, 0, 0], 
	[0, 0, 0, 0, 0.01, 0, 0.33, 0.47, 0.33, 0.05, 0, 0, 0, 0, 0, 0, 0.35, 0, 0, 0, 0, 0], 
	[0, 0, 0, 0, 0, 0.26, 0.32, 0.13, 0, 0, 0, 0, 0, 0, 0, 0.34, 0, 0, 0, 0, 0, 0], 
	[0, 0, 0, 0, 0, 0.22, 0.25, 0.03, 0, 0, 0, 0, 0, 0, 0.46, 0, 0, 0, 0, 0, 0, 0], 
	[0, 0, 0, 0, 0, 0, 0.09, 0.2, 0.22, 0.23, 0.23, 0.22, 0.3, 0.3, 0, 0, 0, 0, 0, 0, 0, 0]
];

function initFishGrid() {
	const grid = Array(GRID_HEIGHT).fill().map(() => Array(GRID_WIDTH).fill(0));
	
	const centerX = Math.floor(GRID_HEIGHT/2 - fish.length/2);
	const centerY = Math.floor(GRID_WIDTH/2 - fish[0].length/2);
	
	const startX = Math.max(0, centerX);
	const startY = Math.max(0, centerY);
	const endX = Math.min(GRID_HEIGHT, startX + fish.length);
	const endY = Math.min(GRID_WIDTH, startY + fish[0].length);
	
	for (let i = startX; i < endX; i++) {
		for (let j = startY; j < endY; j++) {
			const fishI = i - startX;
			const fishJ = j - startY;
			grid[i][j] = fish[fishI][fishJ];
		}
	}
	
	return grid;
}

function initMultiFishGrid() {
    const grid = Array(GRID_HEIGHT).fill().map(() => Array(GRID_WIDTH).fill(0));
    
    const fishHeight = fish.length;
    const fishWidth = fish[0].length;
    const buffer = 3;
    
    const placedFish = [];
    
    for (let n = 0; n < 3; n++) {
        let validPosition = false;
        let attempts = 0;
        let startX, startY;
        
        while (!validPosition && attempts < 100) {
            attempts++;
            startX = Math.floor(Math.random() * (GRID_HEIGHT - fishHeight - 2 * buffer)) + buffer;
            startY = Math.floor(Math.random() * (GRID_WIDTH - fishWidth - 2 * buffer)) + buffer;
            
            validPosition = true;
            
            for (const fish of placedFish) {
                const horizontalOverlap = 
                    (startX < fish.x + fish.height + buffer) && 
                    (startX + fishHeight + buffer > fish.x);
                
                const verticalOverlap = 
                    (startY < fish.y + fish.width + buffer) && 
                    (startY + fishWidth + buffer > fish.y);
                
                if (horizontalOverlap && verticalOverlap) {
                    validPosition = false;
                    break;
                }
            }
        }
        
        if (validPosition) {
            const flipHorizontal = Math.random() > 0.5;
            const flipVertical = Math.random() > 0.5;
            
            for (let i = 0; i < fishHeight; i++) {
                for (let j = 0; j < fishWidth; j++) {
                    const patternI = flipVertical ? fishHeight - 1 - i : i;
                    const patternJ = flipHorizontal ? fishWidth - 1 - j : j;
                    
                    const x = (startX + i) % GRID_HEIGHT;
                    const y = (startY + j) % GRID_WIDTH;
                    grid[x][y] = fish[patternI][patternJ];
                }
            }
            
            placedFish.push({
                x: startX,
                y: startY,
                height: fishHeight,
                width: fishWidth,
                flipH: flipHorizontal,
                flipV: flipVertical
            });
        }
    }
    
    return grid;
}

function initRandomGrid() {
	return Array(GRID_HEIGHT).fill().map(() =>
		Array(GRID_WIDTH).fill().map(() => Math.random()));
}

function initOrbiumGrid() {
    const grid = Array(GRID_HEIGHT).fill().map(() => Array(GRID_WIDTH).fill(0));
    const centerX = Math.floor(GRID_HEIGHT/2);
    const centerY = Math.floor(GRID_WIDTH/2);
    
    for (let i = -10; i <= 10; i++) {
        for (let j = -10; j <= 10; j++) {
            const distance = Math.sqrt(i*i + j*j) / 10;
            if (distance <= 1) {
                const value = Math.exp(-30 * Math.pow(distance - 0.7, 2));
                const x = (centerX + i + GRID_HEIGHT) % GRID_HEIGHT;
                const y = (centerY + j + GRID_WIDTH) % GRID_WIDTH;
                grid[x][y] = value;
            }
        }
    }
    
    return grid;
}

function initChromiumGrid() {
    const grid = Array(GRID_HEIGHT).fill().map(() => Array(GRID_WIDTH).fill(0));
    const centerX = Math.floor(GRID_HEIGHT/2);
    const centerY = Math.floor(GRID_WIDTH/2);
    const val = Math.floor(Math.min(GRID_HEIGHT, GRID_HEIGHT) / 2);
    for (let i = -val; i <= val; i++) {
        for (let j = -val; j <= val; j++) {
            const angle = Math.atan2(j, i);
            const distance = Math.sqrt(i*i + j*j) / val;
            const wave = 0.5 * (1 + Math.cos(5 * angle));
            
            if (distance <= 1) {
                const value = wave * Math.exp(-20 * Math.pow(distance - 0.6, 2));
                const x = (centerX + i + GRID_HEIGHT) % GRID_HEIGHT;
                const y = (centerY + j + GRID_WIDTH) % GRID_WIDTH;
                grid[x][y] = Math.min(1, value);
            }
        }
    }
    
    return grid;
}

function initCyanorGrid() {
    const grid = Array(GRID_HEIGHT).fill().map(() => Array(GRID_WIDTH).fill(0));
    const centerX = Math.floor(GRID_HEIGHT/2);
    const centerY = Math.floor(GRID_WIDTH/2);
    
    for (let i = -12; i <= 12; i++) {
        for (let j = -12; j <= 12; j++) {
            const distance = Math.sqrt(i*i + j*j) / 12;
            if (distance > 0.3 && distance < 0.9) {
                const value = 0.8 * Math.exp(-15 * Math.pow(distance - 0.6, 2));
                const x = (centerX + i + GRID_HEIGHT) % GRID_HEIGHT;
                const y = (centerY + j + GRID_WIDTH) % GRID_WIDTH;
                grid[x][y] = value;
            }
        }
    }
    
    return grid;
}

function initSimulation() {
	if (isRunning) {
		cancelAnimationFrame(animationId);
		isRunning = false;
	}
	
	currentFrame = 0;
	frameCount.textContent = currentFrame;
	
    const type = simulationType.value;
    switch(type) {
        case "fish":
            grid = initFishGrid();
            break;
        case "multiFish":
            grid = initMultiFishGrid();
            break;
        case "orbium":
            grid = initOrbiumGrid();
            break;
        case "chromium":
            grid = initChromiumGrid();
            break;
        case "cyanor":
            grid = initCyanorGrid();
            break;
        default:
            grid = initRandomGrid();
    }
	
	nextGrid = grid.map(row => [...row]);
	
	drawGrid();
	
	statusText.textContent = "Ready";
	startBtn.disabled = false;
	pauseBtn.disabled = true;
	stepBtn.disabled = false;
}

function handleCanvasResize() {
    const container = canvas.parentElement;
    const containerWidth = container.clientWidth - 40;
    const containerHeight = container.clientHeight - 40;
    
    const cellSize = Math.min(
        containerWidth / GRID_WIDTH,
        containerHeight / GRID_HEIGHT
    );
    
    const canvasWidth = Math.floor(cellSize * GRID_WIDTH);
    const canvasHeight = Math.floor(cellSize * GRID_HEIGHT);
    
    if (canvas.width !== canvasWidth || canvas.height !== canvasHeight) {
        canvas.width = canvasWidth;
        canvas.height = canvasHeight;
        return true;
    }
    return false;
}

function textureToArray(texture) {
	const array = [];
	for (let i = 0; i < GRID_HEIGHT; i++) {
		array[i] = [];
		for (let j = 0; j < GRID_WIDTH; j++) {
			array[i][j] = texture[i][j];
		}
	}
	return array;
}

function drawGrid() {
    if (handleCanvasResize()) {
        // Le canvas a été redimensionné
    }
    
    const cellWidth = canvas.width / GRID_WIDTH;
    const cellHeight = canvas.height / GRID_HEIGHT;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (interpolationFactor > 0) {
        const imageData = ctx.createImageData(canvas.width, canvas.height);
        const data = imageData.data;
        
        for (let py = 0; py < canvas.height; py++) {
            for (let px = 0; px < canvas.width; px++) {
                const gy = py / cellHeight;
                const gx = px / cellWidth;
                
                const value = bicubicInterpolation(gx, gy);
                const color = viridisColor(value);
                const rgb = color.match(/\d+/g);
                const r = parseInt(rgb[0]);
                const g = parseInt(rgb[1]);
                const b = parseInt(rgb[2]);
                
                const index = (py * canvas.width + px) * 4;
                data[index] = r;
                data[index+1] = g;
                data[index+2] = b;
                data[index+3] = 255;
            }
        }
        
        ctx.putImageData(imageData, 0, 0);
    } else {
        const imageData = ctx.createImageData(canvas.width, canvas.height);
        const data = imageData.data;
        
        for (let i = 0; i < GRID_HEIGHT; i++) {
            for (let j = 0; j < GRID_WIDTH; j++) {
                const value = grid[i][j];
                const color = viridisColor(value);
                const rgb = color.match(/\d+/g);
                const r = parseInt(rgb[0]);
                const g = parseInt(rgb[1]);
                const b = parseInt(rgb[2]);
                
                const x = Math.floor(j * cellWidth);
                const y = Math.floor(i * cellHeight);
                const width = Math.ceil(cellWidth);
                const height = Math.ceil(cellHeight);
                
                for (let dy = 0; dy < height; dy++) {
                    for (let dx = 0; dx < width; dx++) {
                        const px = (y + dy) * canvas.width + (x + dx);
                        const index = px * 4;
                        
                        if (index + 3 < data.length) { 
                            data[index] = r;
                            data[index+1] = g;
                            data[index+2] = b;
                            data[index+3] = 255;
                        }
                    }
                }
            }
        }
        
        ctx.putImageData(imageData, 0, 0);
    }
}

function evolve() {
	const potentials = convolveKernels.map(kernel => convolve(grid, kernel));
	
	const growthTerms = [];
	for (let c = 0; c < potentials.length; c++) {
		const term = Array(GRID_HEIGHT).fill().map(() => Array(GRID_WIDTH).fill(0));
		
		for (let i = 0; i < GRID_HEIGHT; i++) {
			for (let j = 0; j < GRID_WIDTH; j++) {
				term[i][j] = growth(potentials[c][i][j], MU[c], SIGMA[c]);
			}
		}
		
		growthTerms.push(term);
	}
	
	const avgGrowth = Array(GRID_HEIGHT).fill().map(() => Array(GRID_WIDTH).fill(0));
	for (let i = 0; i < GRID_HEIGHT; i++) {
		for (let j = 0; j < GRID_WIDTH; j++) {
			let sum = 0;
			for (let c = 0; c < growthTerms.length; c++) {
				sum += growthTerms[c][i][j];
			}
			avgGrowth[i][j] = sum / growthTerms.length;
		}
	}
	
	for (let i = 0; i < GRID_HEIGHT; i++) {
		for (let j = 0; j < GRID_WIDTH; j++) {
			nextGrid[i][j] = Math.max(0, Math.min(1, grid[i][j] + DT * avgGrowth[i][j]));
		}
	}
	
	const temp = grid;
	grid = nextGrid;
	nextGrid = temp;
}

function convolve(grid, kernel) {
	const height = grid.length;
	const width = grid[0].length;
	const kSize = kernel.length;
	const kRadius = Math.floor(kSize / 2);
	
	const result = Array(height).fill().map(() => Array(width).fill(0));
	
	for (let i = 0; i < height; i++) {
		for (let j = 0; j < width; j++) {
			let sum = 0;
			
			for (let ki = 0; ki < kSize; ki++) {
				for (let kj = 0; kj < kSize; kj++) {
					const ii = (i + ki - kRadius + height) % height;
					const jj = (j + kj - kRadius + width) % width;
					
					sum += grid[ii][jj] * kernel[ki][kj];
				}
			}
			
			result[i][j] = sum;
		}
	}
	
	return result;
}

function animate(timestamp) {
	if (!lastTimestamp) lastTimestamp = timestamp;
	const delta = timestamp - lastTimestamp;
	lastTimestamp = timestamp;
	
	frameTimes.push(delta);
	if (frameTimes.length > 10) frameTimes.shift();
	const avgFrameTime = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
	fpsCounter.textContent = Math.round(1000 / avgFrameTime);
	
	evolve();
	if (autoMode) updateAutoRemap();
	drawGrid();
	currentFrame++;
	frameCount.textContent = currentFrame;
	
	if (isRunning) {
		animationId = requestAnimationFrame(animate);
	}
}

startBtn.addEventListener('click', () => {
	isRunning = true;
	statusText.textContent = "Running...";
	startBtn.disabled = true;
	pauseBtn.disabled = false;
	stepBtn.disabled = true;
	lastTimestamp = 0;
	frameTimes = [];
	animationId = requestAnimationFrame(animate);
});

pauseBtn.addEventListener('click', () => {
	isRunning = false;
	statusText.textContent = "En pause";
	startBtn.disabled = false;
	pauseBtn.disabled = true;
	stepBtn.disabled = false;
});

resetBtn.addEventListener('click', initSimulation);

stepBtn.addEventListener('click', () => {
	evolve();
	drawGrid();
	currentFrame++;
	frameCount.textContent = currentFrame;
});

simulationType.addEventListener('change', initSimulation);

window.addEventListener('resize', () => {
    if (handleCanvasResize() && !isRunning) {
        drawGrid();
    }
});

// Initialisation
createParameterControls();
initKernels();
initSimulation();