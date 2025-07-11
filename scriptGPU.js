// Paramètres de simulation
const DT = 0.2;
const R = 10;
const MU = [0.156, 0.193, 0.342];
const SIGMA = [0.0118, 0.049, 0.0891];
const BS = [[1, 5/12, 2/3], [1/12, 1], [1]];
const KERNEL_SIZE = 2 * R + 1;
let k = 0;
let a = 1;
let autoMode = false;
let useGPU = false;

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

// Variables d'état
let grid;
let nextGrid;
let animationId;
let currentFrame = 0;
let isRunning = false;
let lastTimestamp = 0;
let frameTimes = [];
const GRID_WIDTH = 70;
const GRID_HEIGHT = GRID_WIDTH;

// Initialisation du GPU
const gpu = new GPU();
if (!gpu) {
	console.error("GPU.js n'a pas pu s'initialiser. La simulation fonctionnera sur CPU.");
}

// Kernels GPU
let convolveKernels = [];
let growthKernel;
let averageKernel;
let updateKernel;
let drawKernel;

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

function initGPUKernels() {
    try {
		return false;
        // Kernel de convolution
        convolveKernels = BS.map((b, idx) => {
            // Construire le kernel pour cette couche
            let kernelArray = Array(KERNEL_SIZE).fill().map(() => Array(KERNEL_SIZE).fill(0));
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
                        kernelArray[i][j] = value;
                        total += value;
                    }
                }
            }

            // Normaliser le kernel
            for (let i = 0; i < KERNEL_SIZE; i++) {
                for (let j = 0; j < KERNEL_SIZE; j++) {
                    kernelArray[i][j] /= total;
                }
            }

            return gpu.createKernel(function(grid) {
                let sum = 0;
                const kSize = this.constants.kSize;
                const kRadius = this.constants.kRadius;
                const height = this.constants.height;
                const width = this.constants.width;
                
                for (let ki = 0; ki < kSize; ki++) {
                    for (let kj = 0; kj < kSize; kj++) {
                        const ii = (this.thread.y + ki - kRadius + height) % height;
                        const jj = (this.thread.x + kj - kRadius + width) % width;
                        sum += grid[ii][jj] * this.constants.kernel[ki][kj];
                    }
                }
                
                return sum;
            }).setConstants({
                kernel: kernelArray,
                kSize: KERNEL_SIZE,
                kRadius: R,
                height: GRID_HEIGHT,
                width: GRID_WIDTH
            }).setOutput([GRID_WIDTH, GRID_HEIGHT]);
        });
        
        // Kernel de croissance
        growthKernel = gpu.createKernel(function(potential, mu, sigma) {
            const x = potential[this.thread.y][this.thread.x];
            const t = (x - mu) / sigma;
            return 2 * Math.exp(-t * t / 2) - 1;
        }).setOutput([GRID_WIDTH, GRID_HEIGHT]);
        
        // Kernel de moyenne
        averageKernel = gpu.createKernel(function(term1, term2, term3) {
            return (term1[this.thread.y][this.thread.x] + 
                    term2[this.thread.y][this.thread.x] + 
                    term3[this.thread.y][this.thread.x]) / 3;
        }).setOutput([GRID_WIDTH, GRID_HEIGHT]);
        
        // Kernel de mise à jour
        updateKernel = gpu.createKernel(function(grid, avgGrowth) {
            const value = grid[this.thread.y][this.thread.x] + 
                         this.constants.dt * avgGrowth[this.thread.y][this.thread.x];
            return Math.max(0, Math.min(1, value));
        }).setConstants({ dt: DT }).setOutput([GRID_WIDTH, GRID_HEIGHT]);
        
        // Kernel de rendu
        drawKernel = gpu.createKernel(function(grid) {
            const value = grid[this.thread.y][this.thread.x];
            
            // Palette de couleurs Viridis
            if (value < 0.1) this.color(68/255, 1/255, 84/255, 1);
            else if (value < 0.2) this.color(71/255, 44/255, 122/255, 1);
            else if (value < 0.3) this.color(59/255, 81/255, 139/255, 1);
            else if (value < 0.4) this.color(44/255, 113/255, 142/255, 1);
            else if (value < 0.5) this.color(33/255, 144/255, 141/255, 1);
            else if (value < 0.6) this.color(39/255, 173/255, 129/255, 1);
            else if (value < 0.7) this.color(92/255, 200/255, 99/255, 1);
            else if (value < 0.8) this.color(170/255, 220/255, 50/255, 1);
            else this.color(253/255, 231/255, 36/255, 1);
        }).setGraphical(true).setOutput([GRID_WIDTH, GRID_HEIGHT]);

        return true;
    } catch (e) {
        console.error("Erreur d'initialisation GPU:", e);
        return false;
    }
}

function initKernels() {
	// Initialisation des kernels GPU
	const gpuAvailable = initGPUKernels();
	
	if (!gpuAvailable) {
		console.log("Utilisation du CPU comme solution de secours");
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
}

function gauss(x, mu, sigma) {
	return Math.exp(-Math.pow((x - mu) / sigma, 2) / 2);
}

function growth(x, mu, sigma) {
	return 2 * gauss(x, mu, sigma) - 1;
}

function initFishGrid() {
	const grid = Array(GRID_HEIGHT).fill().map(() => Array(GRID_WIDTH).fill(0));
	
	const fish = [
		[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06, 0.1, 0.04, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.37, 0.5, 0.44, 0.19, 0.23, 0.3, 0.23, 0.15, 0.01, 0.0, 0.0, 0.0, 0.0], 
		[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.32, 0.78, 0.26, 0.0, 0.11, 0.11, 0.1, 0.08, 0.18, 0.16, 0.17, 0.24, 0.09, 0.0, 0.0, 0.0], 
		[0.0, 0.0, 0.0, 0.0, 0.45, 0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.15, 0.16, 0.15, 0.1, 0.09, 0.21, 0.24, 0.12, 0.0, 0.0], 
		[0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.17, 0.39, 0.43, 0.34, 0.25, 0.15, 0.16, 0.15, 0.25, 0.03, 0.0], 
		[0.0, 0.15, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.24, 0.72, 0.92, 0.85, 0.61, 0.47, 0.39, 0.27, 0.12, 0.18, 0.17, 0.0], 
		[0.0, 0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.73, 0.6, 0.56, 0.31, 0.12, 0.15, 0.24, 0.01], 
		[0.0, 0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.76, 1.0, 1.0, 1.0, 1.0, 0.76, 0.72, 0.65, 0.39, 0.1, 0.17, 0.24, 0.05], 
		[0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.21, 0.83, 1.0, 1.0, 1.0, 1.0, 0.86, 0.85, 0.76, 0.36, 0.17, 0.13, 0.21, 0.07], 
		[0.0, 0.05, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.4, 0.91, 1.0, 1.0, 1.0, 1.0, 1.0, 0.95, 0.79, 0.36, 0.21, 0.09, 0.18, 0.04], 
		[0.06, 0.08, 0.0, 0.18, 0.21, 0.1, 0.03, 0.38, 0.92, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.64, 0.31, 0.12, 0.07, 0.25, 0.0], 
		[0.05, 0.12, 0.27, 0.4, 0.34, 0.42, 0.93, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.97, 0.33, 0.16, 0.05, 0.1, 0.26, 0.0], 
		[0.0, 0.25, 0.21, 0.39, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.86, 0.89, 0.94, 0.83, 0.13, 0.0, 0.0, 0.04, 0.21, 0.18, 0.0], 
		[0.0, 0.06, 0.29, 0.63, 0.84, 0.97, 1.0, 1.0, 1.0, 0.96, 0.46, 0.33, 0.36, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03, 0.35, 0.0, 0.0], 
		[0.0, 0.0, 0.13, 0.22, 0.59, 0.85, 0.99, 1.0, 0.98, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.34, 0.14, 0.0, 0.0], 
		[0.0, 0.0, 0.0, 0.0, 0.33, 0.7, 0.95, 0.8, 0.33, 0.11, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11, 0.26, 0.0, 0.0, 0.0], 
		[0.0, 0.0, 0.0, 0.0, 0.16, 0.56, 0.52, 0.51, 0.4, 0.18, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.42, 0.0, 0.0, 0.0, 0.0], 
		[0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.33, 0.47, 0.33, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0], 
		[0.0, 0.0, 0.0, 0.0, 0.0, 0.26, 0.32, 0.13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.34, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
		[0.0, 0.0, 0.0, 0.0, 0.0, 0.22, 0.25, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.46, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
		[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09, 0.2, 0.22, 0.23, 0.23, 0.22, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	];
	
	for (let k = 0; k < 1; k++) {
		const x = Math.floor(GRID_HEIGHT/2 - fish.length/2);
		const y = Math.floor(GRID_WIDTH/2 - fish[0].length/2);
		
		for (let i = 0; i < fish.length; i++) {
			for (let j = 0; j < fish[0].length; j++) {
				grid[x + i][y + j] = fish[i][j];
			}
		}
	}
	
	return grid;
}

function initRandomGrid() {
	return Array(GRID_HEIGHT).fill().map(() =>
		Array(GRID_WIDTH).fill().map(() => Math.random()));
}

/* drawKernel = gpu.createKernel(function(grid) {
    const value = grid[this.thread.y][this.thread.x];
}, {
    graphical: true,
    output: [GRID_WIDTH, GRID_HEIGHT]
}); */

function initSimulation() {
    if (isRunning) {
        cancelAnimationFrame(animationId);
        isRunning = false;
    }
    
    currentFrame = 0;
    frameCount.textContent = currentFrame;
    
    const type = simulationType.value;
    if (type === "39") {
        grid = initFishGrid();
    } else {
        grid = initRandomGrid();
    }
    
    // Initialiser nextGrid comme copie de grid
    nextGrid = grid.map(row => [...row]);
    
    drawGrid();
    
    statusText.textContent = "Ready";
    startBtn.disabled = false;
    pauseBtn.disabled = true;
    stepBtn.disabled = false;
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
	if (drawKernel) {
		// Utilisation du kernel GPU pour le rendu
		drawKernel(grid);

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(drawKernel.canvas, 0, 0, canvas.width, canvas.height);
        return;
	}
	
	// Rendu CPU
	const cellWidth = canvas.width / GRID_WIDTH;
	const cellHeight = canvas.height / GRID_HEIGHT;
	ctx.clearRect(0, 0, canvas.width, canvas.height);
	
	for (let i = 0; i < GRID_HEIGHT; i++) {
		for (let j = 0; j < GRID_WIDTH; j++) {
			const value = grid[i][j];
			ctx.fillStyle = viridisColor(value);
			ctx.fillRect(j * cellWidth, i * cellHeight, cellWidth, cellHeight);
		}
	}
}

function evolve() {
    if (convolveKernels.length > 0 && convolveKernels[0].kernel) {
        // Version GPU - utiliser GPU.createKernel pour gérer les textures
        const potentials = [
            convolveKernels[0](grid),
            convolveKernels[1](grid),
            convolveKernels[2](grid)
        ];
        
        const growthTerms = [
            growthKernel(potentials[0], MU[0], SIGMA[0]),
            growthKernel(potentials[1], MU[1], SIGMA[1]),
            growthKernel(potentials[2], MU[2], SIGMA[2])
        ];
        
        const avgGrowth = averageKernel(growthTerms[0], growthTerms[1], growthTerms[2]);
        
        // Créer un kernel temporaire pour la mise à jour
		const tempGrid = updateKernel(grid, avgGrowth);
		grid = tempGrid;
        
        // Copier les données dans la grille principale
        for (let i = 0; i < GRID_HEIGHT; i++) {
            for (let j = 0; j < GRID_WIDTH; j++) {
                grid[i][j] = tempGrid[i][j];
            }
        }
	} else {
		// Version CPU
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
	}
	
	// Échange des grilles
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
	
	// Calcul du FPS
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

// Initialisation
initKernels();
initSimulation();