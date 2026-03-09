'use strict';

/**
 * Paste image as PuzzleScript object: crop to 5x5, quantize to 10 colours + transparency,
 * insert object definition (name, hex palette, 5x5 pattern) at cursor.
 */

const PASTE_SIZE = 5;
const MAX_COLORS = 10;
const ALPHA_THRESHOLD = 128;

function rgbToHex(r, g, b) {
	return '#' + [r, g, b].map(x => {
		const h = Math.max(0, Math.min(255, Math.round(x))).toString(16);
		return h.length === 1 ? '0' + h : h;
	}).join('').toUpperCase();
}

function distSq(a, b) {
	return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;
}

/**
 * Quantize list of [r,g,b] colors to at most k colors using k-means.
 */
function quantizeColors(colorList, k) {
	// K-means: initialize centroids with first k distinct colors
	const distinct = [];
	const seen = new Map();
	var too_many_colours = false;
	for (const c of colorList) {
		const key = c[0] + ',' + c[1] + ',' + c[2];
		if (!seen.has(key)) {
			seen.set(key, true);
			distinct.push(c.slice());
			if (distinct.length >= k) {
				too_many_colours = true;
				break;
			}
		}
	}
	if (!too_many_colours) { //early out
		return distinct;
	}
	let centroids = distinct.slice(0, k);
	for (let iter = 0; iter < 15; iter++) {
		const sums = Array(k).fill(0).map(() => [0, 0, 0]);
		const counts = Array(k).fill(0);
		for (const c of colorList) {
			let best = 0;
			let bestD = distSq(c, centroids[0]);
			for (let i = 1; i < k; i++) {
				const d = distSq(c, centroids[i]);
				if (d < bestD) { bestD = d; best = i; }
			}
			sums[best][0] += c[0]; 
			sums[best][1] += c[1]; 
			sums[best][2] += c[2];

			counts[best]++;
		}
		for (let i = 0; i < k; i++) {
			if (counts[i] > 0) {
				centroids[i] = [
					sums[i][0] / counts[i],
					sums[i][1] / counts[i],
					sums[i][2] / counts[i]
				];
			}
		}
	}
	return centroids;
}

function suggest_unused_name(base, editor) {
	const raw = (editor && typeof editor.getValue === 'function') ? editor.getValue() : '';
	const text = (typeof raw === 'string' && raw) ? raw : '';
	for (let i = 1; i < 100; i++) {
		const name = base + '_' + i;
		if (text.indexOf(name) === -1) return name;
	}
	return base;
}

/**
 * Process image blob: crop top-left 5x5 (no resize); if image is smaller, pad with transparency.
 */
function imageBlobToObjectText(blob, editor) {
	return new Promise((resolve, reject) => {
		const img = new Image();
		const url = URL.createObjectURL(blob);
		img.onload = function () {
			URL.revokeObjectURL(url);
			try {
				const w = 5;
				const h = 5;
				const canvas = document.createElement('canvas');
				canvas.width = w;
				canvas.height = h;
				const ctx = canvas.getContext('2d');
				ctx.drawImage(img, 0, 0);
				const data = ctx.getImageData(0, 0, w, h).data;

				const pixels = [];
				const opaqueColors = [];
				for (let row = 0; row < PASTE_SIZE; row++) {
					for (let col = 0; col < PASTE_SIZE; col++) {
						if (col < w && row < h) {
							const i = (row * w + col) * 4;
							const r = data[i];
							const g = data[i + 1];
							const b = data[i + 2];
							const a = data[i + 3];
							if (a < ALPHA_THRESHOLD) {
								pixels.push(null);
							} else {
								pixels.push([r, g, b]);
								opaqueColors.push([r, g, b]);
							}
						} else {
							pixels.push(null);
						}
					}
				}

				const palette = quantizeColors(opaqueColors, MAX_COLORS);
				const paletteHex = palette.map(c => rgbToHex(c[0], c[1], c[2]));

				// Map each pixel to palette index digit or '.' for transparent
				const grid = [];
				for (let i = 0; i < pixels.length; i++) {
					const p = pixels[i];
					if (p === null) {
						grid.push('.');
						continue;
					}
					let best = 0;
					let bestD = distSq(p, palette[0]);
					for (let j = 1; j < palette.length; j++) {
						const d = distSq(p, palette[j]);
						if (d < bestD) { bestD = d; best = j; }
					}
					grid.push(String(best));
				}
				const name = suggest_unused_name("pasted", editor);
				const colorLine = paletteHex.join(' ');
				const gridLines = [];
				for (let row = 0; row < PASTE_SIZE; row++) {
					gridLines.push(grid.slice(row * PASTE_SIZE, (row + 1) * PASTE_SIZE).join(''));
				}
				const text = '\n' + name + '\n' + colorLine + '\n' + gridLines.join('\n') + '\n';
				resolve(text);
			} catch (e) {
				reject(e);
			}
		};
		img.onerror = function () {
			URL.revokeObjectURL(url);
			reject(new Error('Failed to load image'));
		};
		img.src = url;
	});
}

/**
 * Install paste handler on the CodeMirror editor: if clipboard contains an image,
 * prevent default paste, convert image to 5x5 object text, and insert at cursor.
 */
function installImagePasteHandler(editor) {
	const wrapper = editor.getWrapperElement();
	wrapper.addEventListener('paste', function (e) {
		const items = e.clipboardData && e.clipboardData.items;
		if (!items) return;
		let imageItem = null;
		for (let i = 0; i < items.length; i++) {
			if (items[i].type.indexOf('image') !== -1) {
				imageItem = items[i];
				break;
			}
		}
		if (!imageItem) return;
		e.preventDefault();
		e.stopPropagation();
		const blob = imageItem.getAsFile();
		if (!blob) return;
		imageBlobToObjectText(blob, editor).then(function (text) {
			editor.replaceSelection(text);
			editor.focus();
		}).catch(function (err) {
			consoleError('Paste image failed: ' + (err && err.message ? err.message : String(err)));
		});
	}, true);
}
