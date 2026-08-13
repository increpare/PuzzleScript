'use strict';

const MINIMUM_COLOR_CONTRAST_RATIO = 2.361;

function parseHexColor(hexColor) {
	hexColor = hexColor.trim();
	if (!/^#(?:[0-9a-f]{3}|[0-9a-f]{6})$/i.test(hexColor)) {
		return null;
	}

	if (hexColor.length === 4) {
		return [
			parseInt(hexColor.charAt(1), 16) * 0x11,
			parseInt(hexColor.charAt(2), 16) * 0x11,
			parseInt(hexColor.charAt(3), 16) * 0x11
		];
	}

	return [
		parseInt(hexColor.slice(1, 3), 16),
		parseInt(hexColor.slice(3, 5), 16),
		parseInt(hexColor.slice(5, 7), 16)
	];
}

function relativeLuminance(rgb) {
	const channels = rgb.map(channel => {
		channel /= 255;
		return channel <= 0.03928
			? channel / 12.92
			: Math.pow((channel + 0.055) / 1.055, 2.4);
	});
	return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
}

function contrastRatio(firstColor, secondColor) {
	const firstLuminance = relativeLuminance(firstColor);
	const secondLuminance = relativeLuminance(secondColor);
	const lighter = Math.max(firstLuminance, secondLuminance);
	const darker = Math.min(firstLuminance, secondLuminance);
	return (lighter + 0.05) / (darker + 0.05);
}

function updateFocusBorderColour(backgroundColor) {
	const gamePanel = document.getElementById("righttophalf");
	if (gamePanel === null) {
		return;
	}

	const orange = parseHexColor("#FFA500");
	const green = parseHexColor("#1DC116");
	const background = parseHexColor(backgroundColor);
	let focusColour = "orange";

	if (background !== null) {
		const orangeContrast = contrastRatio(background, orange);
		const greenContrast = contrastRatio(background, green);
		if (orangeContrast < MINIMUM_COLOR_CONTRAST_RATIO && greenContrast > orangeContrast) {
			focusColour = "#1DC116";
		}
	}

	gamePanel.style.setProperty("--focus-border-colour", focusColour);
}
