'use strict';

let audio;


function newSound(instrument) {
	let seed = instrument + 100 * ((Math.random() * 1000000) | 1);
	//document.getElementById('sounddat').value = seed;

	let frame = parent.frames[4];
	let code = document.getElementById('consoletextarea');
	consolePrint(generatorNames[instrument] + ' : ' + '<span class="cm-SOUND" onclick="playSound(' + seed.toString() + ',true)">' + seed.toString() + '</span>', true);
	let params = generateFromSeed(seed);
	params.sound_vol = SOUND_VOL;
	params.sample_rate = SAMPLE_RATE;
	params.bit_depth = BIT_DEPTH;
	let sound = SoundEffect.generate(params);
	sound.play();
}

function buttonPress() {
	let generatortype = 3;
	let seed = document.getElementById('sounddat').value;
	let params = generateFromSeed(seed);
	params.sound_vol = SOUND_VOL;
	params.sample_rate = SAMPLE_RATE;
	params.bit_depth = BIT_DEPTH;
	let sound = SoundEffect.generate(params);
	sound.play();
}
