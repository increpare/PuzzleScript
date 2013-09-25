
var audio;


function newSound(instrument) {
	var seed = instrument + 100 * ((Math.random() * 1000000) | 1);
	document.getElementById('sounddat').value = seed;

	var frame = parent.frames[4];
	var code = document.getElementById('consoletextarea');
	consolePrint(generatorNames[instrument] + ' : ' + '<span class="cm-SOUND" onclick="playSound(' + seed.toString() + ')">' + seed.toString() + '</span>');
	var params = generateFromSeed(seed);
	params.sound_vol = SOUND_VOL;
	params.sample_rate = SAMPLE_RATE;
	params.sample_size = SAMPLE_SIZE;
	var sound = generate(params);
	var audio = new Audio();
	audio.src = sound.dataURI;
	audio.play();
}

function buttonPress() {
	var generatortype = 3;
	var seed = document.getElementById('sounddat').value;
	var params = generateFromSeed(seed);
	params.sound_vol = SOUND_VOL;
	params.sample_rate = SAMPLE_RATE;
	params.sample_size = SAMPLE_SIZE;
	var sound = generate(params);
	var audio = new Audio();
	audio.src = sound.dataURI;
	audio.play();
}
