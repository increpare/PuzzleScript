
String.prototype.replaceAll = function(search, replacement) {
    var target = this;
    return target.replace(new RegExp(search, 'g'), replacement);
};

function ARDU_spriteGlyphs(){
    var strb=`PROGMEM const byte tiles_b[][8] = {\n`;
    var strw=`PROGMEM const byte tiles_w[][8] = {\n`;
    for (var i = 0; i < sprites.length; i++) {
        strb+="\t{\n"
        strw+="\t{\n"
        if (sprites[i] == undefined) {
            continue;
        }

        var s = sprites[i];

        for (var j=0;j<8;j++){
            var black="";
            var white="";
            for (var k=0;k<8;k++){
                var p = s.dat[k][j]
                var c = (p==-1||p==".")?".":s.colors[ p ];                
                c = c.toLowerCase();
                if (c==="black"||c==="#000000"){
                    black="1"+black;
                } else {
                    black="0"+black;
                }
                if (c==="white"||c==="#ffffff"){
                    white="1"+white;
                } else {
                    white="0"+white;
                }
            }
            strb+="\t\t0b"+black+",\n"
            strw+="\t\t0b"+white+",\n"
        }
        strb+="\t},\n"
        strw+="\t},\n"
    }
    strb+=`};\n`
    strw+=`};\n`
    return `const int GLYPH_COUNT = ` +sprites.length+`;\n\n`+strb+'\n'+strw;
}

function ARDU_levelDat(){
	var levelDat = `const byte levels[][128] {\n`
	for (var i=0;i<state.levels.length;i++){
		var level=state.levels[i]
		levelDat+="\t{\n\t\t"
		for (var j=0;j<8;j++){
			for (var k=0;k<16;k++){
				var idx = j+8*k
				levelDat+=level.objects[idx]+","
			}
			levelDat+="\n"
			if (j<7){
				levelDat+="\t\t"
			}
		}
		levelDat+="\t},\n"
	}
	levelDat+="};\n"
	return levelDat
}

function printInt(n,bits){
    var result = n.toString(2);
    while(result.length<bits){
        result = '0'+result;;
    }
    return '0b'+result;
}

function ARDU_playerConstants(){
    var result="";

    result+= "const byte PLAYER_MASK = "+printInt(state.playerMask.data[0],8)+";\n";
    
    var playerLayers = getLayersOfMask(state.playerMask)
    var playerLayerMask = 0;
    for (var i=0;i<playerLayers.length;i++){
        var l = playerLayers.length;
        playerLayerMask+=0b11111<<(5*l);
    }
    result += "const word PLAYER_LAYERMASK = "+printInt(playerLayerMask,16)+";\n";

    return result;
}
function exportEmbeddedClick(){


	var sourceCode = editor.getValue();

	compile("restart");
    
    var outputTxt="\n";

    var playerConsts = ARDU_playerConstants();
    outputTxt+=playerConsts;
    
	var glyphText = ARDU_spriteGlyphs();
	outputTxt+=glyphText+"\n";
    
    var levelText = ARDU_levelDat();
	outputTxt+=levelText+"\n";

	addToConsole(outputTxt)
	console.log(outputTxt)

}