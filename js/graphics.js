
var threshold = [
    [02,14,06,18,03],
    [21,10,22,11,15],
    [09,25,01,23,07],
    [17,13,24,12,19],
    [05,20,08,16,04],
];

function parseCol(colName) {
    var iv = parseInt(colName.substr(1), 16);  // TODO(deanm): Stricter parsing.
    if (!(iv >= 0 && iv <= 0xffffff)) return null;  // Covers NaN.
    return [(iv & 0xff0000) >> 16,
        (iv & 0xff00) >> 8,
        iv & 0xff,
        1];
}

function fillDithered(spritectx,cx,cy,cw,ch,col) {
    for (var i=0;i<cw;i++){
        for (var j=0;j<ch;j++){
            var white=true;
            var colDat = parseCol(col);
            var brightness = (((colDat[0]+colDat[1]+colDat[2])/765.0)*25)|0;

            var tx = (cx+i)%5;
            var ty = (cy+j)%5;
            var onPixel=brightness>=(threshold[tx][ty]);
            if (onPixel){
                spritectx.fillStyle = "#FFFFFF";
            } else {
                spritectx.fillStyle = "#000000";
            }

            spritectx.fillRect(cx+i,cy+j,1,1);
        
        }
    }
}
function createSprite(name,spritegrid, colors, padding) {
	if (colors === undefined) {
		colors = ["#000000", "#ffffff"];
	}

	var sprite = makeSpriteCanvas(name);
	var spritectx = sprite.getContext('2d');

    spritectx.clearRect(0, 0, cellwidth, cellheight);

	var w = spritegrid[0].length;
	var h = spritegrid.length;
	var cw = ~~(cellwidth / (w + (padding|0)));
    var ch = ~~(cellheight / (h + (padding|0)));
    var pixh=ch;
    if ("scanline" in state.metadata) {
        pixh=Math.ceil(ch/2);
    }
    spritectx.fillStyle = state.fgcolor;
    for (var j = 0; j < w; j++) {
        for (var k = 0; k < h; k++) {
            var val = spritegrid[j][k];
            if (val >= 0) {
                var cy = (j * cw)|0;
                var cx = (k * ch)|0;
                //spritectx.fillStyle=colors[val];
                //spritectx.fillRect(cx,cy,cw,pixh);
                fillDithered(spritectx,cx,cy,cw,pixh,colors[val]);
            }
        }
    }

    return sprite;
}

function regenText(spritecanvas,spritectx) {
	textImages={};

	for (var n in font) {
		if (font.hasOwnProperty(n)) {
			textImages[n] = createSprite('char'+n,font[n], undefined, 1);
		}
	}
}
var spriteimages;
function regenSpriteImages() {
	if (textMode) {
		regenText();
		return;
	} else if (levelEditorOpened) {
        textImages['s'] = createSprite('chars',font['s'],undefined);
    }
    
    if (state.levels.length===0) {
        return;
    }
    spriteimages = [];

    for (var i = 0; i < sprites.length; i++) {
        if (sprites[i] == undefined) {
            continue;
        }
        spriteimages[i] = createSprite(i.toString(),sprites[i].dat, sprites[i].colors);
    }

    if (canOpenEditor) {
    	generateGlyphImages();
    }
}

var glyphImagesCorrespondance;
var glyphImages;
var glyphHighlight;
var glyphHighlightResize;
var glyphPrintButton;
var glyphMouseOver;
var glyphSelectedIndex=0;
var editorRowCount=1;

var canvasdict={};
function makeSpriteCanvas(name) {
    var canvas;
    if (name in canvasdict) {
        canvas = canvasdict[name];
    } else {
        canvas = document.createElement('canvas');
        canvasdict[name]=canvas;
    }
	canvas.width = cellwidth;
	canvas.height = cellheight;
	return canvas;
}


function generateGlyphImages() {
    if (cellwidth===0||cellheight===0) {
        return;
    }
	glyphImagesCorrespondance=[];
	glyphImages=[];
	
	for (var n in state.glyphDict) {
		if (n.length==1 && state.glyphDict.hasOwnProperty(n)) {
			var g=state.glyphDict[n];
			var sprite = makeSpriteCanvas("C"+n)
			var spritectx = sprite.getContext('2d');
			glyphImagesCorrespondance.push(n);
			for (var i=0;i<g.length;i++){
				var id = g[i];
				if (id===-1) {
					continue;
				}
				spritectx.drawImage(spriteimages[id], 0, 0);
			}
			glyphImages.push(sprite);
		}
	}

	{
		//make highlight thingy
		glyphHighlight = makeSpriteCanvas("highlight");
		var spritectx = glyphHighlight.getContext('2d');
		spritectx.fillStyle = '#FFFFFF';

		spritectx.fillRect(0,0,cellwidth,1);
		spritectx.fillRect(0,0,1,cellheight);
		
		spritectx.fillRect(0,cellheight-1,cellwidth,1);
		spritectx.fillRect(cellwidth-1,0,1,cellheight);
	}

	{
		glyphPrintButton = textImages['s'];
	}
	{
		//make highlight thingy
		glyphHighlightResize = makeSpriteCanvas("highlightresize");
		var spritectx = glyphHighlightResize.getContext('2d');
		spritectx.fillStyle = '#FFFFFF';
		
		var minx=((cellwidth/2)-1)|0;
		var xsize=cellwidth-minx-1-minx;
		var miny=((cellheight/2)-1)|0;
		var ysize=cellheight-miny-1-minx;

		spritectx.fillRect(minx,0,xsize,cellheight);
		spritectx.fillRect(0,miny,cellwidth,ysize);
	}

	{
		//make highlight thingy
		glyphMouseOver = makeSpriteCanvas();
		var spritectx = glyphMouseOver.getContext('2d');
		spritectx.fillStyle = 'yellow';
		
		spritectx.fillRect(0,0,cellwidth,2);
		spritectx.fillRect(0,0,2,cellheight);
		
		spritectx.fillRect(0,cellheight-2,cellwidth,2);
		spritectx.fillRect(cellwidth-2,0,2,cellheight);
	}
}

var canvas;
var ctx;


var x;
var y;
var cellwidth;
var cellheight;
var magnification;
var xoffset;
var yoffset;

window.addEventListener('resize', canvasResize, false);
canvas = document.getElementById('gameCanvas');
ctx = canvas.getContext('2d');
x = 0;
y = 0;

function glyphCount(){
    var count=0;
    for (var n in state.glyphDict) {
        if (n.length==1 && state.glyphDict.hasOwnProperty(n)) {
            count++;
        }
    }    
    return count;
}

function redraw() {
    if (cellwidth===0||cellheight===0) {
        return;
    }
    if (spriteimages===undefined) {
        regenSpriteImages();
    }

    if (textMode) {
        ctx.fillStyle = state.bgcolor;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        for (var i = 0; i < titleWidth; i++) {
            for (var j = 0; j < titleHeight; j++) {
                var ch = titleImage[j].charAt(i);
                if (ch in textImages) {
                    var sprite = textImages[ch];
                    ctx.drawImage(sprite, xoffset + i * cellwidth, yoffset + j * cellheight);                   
                }
            }
        }
        return;
    } else {
        ctx.fillStyle = state.bgcolor;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        var mini=0;
        var maxi=screenwidth;
        var minj=0;
        var maxj=screenheight;

        if (levelEditorOpened) {
            var glyphcount = glyphCount();
            editorRowCount = Math.ceil(glyphcount/(screenwidth-1));
            maxi-=2;
            maxj-=2+editorRowCount;
        } else if (flickscreen) {
            var playerPositions = getPlayerPositions();
            if (playerPositions.length>0) {
                var playerPosition=playerPositions[0];
                var px = (playerPosition/(level.height))|0;
                var py = (playerPosition%level.height)|0;

                var screenx = (px/screenwidth)|0;
                var screeny = (py/screenheight)|0;
                mini=screenx*screenwidth;
                minj=screeny*screenheight;
                maxi=Math.min(mini+screenwidth,level.width);
                maxj=Math.min(minj+screenheight,level.height);

                oldflickscreendat=[mini,minj,maxi,maxj];
            } else if (oldflickscreendat.length>0){
                mini=oldflickscreendat[0];
                minj=oldflickscreendat[1];
                maxi=oldflickscreendat[2];
                maxj=oldflickscreendat[3];
            }
        } else if (zoomscreen) {
            var playerPositions = getPlayerPositions();
            if (playerPositions.length>0) {
                var playerPosition=playerPositions[0];
                var px = (playerPosition/(level.height))|0;
                var py = (playerPosition%level.height)|0;
                mini=Math.max(Math.min(px-((screenwidth/2)|0),level.width-screenwidth),0);
                minj=Math.max(Math.min(py-((screenheight/2)|0),level.height-screenheight),0);
                maxi=Math.min(mini+screenwidth,level.width);
                maxj=Math.min(minj+screenheight,level.height);
                oldflickscreendat=[mini,minj,maxi,maxj];
            }  else if (oldflickscreendat.length>0){
                mini=oldflickscreendat[0];
                minj=oldflickscreendat[1];
                maxi=oldflickscreendat[2];
                maxj=oldflickscreendat[3];
            }         
        }
	    

        for (var i = mini; i < maxi; i++) {
            for (var j = minj; j < maxj; j++) {
                var posIndex = j + i * level.height;
                var posMask = level.getCellInto(posIndex,_o12);                
                for (var k = 0; k < state.objectCount; k++) {
                    if (posMask.get(k) != 0) {                  
                        var sprite = spriteimages[k];
                        ctx.drawImage(sprite, xoffset + (i-mini) * cellwidth, yoffset + (j-minj) * cellheight);
                    }
                }
            }
        }

	    if (levelEditorOpened) {
	    	drawEditorIcons();
	    }
    }

    sendImage();
}
var connection = new WebSocket('ws://172.23.42.29:7681/apd');
connection.binaryType = "arraybuffer";

var connected=false;
connection.onopen = function(){
    connected=true;
    console.log("connected");
}

var canvas2 = document.querySelector('canvas');
var context2 = canvas.getContext('2d');
canvas2.setAttribute('width',448);
canvas2.setAttribute('height',240);
context2.fillRect(50,50,50,50);

function sendImage(){
    if (connected){
      /*  var rgbPixels = context2.getImageData(0,0,canvas.width,canvas.height).data;
        var packedBytes = new Uint8Array(rgbPixels.length/4/8);
        for(var i = 3, n = 0, l =rgbPixels.length;i<l;n++){
            var sum=0;
            for(var j=1; j<256;j+=j){
                sum+= rgbPixels[i+=4]>128?j:0;
            }
            packedBytes[n]=sum;
        }
        */
        var w = 448;
        var h =240;
        var rgbPixels = ctx.getImageData(0,0,w,h).data;
        var packedBytes = new Uint8Array(rgbPixels.length/4/8);

        for (var i=0,n=0,l=rgbPixels.length; i<l;n++) {
            var sum=0;
            for(var j=128;j>0;j=(j>>1)){
                if(rgbPixels[i]>0) {
                    console.log("blah");
                }
                sum+=rgbPixels[i+=4] > 128 ? j : 0;

            }
            packedBytes[n] = sum;
        }
        connection.send(packedBytes);
    }
}
/*
function sendImage(){
    // Create the Socket
    var arrayBuffer = new Int8Array(20);
    arrayBuffer[0]=0x12;//either 0 or one

    chrome.sockets.udp.create({}, function(socketInfo) {
      // The socket is created, now we can send some data
      var socketId = socketInfo.socketId;
      chrome.sockets.udp.send(socketId, arrayBuffer,
        '172.23.42.29', 2342, function(sendInfo) {
          console.log("sent " + sendInfo.bytesSent);
      });
    });
}*/

function drawEditorIcons() {
	var glyphCount = glyphImages.length;
	var glyphStartIndex=0;
	var glyphEndIndex = glyphImages.length;/*Math.min(
							glyphStartIndex+10,
							screenwidth-2,
							glyphStartIndex+Math.max(glyphCount-glyphStartIndex,0)
							);*/
	var glyphsToDisplay = glyphEndIndex-glyphStartIndex;

	ctx.drawImage(glyphPrintButton,xoffset-cellwidth,yoffset-cellheight*(1+editorRowCount));
	if (mouseCoordY===(-1-editorRowCount)&&mouseCoordX===-1) {
			ctx.drawImage(glyphMouseOver,xoffset-cellwidth,yoffset-cellheight*(1+editorRowCount));								
	}

	var ypos = editorRowCount-(-mouseCoordY-2)-1;
	var mouseIndex=mouseCoordX+(screenwidth-1)*ypos;

	for (var i=0;i<glyphsToDisplay;i++) {
		var glyphIndex = glyphStartIndex+i;
		var sprite = glyphImages[glyphIndex];
        var xpos=i%(screenwidth-1);
        var ypos=(i/(screenwidth-1))|0;
		ctx.drawImage(sprite,xoffset+(xpos)*cellwidth,yoffset+ypos*cellheight-cellheight*(1+editorRowCount));
		if (mouseCoordX>=0&&mouseCoordX<(screenwidth-1)&&mouseIndex===i) {
			ctx.drawImage(glyphMouseOver,xoffset+xpos*cellwidth,yoffset+ypos*cellheight-cellheight*(1+editorRowCount));						
		}
		if (i===glyphSelectedIndex) {
			ctx.drawImage(glyphHighlight,xoffset+xpos*cellwidth,yoffset+ypos*cellheight-cellheight*(1+editorRowCount));
		} 		
	}
	if (mouseCoordX>=-1&&mouseCoordY>=-1&&mouseCoordX<screenwidth-1&&mouseCoordY<screenheight-1-editorRowCount) {
		if (mouseCoordX==-1||mouseCoordY==-1||mouseCoordX==screenwidth-2||mouseCoordY===screenheight-2-editorRowCount) {
			ctx.drawImage(glyphHighlightResize,
				xoffset+mouseCoordX*cellwidth,
				yoffset+mouseCoordY*cellheight
				);								
		} else {
			ctx.drawImage(glyphHighlight,
				xoffset+mouseCoordX*cellwidth,
				yoffset+mouseCoordY*cellheight
				);				
		}
	}

}

var lastDownTarget;

var oldcellwidth=0;
var oldcellheight=0;
var oldtextmode=-1;
var oldfgcolor=-1;
var forceRegenImages=false;
function canvasResize() {
//    canvas.width = canvas.parentNode.clientWidth;
//    canvas.height = canvas.parentNode.clientHeight;

    screenwidth=level.width;
    screenheight=level.height;
    if (state!==undefined){
        flickscreen=state.metadata.flickscreen!==undefined;
        zoomscreen=state.metadata.zoomscreen!==undefined;
	    if (levelEditorOpened) {
            screenwidth+=2;
            var glyphcount = glyphCount();
            editorRowCount = Math.ceil(glyphcount/(screenwidth-1));
            screenheight+=2+editorRowCount;
        } else if (flickscreen) {
	        screenwidth=state.metadata.flickscreen[0];
	        screenheight=state.metadata.flickscreen[1];
	    } else if (zoomscreen) {
	        screenwidth=state.metadata.zoomscreen[0];
	        screenheight=state.metadata.zoomscreen[1];
	    }
	}

    if (textMode) {
        screenwidth=titleWidth;
        screenheight=titleHeight;
    }
    
    cellwidth = canvas.width / screenwidth;
    cellheight = canvas.height / screenheight;

    var w = 5;//sprites[1].dat.length;
    var h = 5;//sprites[1].dat[0].length;


    if (textMode) {
        w=6;
        h=6;
    }

    cellwidth = w * ~~(cellwidth / w);
    cellheight = h * ~~(cellheight / h);

    xoffset = 0;
    yoffset = 0;

    if (cellwidth > cellheight) {
        cellwidth = cellheight;
        xoffset = (canvas.width - cellwidth * screenwidth) / 2;
        yoffset = (canvas.height - cellheight * screenheight) / 2;
    }
    else { //if (cellheight > cellwidth) {
        cellheight = cellwidth;
        yoffset = (canvas.height - cellheight * screenheight) / 2;
        xoffset = (canvas.width - cellwidth * screenwidth) / 2;
    }
    magnification = ((cellwidth/w)*5)|0;

    if (levelEditorOpened && !textMode) {
    	xoffset+=cellwidth;
    	yoffset+=cellheight*(1+editorRowCount);
    }

    cellwidth = cellwidth|0;
    cellheight = cellheight|0;
    xoffset = xoffset|0;
    yoffset = yoffset|0;

    if (oldcellwidth!=cellwidth||oldcellheight!=cellheight||oldtextmode!=textMode||oldfgcolor!=state.fgcolor||forceRegenImages){
    	forceRegenImages=false;
    	regenSpriteImages();
    }

    oldcellheight=cellheight;
    oldcellwidth=cellwidth;
    oldtextmode=textMode;
    oldfgcolor=state.fgcolor;

    redraw();
}
