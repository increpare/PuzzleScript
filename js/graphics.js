

function setPixel(imageData, x, y, r, g, b, a) {
    index = (x + y * imageData.width) * 4;
    imageData.data[index + 0] = r;
    imageData.data[index + 1] = g;
    imageData.data[index + 2] = b;
    imageData.data[index + 3] = a;
}


function createSprite(spritecanvas,spritectx,n) {
    spritectx.clearRect(0, 0, cellwidth, cellheight);

    var spritegrid = font[n];

//      window.console.log(w+","+h+","+cw+","+ch);

	var w=5;
	var h=5;
	var cw = ~~(cellwidth / w);
    var ch = ~~(cellheight / h);

    spritectx.fillStyle = "#ffffff";
    for (var j = 0; j < w; j++) {
        for (var k = 0; k < h; k++) {
            var cy = ~~(j * cw);
            var cx = ~~(k * ch);
            if (spritegrid[j][k] == 1) {
                spritectx.fillRect(cx, cy, cw, ch);
            }
        }
    }


    var new_image_url = spritecanvas.toDataURL();
    var img = document.createElement('img');
    img.onload = redraw;
    img.src = new_image_url;

    return img;
}

function regenText(spritecanvas,spritectx) {
	textImages={};

	for (var n in font) {
		if (font.hasOwnProperty(n)) {
			textImages[n]=createSprite(spritecanvas,spritectx,n);
		}
	}
}
var spriteimages;
function regenSpriteImages() {

    var spritecanvas = document.createElement('canvas');
    spritecanvas.width = cellwidth;
    spritecanvas.height = cellheight;

    var spritectx = spritecanvas.getContext('2d');



    var w = 5;//sprites[0].dat.length;
    var h = 5;//sprites[0].dat[0].length;
    var cw = ~~(cellwidth / w);
    var ch = ~~(cellheight / h);



	if (textMode) {
		regenText(spritecanvas,spritectx);
		return;
	}
    spriteimages = [];



    for (var i = 0; i < sprites.length; i++) {
        if (sprites[i] == undefined) {
            continue;
        }

        spritectx.clearRect(0, 0, cellwidth, cellheight);

        var spritegrid = sprites[i].dat;


//      window.console.log(w+","+h+","+cw+","+ch);

        var colors = sprites[i].colors;
        for (var j = 0; j < w; j++) {
            for (var k = 0; k < h; k++) {
                var val = spritegrid[j][k];
                if (val>=0) {
	                var cy = (j * cw)|0;
	                var cx = (k * ch)|0;
                	spritectx.fillStyle = colors[val];
                    spritectx.fillRect(cx, cy, cw, ch);
                }
            }
        }
       

        var new_image_url = spritecanvas.toDataURL();
        var img = document.createElement('img');
        img.onload = redraw;
        img.src = new_image_url;

        spriteimages[i] = img;
    }

    if (canOpenEditor) {
    	generateGlyphImages(spritecanvas,spritectx);
    }
}

var glyphImagesCorrespondance;
var glyphImages;
var glyphHighlight;
var glyphHighlightResize;
var glyphPrintButton;
var glyphMouseOver;
var glyphSelectedIndex=0;
function generateGlyphImages(spritecanvas,spritectx) {
	glyphImagesCorrespondance=[];
	glyphImages=[];
	
	for (var n in state.glyphDict) {
		if (n.length==1 && state.glyphDict.hasOwnProperty(n)) {
			var g=state.glyphDict[n];
			glyphImagesCorrespondance.push(n);

			spritectx.clearRect(0, 0, cellwidth, cellheight);

			for (var i=0;i<g.length;i++){
				var id = g[i];
				if (id===-1) {
					continue;
				}
				var s = spriteimages[id];
				spritectx.drawImage(s,0,0);
			}


	        var new_image_url = spritecanvas.toDataURL();
	        var img = document.createElement('img');
	        img.onload = redraw;
	        img.src = new_image_url;

			glyphImages.push(img);
		}
	}

	{
		//make highlight thingy
    	spritectx.fillStyle = '#FFFFFF';	
		spritectx.clearRect(0, 0, cellwidth, cellheight);
		
		spritectx.fillRect(0,0,cellwidth,1);
		spritectx.fillRect(0,0,1,cellheight);
		
		spritectx.fillRect(0,cellheight-1,cellwidth,1);
		spritectx.fillRect(cellwidth-1,0,1,cellheight);


	    var new_image_url = spritecanvas.toDataURL();
	    var img = document.createElement('img');
	    img.onload = redraw;
	    img.src = new_image_url;

		glyphHighlight=img;
	}

	{
		glyphPrintButton=createSprite(spritecanvas,spritectx,'s');
	}
	{
		//make highlight thingy
    	spritectx.fillStyle = '#FFFFFF';	
		spritectx.clearRect(0, 0, cellwidth, cellheight);
		
		var minx=((cellwidth/2)-1)|0;
		var xsize=cellwidth-minx-1-minx;
		var miny=((cellheight/2)-1)|0;
		var ysize=cellheight-miny-1-minx;

		spritectx.fillRect(minx,0,xsize,cellheight);
		spritectx.fillRect(0,miny,cellwidth,ysize);


	    var new_image_url = spritecanvas.toDataURL();
	    var img = document.createElement('img');
	    img.onload = redraw;
	    img.src = new_image_url;

		glyphHighlightResize=img;
	}

	{
		//make highlight thingy
    	spritectx.fillStyle = 'yellow';
		spritectx.clearRect(0, 0, cellwidth, cellheight);
		
		spritectx.fillRect(0,0,cellwidth,2);
		spritectx.fillRect(0,0,2,cellheight);
		
		spritectx.fillRect(0,cellheight-2,cellwidth,2);
		spritectx.fillRect(cellwidth-2,0,2,cellheight);


	    var new_image_url = spritecanvas.toDataURL();
	    var img = document.createElement('img');
	    img.onload = redraw;
	    img.src = new_image_url;

		glyphMouseOver=img;
	}
}

var canvas;
var ctx;



var x;
var y;
var cellwidth;
var cellheight;
var xoffset;
var yoffset;

window.console.log('init');
window.addEventListener('resize', canvasResize, false);
canvas = document.getElementById('gameCanvas');
ctx = canvas.getContext('2d');
x = 0;
y = 0;

function redraw() {
    if (textMode) {
        for (var n in textImages) {
            if (textImages.hasOwnProperty(n)) {
                var spriteimage = textImages[n];
                if (!spriteimage.complete)
                    return;
            }
        }

        ctx.fillStyle = '#000000';
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

        if (spriteimages===undefined) {
            regenSpriteImages();
        }

        for (var i = 0; i < spriteimages.length; i++) {
            var spriteimage = spriteimages[i];
            if (spriteimage == undefined)
                continue;
            if (!spriteimage.complete)
                return;
        }

        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        var mini=0;
        var maxi=screenwidth;
        var minj=0;
        var maxj=screenheight;

        if (flickscreen) {
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
            }
        } else if (zoomscreen) {
            var playerPositions = getPlayerPositions();
            if (playerPositions.length>0) {
                var playerPosition=playerPositions[0];
                var px = (playerPosition/(level.height))|0;
                var py = (playerPosition%level.height)|0;
                mini=Math.max(px-((screenwidth/2)|0),0);
                minj=Math.max(py-((screenheight/2)|0),0);
                maxi=Math.min(mini+screenwidth,level.width);
                maxj=Math.min(minj+screenheight,level.height);
            }           
        }
	    if (levelEditorOpened) {
	    	maxi-=2;
	    	maxj-=3;
	    }

        for (var i = mini; i < maxi; i++) {
            for (var j = minj; j < maxj; j++) {
    /*          if (grid[i][j]==0){
                    ctx.fillStyle="#00FF00";
                }
                else {
                    ctx.fillStyle="#0000FF";
                }

                ctx.fillRect(xoffset+i*cellwidth,yoffset+j*cellheight,i+1*cellwidth,j+1*cellheight);
    */
                var posIndex = j + i * level.height;
                var posMask = level.dat[posIndex];

                for (var k = 0; k < state.objectCount; k++) {
                    var spriteMask = 1 << k;
                    if ((posMask & spriteMask) != 0) {                  
                        var sprite = spriteimages[k];
                        ctx.drawImage(sprite, xoffset + (i-mini) * cellwidth, yoffset + (j-minj) * cellheight);
                    }
                }
            }
        }

	    if (levelEditorOpened) {
	    	drawEditorIcons();
	    }
        /*
    //  ctx.drawImage(spriteimages[0],0,0);
        ctx.fillStyle="#000000";
        ctx.fillText("Coordinates: (" + x + "," + y + ")",x,y);
        */
    }
}

function drawEditorIcons() {
	var glyphCount = glyphImages.length;
	var glyphStartIndex=0;
	var glyphEndIndex = glyphImages.length;/*Math.min(
							glyphStartIndex+10,
							screenwidth-2,
							glyphStartIndex+Math.max(glyphCount-glyphStartIndex,0)
							);*/
	var glyphsToDisplay = glyphEndIndex-glyphStartIndex;

	ctx.drawImage(glyphPrintButton,xoffset-cellwidth,yoffset-cellheight*2);
	if (mouseCoordY===-2&&mouseCoordX===-1) {
			ctx.drawImage(glyphMouseOver,xoffset-cellwidth,yoffset-cellheight*2);								
	}

	for (var i=0;i<glyphsToDisplay;i++) {
		var glyphIndex = glyphStartIndex+i;
		var sprite = glyphImages[glyphIndex];
		ctx.drawImage(sprite,xoffset+(i)*cellwidth,yoffset-cellheight*2);
		if (mouseCoordY===-2&&mouseCoordX===i) {
			ctx.drawImage(glyphMouseOver,xoffset+i*cellwidth,yoffset-cellheight*2);						
		}
		if (i===glyphSelectedIndex) {
			ctx.drawImage(glyphHighlight,xoffset+i*cellwidth,yoffset-cellheight*2);
		} 		
	}
	if (mouseCoordX>=-1&&mouseCoordY>=-1&&mouseCoordX<screenwidth-1&&mouseCoordY<screenheight-2) {
		if (mouseCoordX==-1||mouseCoordY==-1||mouseCoordX==screenwidth-2||mouseCoordY===screenheight-3) {
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
var forceRegenImages=false;
function canvasResize() {
//  window.console.log("canvasresize");
    canvas.style.width = canvas.parentNode.clientWidth;
        canvas.style.height = canvas.parentNode.clientHeight;

    canvas.width = canvas.parentNode.clientWidth;
        canvas.height = canvas.parentNode.clientHeight;

    screenwidth=level.width;
    screenheight=level.height;
    if (state!==undefined){
	     flickscreen=state.metadata.flickscreen!==undefined;
	    if (flickscreen) {
	        screenwidth=state.metadata.flickscreen[0];
	        screenheight=state.metadata.flickscreen[1];
	    }
	    zoomscreen=state.metadata.zoomscreen!==undefined;
	    if (zoomscreen) {
	        screenwidth=state.metadata.zoomscreen[0];
	        screenheight=state.metadata.zoomscreen[1];
	    }
	}

    if (textMode) {
    	levelEditorOpened=false;
        screenwidth=titleWidth;
        screenheight=titleHeight;
    }
    if (levelEditorOpened) {
    	screenwidth+=2;
    	screenheight+=3;
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

    if (levelEditorOpened) {
    	xoffset+=cellwidth;
    	yoffset+=cellheight*2;
    }

    cellwidth = cellwidth|0;
    cellheight = cellheight|0;
    xoffset = xoffset|0;
    yoffset = yoffset|0;

    if (oldcellwidth!=cellwidth||oldcellheight!=cellheight||oldtextmode!=textMode||forceRegenImages) {
    	forceRegenImages=false;
    	regenSpriteImages();
    }

    oldcellheight=cellheight;
    oldcellwidth=cellwidth;
    oldtextmode=textMode;

    redraw();
}