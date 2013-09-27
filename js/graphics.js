

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
        /*
    //  ctx.drawImage(spriteimages[0],0,0);
        ctx.fillStyle="#000000";
        ctx.fillText("Coordinates: (" + x + "," + y + ")",x,y);
        */
    }
}


var lastDownTarget;

var oldcellwidth=0;
var oldcellheight=0;
var oldtextmode=-1;
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
    }
    else if (cellheight > cellwidth) {
        cellheight = cellwidth;
        yoffset = (canvas.height - cellheight * screenheight) / 2;
    }
    cellwidth = ~~cellwidth;
    cellheight = ~~cellheight;
    xoffset = ~~xoffset;
    yoffset = ~~yoffset;

    if (oldcellwidth!=cellwidth||oldcellheight!=cellheight||oldtextmode!=textMode) {
    	regenSpriteImages();
    }
    oldcellheight=cellheight;
    oldcellwidth=cellwidth;
    oldtextmode=textMode;
    
    redraw();
}