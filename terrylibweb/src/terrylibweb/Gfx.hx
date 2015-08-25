package terrylibweb;
	
import terrylibweb.util.*;
import openfl.display.*;
import openfl.geom.*;
import openfl.events.*;
import openfl.net.*;
import openfl.text.*;
import openfl.Lib;
import openfl.system.Capabilities;

typedef Drawparams = {
  @:optional var scale:Float;
  @:optional var xscale:Float;
  @:optional var yscale:Float;
  @:optional var rotation:Float;
  @:optional var xpivot:Float;
  @:optional var ypivot:Float;
	@:optional var alpha:Float;
	@:optional var red:Float;
	@:optional var green:Float;
	@:optional var blue:Float;
	@:optional var xalign:Int;
	@:optional var yalign:Int;
}

class Gfx {
	public static var LEFT:Int = -20000;
	public static var RIGHT:Int = -20001;
	public static var TOP:Int = -20002;
	public static var BOTTOM:Int = -20003;
	public static var CENTER:Int = -20004;
	
	public static var screenwidth:Int;
	public static var screenheight:Int;
	public static var screenwidthmid:Int;
	public static var screenheightmid:Int;
	
	public static var screenscale:Int;
	public static var devicexres:Int;
	public static var deviceyres:Int;
	public static var fullscreen:Bool;
	
	public static var currenttilesetname:String;
	public static var backbuffer:BitmapData;
	public static var drawto:BitmapData;
	
	/** Create a screen with a given width, height and scale. Also inits Text. */
	public static function resizescreen(width:Float, height:Float, scale:Int = 1) {
		initgfx(Std.int(width), Std.int(height), scale);
		Text.init(gfxstage);
		gfxstage.addChild(screen);
		
		updategraphicsmode();
	}
	
	/** Change the tileset that the draw functions use. */
	public static function changetileset(tilesetname:String) {
		if (currenttilesetname != tilesetname) {
			if(tilesetindex.exists(tilesetname)){
				currenttileset = tilesetindex.get(tilesetname);
				currenttilesetname = tilesetname;
			}else {
				throw("ERROR: Cannot change to tileset \"" + tilesetname + "\", no tileset with that name found.");
			}
		}
	}
	
	public static function numberoftiles():Int {
		return tiles[currenttileset].tiles.length;
	}
	
	/** Creates a blank tileset, with the name "imagename", with each tile a given width and height, containing "amount" tiles. */
	public static function createtiles(imagename:String, width:Float, height:Float, amount:Int) {
		tiles.push(new Tileset(imagename, Std.int(width), Std.int(height)));
		tilesetindex.set(imagename, tiles.length - 1);
		currenttileset = tiles.length - 1;
		
		for (i in 0 ... amount) {
			var t:BitmapData = new BitmapData(Std.int(width), Std.int(height), true, 0x000000);
			tiles[currenttileset].tiles.push(t);
		}
		
		changetileset(imagename);
	}
	
	/** Returns the width of a tile in the current tileset. */
	public static function tilewidth():Int {
		return tiles[currenttileset].width;
	}
	
	/** Returns the height of a tile in the current tileset. */
	public static function tileheight():Int {
		return tiles[currenttileset].height;
	}
	
	/** Creates a blank image, with the name "imagename", with given width and height. */
	public static function createimage(imagename:String, width:Float, height:Float) {
		imageindex.set(imagename, images.length);
		
		var t:BitmapData = new BitmapData(Math.floor(width), Math.floor(height), true, 0x000000);
		images.push(t);
	}
	
	/** Returns the width of the image. */
	public static function imagewidth(imagename:String):Int {
		if(imageindex.exists(imagename)){
			imagenum = imageindex.get(imagename);
		}else {
			throw("ERROR: In imagewidth, cannot find image \"" + imagename + "\".");
			return 0;
		}
		
		return images[imagenum].width;
	}
	
	/** Returns the height of the image. */
	public static function imageheight(imagename:String):Int {
		if(imageindex.exists(imagename)){
			imagenum = imageindex.get(imagename);
		}else {
			throw("ERROR: In imageheight, cannot find image \"" + imagename + "\".");
			return 0;
		}
		
		return images[imagenum].height;
	}
	
	/** Tell draw commands to draw to the actual screen. */
	public static function drawtoscreen() {
		drawingtoscreen = true;
		drawto.unlock();
		drawto = backbuffer;
		drawto.lock();
	}
	
	/** Tell draw commands to draw to the given image. */
	public static function drawtoimage(imagename:String) {
		drawingtoscreen = false;
		imagenum = imageindex.get(imagename);
		
		drawto.unlock();
		drawto = images[imagenum];
		drawto.lock();
	}
	
	/** Tell draw commands to draw to the given tile in the current tileset. */
	public static function drawtotile(tilenumber:Int) {
		drawingtoscreen = false;
		drawto.unlock();
		drawto = tiles[currenttileset].tiles[tilenumber];
		drawto.lock();
	}
	
	/** Helper function for image drawing functions. */
	private static function imagealignx(x:Float):Float {
		if (x == CENTER) return Gfx.screenwidthmid - Std.int(images[imagenum].width / 2);
		if (x == LEFT || x == TOP) return 0;
		if (x == RIGHT || x == BOTTOM) return images[imagenum].width;
		return x;
	}
	
	/** Helper function for image drawing functions. */
	private static function imagealigny(y:Float):Float {
		if (y == CENTER) return Gfx.screenheightmid - Std.int(images[imagenum].height / 2);
		if (y == LEFT || y == TOP) return 0;
		if (y == RIGHT || y == BOTTOM) return images[imagenum].height;
		return y;
	}
	
	/** Helper function for image drawing functions. */
	private static function imagealignonimagex(x:Float):Float {
		if (x == CENTER) return Std.int(images[imagenum].width / 2);
		if (x == LEFT || x == TOP) return 0;
		if (x == RIGHT || x == BOTTOM) return images[imagenum].width;
		return x;
	}
	
	/** Helper function for image drawing functions. */
	private static function imagealignonimagey(y:Float):Float {
		if (y == CENTER) return Std.int(images[imagenum].height / 2);
		if (y == LEFT || y == TOP) return 0;
		if (y == RIGHT || y == BOTTOM) return images[imagenum].height;
		return y;
	}
	
	/** Draws image by name. 
	 * Parameters can be: rotation, scale, xscale, yscale, xpivot, ypivoy, alpha
	 * x and y can be: Gfx.CENTER, Gfx.TOP, Gfx.BOTTOM, Gfx.LEFT, Gfx.RIGHT. 
	 * */
	public static function drawimage(x:Float, y:Float, imagename:String, ?parameters:Drawparams) {
		if (skiprender && drawingtoscreen) return;
		if (!imageindex.exists(imagename)) {
			throw("ERROR: In drawimage, cannot find image \"" + imagename + "\".");
			return;
		}
		imagenum = imageindex.get(imagename);
		
		tempxpivot = 0;
		tempypivot = 0;
		tempxscale = 1.0;
		tempyscale = 1.0;
		temprotate = 0;
		tempred = 1.0; tempgreen = 1.0;	tempblue = 1.0;	tempalpha = 1.0;
		alphact.redMultiplier = 1.0; alphact.greenMultiplier = 1.0;	alphact.blueMultiplier = 1.0;
		alphact.alphaMultiplier = tempalpha;
		changecolours = false;
		tempxalign = x;	tempyalign = y;
		
		x = imagealignx(x); y = imagealigny(y);
		if (parameters != null) {
			if (parameters.xalign != null) {
				if (parameters.xalign == CENTER) {
					if (tempxalign != CENTER) {
						x = x - Std.int(images[imagenum].width / 2);
					}
				}else if (parameters.xalign == BOTTOM || parameters.xalign == RIGHT) {
					if (tempxalign != RIGHT) {
						x = x - Std.int(images[imagenum].width);
					}
				}
			}
			
			if (parameters.yalign != null) {
				if (parameters.yalign == CENTER) {
					if (tempyalign != CENTER) {
						y = y - Std.int(images[imagenum].height / 2);
					}
				}else if (parameters.yalign == BOTTOM || parameters.yalign == RIGHT) {
					if (tempyalign != BOTTOM) {
						y = y - Std.int(images[imagenum].height);
					}
				}
			}
			
			if (parameters.xpivot != null) tempxpivot = imagealignonimagex(parameters.xpivot);
			if (parameters.ypivot != null) tempypivot = imagealignonimagey(parameters.ypivot); 
			if (parameters.scale != null) {
				tempxscale = parameters.scale;
				tempyscale = parameters.scale;
			}else{
				if (parameters.xscale != null) tempxscale = parameters.xscale;
				if (parameters.yscale != null) tempyscale = parameters.yscale;
			}
			if (parameters.rotation != null) temprotate = parameters.rotation;
			if (parameters.alpha != null) {
				tempalpha = parameters.alpha;
				alphact.alphaMultiplier = tempalpha;
				changecolours = true;
			}
			if (parameters.red != null) {
				tempred = parameters.red;
				alphact.redMultiplier = tempred;
				changecolours = true;
			}
			if (parameters.green != null) {
				tempgreen = parameters.green;
				alphact.greenMultiplier = tempgreen;
				changecolours = true;
			}
			if (parameters.blue != null) {
				tempblue = parameters.blue;
				alphact.blueMultiplier = tempblue;
				changecolours = true;
			}
		}
			
		shapematrix.identity();
		shapematrix.translate( -tempxpivot, -tempypivot);
		if (temprotate != 0) shapematrix.rotate((temprotate * 3.1415) / 180);
		if (tempxscale != 1.0 || tempyscale != 1.0) shapematrix.scale(tempxscale, tempyscale);
		shapematrix.translate(x + tempxpivot, y + tempypivot);
		if (changecolours) {
		  drawto.draw(images[imagenum], shapematrix, alphact);	
		}else {
			drawto.draw(images[imagenum], shapematrix);
		}
		shapematrix.identity();
	}
	
	public static function grabtilefromscreen(tilenumber:Int, x:Float, y:Float) {
		if (currenttileset == -1) {
			throw("ERROR: In grabtilefromscreen, there is no tileset currently set. Use Gfx.changetileset(\"tileset name\") to set the current tileset.");
			return;
		}
		
		settrect(x, y, tilewidth(), tileheight());
		tiles[currenttileset].tiles[tilenumber].copyPixels(backbuffer, trect, tl);
	}
	
	public static function grabtilefromimage(tilenumber:Int, imagename:String, x:Float, y:Float) {
		if (!imageindex.exists(imagename)) {
			throw("ERROR: In grabtilefromimage, \"" + imagename + "\" does not exist.");
			return;
		}
		
		if (currenttileset == -1) {
			throw("ERROR: In grabtilefromimage, there is no tileset currently set. Use Gfx.changetileset(\"tileset name\") to set the current tileset.");
			return;
		}
		
		imagenum = imageindex.get(imagename);
		
		settrect(x, y, tilewidth(), tileheight());
		tiles[currenttileset].tiles[tilenumber].copyPixels(images[imagenum], trect, tl);
	}
	
	public static function grabimagefromscreen(imagename:String, x:Float, y:Float) {
		if (!imageindex.exists(imagename)) {
			throw("ERROR: In grabimagefromscreen, \"" + imagename + "\" does not exist. You need to create an image label first before using this function.");
			return;
		}
		imagenum = imageindex.get(imagename);
		
		settrect(x, y, images[imagenum].width, images[imagenum].height);
		images[imagenum].copyPixels(backbuffer, trect, tl);
	}
	
	public static function grabimagefromimage(imagename:String, imagetocopyfrom:String, x:Float, y:Float) {
		if (!imageindex.exists(imagename)) {
			throw("ERROR: In grabimagefromimage, \"" + imagename + "\" does not exist. You need to create an image label first before using this function.");
			return;
		}
		
		imagenum = imageindex.get(imagename);
		if (!imageindex.exists(imagetocopyfrom)) {
			Webdebug.log("ERROR: No image called \"" + imagetocopyfrom + "\" found.");
		}
		var imagenumfrom:Int = imageindex.get(imagetocopyfrom);
		
		settrect(x, y, images[imagenum].width, images[imagenum].height);
		images[imagenum].copyPixels(images[imagenumfrom], trect, tl);
	}
	
	public static function copytile(totilenumber:Int, fromtileset:String, fromtilenumber:Int) {
		if (tilesetindex.exists(fromtileset)) {
			if (tiles[currenttileset].width == tiles[tilesetindex.get(fromtileset)].width && tiles[currenttileset].height == tiles[tilesetindex.get(fromtileset)].height) {
				tiles[currenttileset].tiles[totilenumber].copyPixels(tiles[tilesetindex.get(fromtileset)].tiles[fromtilenumber], tiles[tilesetindex.get(fromtileset)].tiles[fromtilenumber].rect, tl);		
			}else {
				Webdebug.log("ERROR: Tilesets " + currenttilesetname + " (" + Std.string(tilewidth()) + "x" + Std.string(tileheight()) + ") and " + fromtileset + " (" + Std.string(tiles[tilesetindex.get(fromtileset)].width) + "x" + Std.string(tiles[tilesetindex.get(fromtileset)].height) + ") are different sizes. Maybe try just drawing to the tile you want instead with Gfx.drawtotile()?");
				return;
			}
		}else {
			Webdebug.log("ERROR: Tileset " + fromtileset + " hasn't been loaded or created.");
			return;
		}
	}
	
	/** Draws tile number t from current tileset.
	 * Parameters can be: rotation, scale, xscale, yscale, xpivot, ypivoy, alpha
	 * x and y can be: Gfx.CENTER, Gfx.TOP, Gfx.BOTTOM, Gfx.LEFT, Gfx.RIGHT. 
	 * */
	public static function drawtile(x:Float, y:Float, t:Int, ?parameters:Drawparams) {
		if (skiprender && drawingtoscreen) return;
		if (currenttileset == -1) {
			throw("ERROR: No tileset currently set. Use Gfx.changetileset(\"tileset name\") to set the current tileset.");
			return;
		}
		if (t >= numberoftiles()) {
			if (t == numberoftiles()) {
			  throw("ERROR: Tried to draw tile number " + Std.string(t) + ", but there are only " + Std.string(numberoftiles()) + " tiles in tileset \"" + tiles[currenttileset].name + "\". (Because this includes tile number 0, " + Std.string(t) + " is not a valid tile.)");
				return;
			}else{
				throw("ERROR: Tried to draw tile number " + Std.string(t) + ", but there are only " + Std.string(numberoftiles()) + " tiles in tileset \"" + tiles[currenttileset].name + "\".");
				return;
			}
		}
		
		tempxpivot = 0;
		tempypivot = 0;
		tempxscale = 1.0;
		tempyscale = 1.0;
		temprotate = 0;
		tempred = 1.0; tempgreen = 1.0;	tempblue = 1.0;	tempalpha = 1.0;
		alphact.redMultiplier = 1.0; alphact.greenMultiplier = 1.0;	alphact.blueMultiplier = 1.0;
		alphact.alphaMultiplier = tempalpha;
		changecolours = false;
		tempxalign = x;	tempyalign = y;
		
		x = tilealignx(x); y = tilealigny(y);
		if (parameters != null) {
			if (parameters.xalign != null) {
				if (parameters.xalign == CENTER) {
					if (tempxalign != CENTER) {
						x = x - Std.int(tilewidth() / 2);
					}
				}else if (parameters.xalign == BOTTOM || parameters.xalign == RIGHT) {
					if (tempxalign != RIGHT) {
						x = x - Std.int(tilewidth());
					}
				}
			}
			
			if (parameters.yalign != null) {
				if (parameters.yalign == CENTER) {
					if (tempyalign != CENTER) {
						y = y - Std.int(tileheight() / 2);
					}
				}else if (parameters.yalign == BOTTOM || parameters.yalign == RIGHT) {
					if (tempyalign != BOTTOM) {
						y = y - Std.int(tileheight());
					}
				}
			}
			
			if (parameters.xpivot != null) tempxpivot = tilealignontilex(parameters.xpivot);
			if (parameters.ypivot != null) tempypivot = tilealignontiley(parameters.ypivot); 
			
			if (parameters.scale != null) {
				tempxscale = parameters.scale;
				tempyscale = parameters.scale;
			}else{
				if (parameters.xscale != null) tempxscale = parameters.xscale;
				if (parameters.yscale != null) tempyscale = parameters.yscale;
			}
			
			if (parameters.rotation != null) temprotate = parameters.rotation;
			if (parameters.alpha != null) {
				tempalpha = parameters.alpha;
				alphact.alphaMultiplier = tempalpha;
				changecolours = true;
			}
			if (parameters.red != null) {
				tempred = parameters.red;
				alphact.redMultiplier = tempred;
				changecolours = true;
			}
			if (parameters.green != null) {
				tempgreen = parameters.green;
				alphact.greenMultiplier = tempgreen;
				changecolours = true;
			}
			if (parameters.blue != null) {
				tempblue = parameters.blue;
				alphact.blueMultiplier = tempblue;
				changecolours = true;
			}
		}
		
		shapematrix.identity();
		shapematrix.translate( -tempxpivot, -tempypivot);
		if (temprotate != 0) shapematrix.rotate((temprotate * 3.1415) / 180);
		if (tempxscale != 1.0 || tempyscale != 1.0) shapematrix.scale(tempxscale, tempyscale);
		shapematrix.translate(tempxpivot, tempypivot);
		shapematrix.translate(x, y);
		if (changecolours) {
		  drawto.draw(tiles[currenttileset].tiles[t], shapematrix, alphact);
		}else {
		  drawto.draw(tiles[currenttileset].tiles[t], shapematrix);
		}
		shapematrix.identity();
	}
	
	/** Returns the current animation frame of the current tileset. */
	public static function currentframe():Int {
		return tiles[currenttileset].currentframe;
	}
	
	/** Resets the animation. */
	public static function stopanimation(animationname:String) {
		animationnum = animationindex.get(animationname);
		animations[animationnum].reset();
	}
	
	public static function defineanimation(animationname:String, tileset:String, startframe:Int, endframe:Int, delayperframe:Int) {
		if (delayperframe < 1) {
			throw("ERROR: Cannot have a delay per frame of less than 1.");
			return;
		}
		animationindex.set(animationname, animations.length);
		animations.push(new AnimationContainer(animationname, tileset, startframe, endframe, delayperframe));
	}
	
	public static function drawanimation(x:Float, y:Float, animationname:String, ?parameters:Drawparams) {
		if (skiprender && drawingtoscreen) return;
		oldtileset = currenttilesetname;
		if (!animationindex.exists(animationname)) {
			throw("ERROR: No animated named \"" +animationname+"\" is defined. Define one first using Gfx.defineanimation!");
			return;
		}
		animationnum = animationindex.get(animationname);
		changetileset(animations[animationnum].tileset);
		
		animations[animationnum].update();
		tempframe = animations[animationnum].currentframe;
		
		if (parameters != null) {
		  drawtile(x, y, tempframe, parameters);
		}else {
			drawtile(x, y, tempframe);
		}
		
		changetileset(oldtileset);
	}
	
	private static function tilealignx(x:Float):Float {
		if (x == CENTER) return Gfx.screenwidthmid - Std.int(tiles[currenttileset].width / 2);
		if (x == LEFT || x == TOP) return 0;
		if (x == RIGHT || x == BOTTOM) return tiles[currenttileset].width;
		return x;
	}
	
	private static function tilealigny(y:Float):Float {
		if (y == CENTER) return Gfx.screenheightmid - Std.int(tiles[currenttileset].height / 2);
		if (y == LEFT || y == TOP) return 0;
		if (y == RIGHT || y == BOTTOM) return tiles[currenttileset].height;
		return y;
	}
	
	private static function tilealignontilex(x:Float):Float {
		if (x == CENTER) return Std.int(tiles[currenttileset].width / 2);
		if (x == LEFT || x == TOP) return 0;
		if (x == RIGHT || x == BOTTOM) return tiles[currenttileset].width;
		return x;
	}
	
	private static function tilealignontiley(y:Float):Float {
		if (y == CENTER) return Std.int(tiles[currenttileset].height / 2);
		if (y == LEFT || y == TOP) return 0;
		if (y == RIGHT || y == BOTTOM) return tiles[currenttileset].height;
		return y;
	}
	
	public static function drawline(x1:Float, y1:Float, x2:Float, y2:Float, col:Int, alpha:Float = 1.0) {
    if (skiprender && drawingtoscreen) return;
    tempshape.graphics.clear();
		tempshape.graphics.lineStyle(linethickness, col, alpha);
		tempshape.graphics.moveTo(x1,y1);
    tempshape.graphics.lineTo(x2, y2);
    drawto.draw(tempshape, shapematrix);
	}

	public static function drawhexagon(x:Float, y:Float, radius:Float, angle:Float, col:Int, alpha:Float = 1.0) {
		if (skiprender && drawingtoscreen) return;
		tempshape.graphics.clear();
		tempshape.graphics.lineStyle(linethickness, col, alpha);
		
		temprotate = ((Math.PI * 2) / 6);
		
		tx = (Math.cos(angle) * radius);
		ty = (Math.sin(angle) * radius);
		
		tempshape.graphics.moveTo(tx, ty);
		for (i in 0 ... 7) {
			tx = (Math.cos(angle + (temprotate * i)) * radius);
		  ty = (Math.sin(angle + (temprotate * i)) * radius);
			
			tempshape.graphics.lineTo(tx, ty);
		}
		
		shapematrix.translate(x, y);
		drawto.draw(tempshape, shapematrix);
		shapematrix.translate(-x, -y);
	}
	
	public static function fillhexagon(x:Float, y:Float, radius:Float, angle:Float, col:Int, alpha:Float = 1.0) {
		if (skiprender && drawingtoscreen) return;
		tempshape.graphics.clear();
		temprotate = ((Math.PI * 2) / 6);
		
		tx = (Math.cos(angle) * radius);
		ty = (Math.sin(angle) * radius);
		
		tempshape.graphics.moveTo(tx, ty);
		tempshape.graphics.beginFill(col, alpha);
		for (i in 0 ... 7) {
			tx = (Math.cos(angle + (temprotate * i)) * radius);
		  ty = (Math.sin(angle + (temprotate * i)) * radius);
			
			tempshape.graphics.lineTo(tx, ty);
		}
		tempshape.graphics.endFill();
		
		shapematrix.translate(x, y);
		drawto.draw(tempshape, shapematrix);
		shapematrix.translate(-x, -y);
	}
	
	public static function drawcircle(x:Float, y:Float, radius:Float, col:Int, alpha:Float = 1.0) {
		if (skiprender && drawingtoscreen) return;
		tempshape.graphics.clear();
		tempshape.graphics.lineStyle(linethickness, col, alpha);
		tempshape.graphics.drawCircle(0, 0, radius);
		
		shapematrix.translate(x, y);
		drawto.draw(tempshape, shapematrix);
		shapematrix.translate(-x, -y);
	}
	
	public static function fillcircle(x:Float, y:Float, radius:Float, col:Int, alpha:Float = 1.0) {
		if (skiprender && drawingtoscreen) return;
		tempshape.graphics.clear();
		tempshape.graphics.beginFill(col, alpha);
		tempshape.graphics.drawCircle(0, 0, radius);
		tempshape.graphics.endFill();
		
		shapematrix.translate(x, y);
		drawto.draw(tempshape, shapematrix);
		shapematrix.translate(-x, -y);
	}
	
	public static function drawtri(x1:Float, y1:Float, x2:Float, y2:Float, x3:Float, y3:Float, col:Int, alpha:Float = 1.0) {
		if (skiprender && drawingtoscreen) return;
		tempshape.graphics.clear();
		tempshape.graphics.lineStyle(linethickness, col, alpha);
		tempshape.graphics.lineTo(0, 0);
		tempshape.graphics.lineTo(x2 - x1, y2 - y1);
		tempshape.graphics.lineTo(x3 - x1, y3 - y1);
		tempshape.graphics.lineTo(0, 0);
		tempshape.graphics.endFill();
		
		shapematrix.translate(x1, y1);
		drawto.draw(tempshape, shapematrix);
		shapematrix.translate(-x1, -y1);
	}
	
	public static function filltri(x1:Float, y1:Float, x2:Float, y2:Float, x3:Float, y3:Float, col:Int, alpha:Float = 1.0) {
		if (skiprender && drawingtoscreen) return;
		tempshape.graphics.clear();
		tempshape.graphics.beginFill(col, alpha);
		tempshape.graphics.lineTo(0, 0);
		tempshape.graphics.lineTo(x2 - x1, y2 - y1);
		tempshape.graphics.lineTo(x3 - x1, y3 - y1);
		tempshape.graphics.lineTo(0, 0);
		tempshape.graphics.endFill();
		
		
		shapematrix.translate(x1, y1);
		drawto.draw(tempshape, shapematrix);
		shapematrix.translate(-x1, -y1);
	}

	public static function drawbox(x:Float, y:Float, width:Float, height:Float, col:Int, alpha:Float = 1.0) {
		if (skiprender && drawingtoscreen) return;
		if (width < 0) {
			width = -width;
			x = x - width;
		}
		if (height < 0) {
			height = -height;
			y = y - height;
		}
		
		if (linethickness < 2) {				
			drawline(x, y, x + width, y, col, alpha);
			drawline(x, y + height, x + width, y + height, col, alpha);
			drawline(x, y + 1, x, y + height, col, alpha);
			drawline(x + width - 1, y + 1, x + width - 1, y + height, col, alpha);
		}else{
			tempshape.graphics.clear();
			tempshape.graphics.lineStyle(linethickness, col, alpha);
			tempshape.graphics.lineTo(width, 0);
			tempshape.graphics.lineTo(width, height);
			tempshape.graphics.lineTo(0, height);
			tempshape.graphics.lineTo(0, 0);
			
			shapematrix.translate(x, y);
			drawto.draw(tempshape, shapematrix);
			shapematrix.translate( -x, -y);
		}
	}

	public static function setlinethickness(size:Float) {
		linethickness = size;
		if (linethickness < 1) linethickness = 1;
		if (linethickness > 255) linethickness = 255;
	}
	
	public static function clearscreen(col:Int = 0x000000) {
		if (skiprender && drawingtoscreen) return;
		backbuffer.fillRect(backbuffer.rect, col);
	}
	
	public static function getpixel(x:Float, y:Float):Int {
		return drawto.getPixel32(Std.int(x), Std.int(y));
	}

	public static function fillbox(x:Float, y:Float, width:Float, height:Float, col:Int, alpha:Float = 1.0) {
		if (skiprender && drawingtoscreen) return;
		tempshape.graphics.clear();
		tempshape.graphics.beginFill(col, alpha);
		tempshape.graphics.lineTo(width, 0);
		tempshape.graphics.lineTo(width, height);
		tempshape.graphics.lineTo(0, height);
		tempshape.graphics.lineTo(0, 0);
		tempshape.graphics.endFill();
		
		shapematrix.translate(x, y);
		drawto.draw(tempshape, shapematrix);
		shapematrix.translate(-x, -y);
	}
	
	public static function getred(c:Int):Int {
		return (( c >> 16 ) & 0xFF);
	}
	
	public static function getgreen(c:Int):Int {
		return ( (c >> 8) & 0xFF );
	}
	
	public static function getblue(c:Int):Int {
		return ( c & 0xFF );
	}
	
	public static function RGB(red:Int, green:Int, blue:Int):Int {
		return (blue | (green << 8) | (red << 16));
	}
	
	/** Picks a colour given Hue, Saturation and Lightness values. 
	 *  Hue is between 0-359, Saturation and Lightness between 0.0 and 1.0. */
	public static function HSL(hue:Float, saturation:Float, lightness:Float):Int{
		var q:Float = if (lightness < 1 / 2) {
			lightness * (1 + saturation);
		}else {
			lightness + saturation - (lightness * saturation);
		}
		
		var p:Float = 2 * lightness - q;
		
		var hk:Float = ((hue % 360) / 360);
		
		hslval[0] = hk + 1 / 3;
		hslval[1] = hk;
		hslval[2] = hk - 1 / 3;
		for (n in 0 ... 3){
			if (hslval[n] < 0) hslval[n] += 1;
			if (hslval[n] > 1) hslval[n] -= 1;
			hslval[n] = if (hslval[n] < 1 / 6){
				p + ((q - p) * 6 * hslval[n]);
			}else if (hslval[n] < 1 / 2)	{
				q;
			}else if (hslval[n] < 2 / 3){
				p + ((q - p) * 6 * (2 / 3 - hslval[n]));
			}else{
				p;
			}
		}
		
		return RGB(Std.int(hslval[0] * 255), Std.int(hslval[1] * 255), Std.int(hslval[2] * 255));
	}
	
	private static function setzoom(t:Int) {
		screen.width = screenwidth * t;
		screen.height = screenheight * t;
		screen.x = (screenwidth - (screenwidth * t)) / 2;
		screen.y = (screenheight - (screenheight * t)) / 2;
	}
	
	private static function updategraphicsmode() {
		if (fullscreen) {
			Lib.current.stage.displayState = StageDisplayState.FULL_SCREEN_INTERACTIVE;
			gfxstage.scaleMode = StageScaleMode.NO_SCALE;
			
			var xScaleFresh:Float = cast(devicexres, Float) / cast(screenwidth, Float);
			var yScaleFresh:Float = cast(deviceyres, Float) / cast(screenheight, Float);
			if (xScaleFresh < yScaleFresh){
				screen.width = screenwidth * xScaleFresh;
				screen.height = screenheight * xScaleFresh;
			}else if (yScaleFresh < xScaleFresh){
				screen.width = screenwidth * yScaleFresh;
				screen.height = screenheight * yScaleFresh;
			} else {
				screen.width = screenwidth * xScaleFresh;
				screen.height = screenheight * yScaleFresh;
			}
			screen.x = (cast(devicexres, Float) / 2.0) - (screen.width / 2.0);
			screen.y = (cast(deviceyres, Float) / 2.0) - (screen.height / 2.0);
			//Mouse.hide();
		}else {
			Lib.current.stage.displayState = StageDisplayState.NORMAL;
			screen.width = screenwidth * screenscale;
			screen.height = screenheight * screenscale;
			screen.x = 0.0;
			screen.y = 0.0;
			gfxstage.scaleMode = StageScaleMode.SHOW_ALL;
			gfxstage.quality = StageQuality.HIGH;
		}
	}
	
	/** Just gives Gfx access to the stage. */
	private static function init(stage:Stage) {
		gfxstage = stage;
		setlinethickness(1);
	}
	
	/** Called from resizescreen(). Sets up all our graphics buffers. */
	private static function initgfx(width:Int, height:Int, scale:Int) {
		//We initialise a few things
		screenwidth = width; screenheight = height;
		screenwidthmid = Std.int(screenwidth / 2); screenheightmid = Std.int(screenheight / 2);
		
		devicexres = Std.int(flash.system.Capabilities.screenResolutionX);
		deviceyres = Std.int(flash.system.Capabilities.screenResolutionY);
		screenscale = scale;
		
		trect = new Rectangle(); tpoint = new Point();
		tbuffer = new BitmapData(1, 1, true);
		ct = new ColorTransform(0, 0, 0, 1, 255, 255, 255, 1); //Set to white
		alphact = new ColorTransform();
		hslval.push(0.0); hslval.push(0.0); hslval.push(0.0);
		
		backbuffer = new BitmapData(screenwidth, screenheight, false, 0x000000);
		drawto = backbuffer;
		drawingtoscreen = true;
		
		screen = new Bitmap(backbuffer);
		screen.width = screenwidth * scale;
		screen.height = screenheight * scale;
		
		fullscreen = false;
	}
	
	/** Sets the values for the temporary rect structure. Probably better than making a new one, idk */
	private static function settrect(x:Float, y:Float, w:Float, h:Float) {
		trect.x = x;
		trect.y = y;
		trect.width = w;
		trect.height = h;
	}
	
	/** Sets the values for the temporary point structure. Probably better than making a new one, idk */
	private static function settpoint(x:Float, y:Float) {
		tpoint.x = x;
		tpoint.y = y;
	}
	
	private static var tiles:Array<Tileset> = new Array<Tileset>();
	private static var tilesetindex:Map<String, Int> = new Map<String, Int>();
	private static var currenttileset:Int = -1;
	
	private static var animations:Array<AnimationContainer> = new Array<AnimationContainer>();
	private static var animationnum:Int;
	private static var animationindex:Map<String, Int> = new Map<String, Int>();
	
	private static var images:Array<BitmapData> = new Array<BitmapData>();
	private static var imagenum:Int;
	private static var ct:ColorTransform;
	private static var alphact:ColorTransform;
	private static var images_rect:Rectangle;
	private static var tl:Point = new Point(0, 0);
	private static var trect:Rectangle;
	private static var tpoint:Point;
	private static var tbuffer:BitmapData;
	private static var imageindex:Map<String, Int> = new Map<String, Int>();
	
	private static var temprotate:Float;
	private static var tempxscale:Float;
	private static var tempyscale:Float;
	private static var tempxpivot:Float;
	private static var tempypivot:Float;
	private static var tempalpha:Float;
	private static var tempred:Float;
	private static var tempgreen:Float;
	private static var tempblue:Float;
	private static var tempframe:Int;
	private static var tempxalign:Float;
	private static var tempyalign:Float;
	private static var changecolours:Bool;
	private static var oldtileset:String;
	private static var tx:Float;
	private static var ty:Float;
	
	private static var linethickness:Float;
	
	private static var buffer:BitmapData;
	
	private static var temptile:BitmapData;
	//Actual backgrounds
	private static var screen:Bitmap;
	private static var tempshape:Shape = new Shape();
	private static var shapematrix:Matrix = new Matrix();
	
	private static var alphamult:Int;
	private static var gfxstage:Stage;
	
	//HSL conversion variables 
	private static var hslval:Array<Float> = new Array<Float>();
	
	private static var skiprender:Bool;
	private static var drawingtoscreen:Bool;
}