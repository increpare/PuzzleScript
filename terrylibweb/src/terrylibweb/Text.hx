package terrylibweb;

import terrylibweb.util.*;
import openfl.Assets;
import openfl.display.*;
import openfl.geom.*;
import openfl.events.*;
import openfl.net.*;
import openfl.text.*;

typedef Drawparamstext = {
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
	@:optional var align:Int;
}

@:access(terrylibweb.Gfx)
class Text {
	public static function init(stage:Stage) {
		drawto = Gfx.backbuffer;
		gfxstage = stage;
		enabletextfield();
		alphact = new ColorTransform();
		input_cursorglow = 0;
	}
	
	//Text Input functions
	
	private static function enabletextfield() {
		gfxstage.addChild(inputField);
		inputField.border = true;
		inputField.width = Gfx.screenwidth;
		inputField.height = 20;
		inputField.x = 0;
		inputField.y = Gfx.screenheight + 10;
		inputField.type = TextFieldType.INPUT;
		inputField.visible = false;
		
		inputField.maxChars = 80;
		
		resetinput("");
	}
	
	private static function input_checkfortext() {
		gfxstage.focus = inputField;
		#if flash
		inputField.setSelection(inputField.text.length, inputField.text.length);
		#else
		inputField.setSelection(inputField.text.length, inputField.text.length);
		#end
		inputtext = inputField.text;
	}
	
	/** Return characters from the middle of a string. */
	private static function mid(s:String, start:Int = 0, length:Int = 1):String {
		return s.substr(start,length);
	}
	
	/** Reverse a string. */
	private static function reversetext(t:String):String {
		var t2:String = "";
		
		for (i in 0...t.length) {
			t2 += mid(t, t.length-i-1, 1);
		}
		return t2;
	}
	
	public static function resetinput(t:String) {
		#if flash
		inputField.text = t; inputtext = t;
		#else
		inputField.text = reversetext(t); inputtext = reversetext(t);
		#end
		input_show = 0;
	}
	
	public static function input(x:Float, y:Float, text:String, col:Int = 0xFFFFFF, responsecol:Int = 0xCCCCCC):Bool {
		input_show = 2;
		
		input_font = currentfont;
		input_textsize = currentsize;
		typeface[currentindex].tf.text = text + inputtext;
		x = alignx(x); y = aligny(y);
		input_textxp = x;
		input_textyp = y;
		
		typeface[currentindex].tf.text = text;
		input_responsexp = input_textxp + Math.floor(typeface[currentindex].tf.textWidth);
		input_responseyp = y;
		
		input_text = text;
		input_response = inputtext;
		input_textcol = col;
		input_responsecol = responsecol;
		input_checkfortext();
		
		if (Input.justpressed(Key.ENTER) && inputtext != "") {
			return true;
		}
		return false;
	}
	
	/** Returns the entered string, and resets the input for next time. */
	public static function getinput():String {
		var response:String = inputtext;
		lastentry = inputtext;
		inputtext = "";
		inputField.text = "";
		input_show = 0;
		
		return response;
	}
	
	public static function drawstringinput() {
		if (input_show > 0) {
			Text.changefont(input_font);
			Text.changesize(input_textsize);
			input_cursorglow++;
			if (input_cursorglow >= 96) input_cursorglow = 0;
			
			display(input_textxp, input_textyp, input_text, input_textcol);
			if (input_cursorglow % 48 < 24) {
				display(input_responsexp, input_responseyp, input_response, input_responsecol);
			}else {
				display(input_responsexp, input_responseyp, input_response + "_", input_responsecol);
			}
		}
		
		input_show--;
		if (input_show < 0) input_show = 0;
	}
	
	//Text display functions
	public static function len(t:String):Float {
		typeface[currentindex].tf.text = t;
		return typeface[currentindex].tf.textWidth;
	}
	
	public static function height():Float {
		typeface[currentindex].tf.text = "???";
		return typeface[currentindex].tf.textHeight;
	}
	
	private static function alignx(x:Float):Float {
		if (x == CENTER) return Math.floor(Gfx.screenwidthmid - (typeface[currentindex].tf.textWidth / 2));
		if (x == LEFT || x == TOP) return 0;
		if (x == RIGHT || x == BOTTOM) return Math.floor(Gfx.screenwidth - (typeface[currentindex].tf.textWidth));
		
		return x;
	}
	
	private static function aligny(y:Float):Float {
		if (y == CENTER) return Math.floor(Gfx.screenheightmid - (typeface[currentindex].tf.textHeight / 2));
		if (y == LEFT || y == TOP) return 0;
		if (y == RIGHT || y == BOTTOM) return Math.floor(Gfx.screenheight - (typeface[currentindex].tf.textHeight));
		
		return y;
	}
	
	private static function aligntextx(t:String, x:Float):Float {
		if (x == CENTER) return Math.floor(len(t) / 2);
		if (x == LEFT || x == TOP) return 0;
		if (x == RIGHT || x == BOTTOM) return len(t);
		return x;
	}
	
	private static function aligntexty(y:Float):Float {
		if (y == CENTER) return Math.floor(height() / 2);
		if (y == TOP || y == LEFT) return 0;
		if (y == BOTTOM || y == RIGHT) return height();
		return y;
	}
	
	public static function display(x:Float, y:Float, text:String, col:Int = 0xFFFFFF, ?parameters:Drawparamstext) {
		// This was called "print" once. Maybe it was better that way? eh, stuck with display now
		if (Gfx.skiprender && Gfx.drawingtoscreen) return;
		if (parameters == null) {
			typeface[currentindex].tf.textColor = col;
			typeface[currentindex].tf.text = text;
			
			x = alignx(x); y = aligny(y);
			
			fontmatrix.identity();
			fontmatrix.translate(x, y);
			typeface[currentindex].tf.textColor = col;
			drawto.draw(typeface[currentindex].tf, fontmatrix);
		}else {
			drawto = typeface[currentindex].tfbitmap;
			typeface[currentindex].clearbitmap();
			
			tempxpivot = 0;
			tempypivot = 0;
			tempxscale = 1.0;
			tempyscale = 1.0;
			temprotate = 0;
			tempalpha = 1.0;
			tempred = 1.0; tempgreen = 1.0; tempblue = 1.0;
			changecolours = false;
			
			display(0, 0, text, col);
			
			x = alignx(x); y = aligny(y);
			
			if (parameters.align != null) {
				if (parameters.align == CENTER) {
					x = x - (len(text) / 2);
				}else if (parameters.align == RIGHT || parameters.align == BOTTOM) {
					x = x - len(text);
				}
			}
			
			if (parameters.xpivot != null) tempxpivot = aligntextx(text, parameters.xpivot);
		  if (parameters.ypivot != null) tempypivot = aligntexty(parameters.ypivot);		
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
			
			fontmatrix.identity();
			fontmatrix.translate(-tempxpivot, -tempypivot);
			fontmatrix.scale(tempxscale, tempyscale);
			fontmatrix.rotate((temprotate * 3.1415) / 180);
			fontmatrix.translate(x + tempxpivot, y + tempypivot);
			drawto = Gfx.drawto;
			if (changecolours) {
				drawto.draw(typeface[currentindex].tfbitmap, fontmatrix, alphact);
			}else {
			  drawto.draw(typeface[currentindex].tfbitmap, fontmatrix);	
			}
		}
	}
	
	public static function changefont(t:String) {
		if(t != currentfont){
			currentfont = t;
			if (currentsize != -1) {
				if (typefaceindex.exists(currentfont + "_" + Std.string(currentsize))) {
					currentindex = typefaceindex.get(currentfont + "_" + Std.string(currentsize));
				}else {
					addtypeface(currentfont, currentsize);
					currentindex = typefaceindex.get(currentfont + "_" + Std.string(currentsize));
				}
			}
		}
	}
	
	public static function changesize(t:Int) {
		if (t != currentsize){
			currentsize = t;
			if (currentfont != "null") {
				if (typefaceindex.exists(currentfont + "_" + Std.string(currentsize))) {
					currentindex = typefaceindex.get(currentfont + "_" + Std.string(currentsize));
				}else {
					addtypeface(currentfont, currentsize);
					currentindex = typefaceindex.get(currentfont + "_" + Std.string(currentsize));
				}
			}
		}
	}
	
	public static function addfont(t:String, defaultsize:Int) {
		fontfile.push(new Fontfile(t));
		fontfileindex.set(t, fontfile.length - 1);
		currentfont = t;
		
		changesize(defaultsize);
	}
	
	private static function addtypeface(_name:String, _size:Int) {
		typeface.push(new Fontclass(_name, _size));
		typefaceindex.set(_name+"_" + Std.string(_size), typeface.length - 1);
	}
	
	/** Return a font's internal TTF name. Used for loading in fonts during setup. */
	public static function getfonttypename(fontname:String):String {
		return fontfile[Text.fontfileindex.get(fontname)].typename;
	}
	
	private static var fontfile:Array<Fontfile> = new Array<Fontfile>();
	private static var fontfileindex:Map<String,Int> = new Map<String,Int>();
	
	private static var typeface:Array<Fontclass> = new Array<Fontclass>();
	private static var typefaceindex:Map<String,Int> = new Map<String,Int>();
	
	private static var fontmatrix:Matrix = new Matrix();
	private static var currentindex:Int = -1;
	public static var currentfont:String = "null";
	public static var currentsize:Int = -1;
	private static var gfxstage:Stage;
	
	public static var drawto:BitmapData;
	
	public static var LEFT:Int = -20000;
	public static var RIGHT:Int = -20001;
	public static var TOP:Int = -20002;
	public static var BOTTOM:Int = -20003;
	public static var CENTER:Int = -20004;
	
	private static var temprotate:Float;
	private static var tempxscale:Float;
	private static var tempyscale:Float;
	private static var tempxpivot:Float;
	private static var tempypivot:Float;
	private static var tempalpha:Float;
	private static var tempred:Float;
	private static var tempgreen:Float;
	private static var tempblue:Float;
	private static var changecolours:Bool;
	private static var alphact:ColorTransform;
	
	//Text input variables
	private static var inputField:TextField = new TextField();
	private static var inputtext:String;
	private static var lastentry:String;
	
	private static var input_textxp:Float;
	private static var input_textyp:Float;
	private static var input_responsexp:Float;
	private static var input_responseyp:Float;
	private static var input_textcol:Int;
	private static var input_responsecol:Int;
	private static var input_text:String;
	private static var input_response:String;
	private static var input_cursorglow:Int;
	private static var input_font:String;
	private static var input_textsize:Int;
	/** Non zero when an input string is being checked. So that I can use 
	 * the M and F keys without muting or changing to fullscreen.*/
	public static var input_show:Int;
}