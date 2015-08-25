import terrylib.*;

@:expose
class Webbridge {
	public function runScript(s:String) {
		Webscript.loadscript(s);
  }
	
	public function get_background_colour():Int {
		return Webscript.background_color;
	}
	
	public function get_title():String {
		return Webscript.title;
	}
	
	public function get_homepage():String {
		return Webscript.homepage;
	}
	
	private var functionlist:String;
	private function addf(f:String) {
		functionlist += f + "\n";
	}
	
	public function get_functions():String {
		functionlist = "";
		addf("Gfx:");
		addf("----------");
		addf("Gfx.resizescreen(width, height, scale);");
		addf("Gfx.clearscreen(color);");
		addf("Gfx.drawbox(x, y, width, height, col);");
		addf("Gfx.fillbox(x, y, width, height, col);");
		addf("Gfx.drawtri(x1, y1, x2, y2, x3, y3, col);");
		addf("Gfx.filltri(x1, y1, x2, y2, x3, y3, col);");
		addf("Gfx.drawcircle(x, y, radius, col);");
		addf("Gfx.fillcircle(x, y, radius, col);");
		addf("Gfx.drawhexagon(x, y, radius, angle, col);");
		addf("Gfx.fillhexagon(x, y, radius, angle, col);");
		addf("Gfx.drawline(x1, y1, x2, y2, col);");
		addf("Gfx.setlinethickness(linethickness);");
		addf("var p:Int = Gfx.getpixel(x, y);");
		addf("var col:Int = Gfx.RGB(red, green, blue);");
		addf("var col:Int = Gfx.HSL(hue, saturation, lightness);");
		addf("var redvalue:Int = Gfx.getred(col);");
		addf("var greenvalue:Int = Gfx.getgreen(col);");
		addf("var bluevalue:Int = Gfx.getblue(col);");
		addf("t = Gfx.screenwidth;");
		addf("t = Gfx.screenheight;");
		addf("Gfx.drawimage(x, y, \"imagename\", optional parameters);");
		addf("Gfx.changetileset(\"newtileset\");");
		addf("Gfx.drawtile(x, y, tilenumber, optional parameters);");
		addf("t = Gfx.imagewidth(\"imagename\");");
		addf("t = Gfx.imageheight(\"imagename\");");
		addf("t = Gfx.tilewidth();");
		addf("t = Gfx.tileheight();");
		addf("Gfx.createimage(\"imagename\", width, height); ");
		addf("Gfx.createtiles(\"imagename\", width, height, amount);");
		addf("t = Gfx.numberoftiles();");
		addf("Gfx.drawtoscreen();");
		addf("Gfx.drawtoimage(\"imagename\");");
		addf("Gfx.drawtotile(tilenumber);");
		addf("Gfx.copytile(tilenumber in current tileset, \"tileset to copy from\", tilenumber in other tileset);");
		addf("Gfx.grabtilefromscreen(tilenumber, screen x pos, screen y pos);");
		addf("Gfx.grabtilefromimage(tilenumber, \"imagename\", x position in image, y position in image);");
		addf("Gfx.grabimagefromscreen(\"imagename\", x position in screen, y position in screen);");
		addf("Gfx.grabimagefromimage(\"imagename\", \"imagetocopyfrom\", x position in image, y position in image);");
		addf("Gfx.defineanimation(\"animationname\", \"tileset\", start frame, end frame, delay per frame);");
		addf("Gfx.drawanimation(x, y, \"animation name\", optional parameters);");
		addf("Gfx.stopanimation(\"animation name\");");
		addf("");
		
		addf("Col:");
		addf("----------");
		addf("Col.BLACK");
		addf("Col.GREY");
		addf("Col.WHITE");
		addf("Col.RED");
		addf("Col.PINK");
		addf("Col.DARKBROWN");
		addf("Col.BROWN");
		addf("Col.ORANGE");
		addf("Col.YELLOW");
		addf("Col.DARKGREEN");
		addf("Col.GREEN");
		addf("Col.LIGHTGREEN");
		addf("Col.NIGHTBLUE");
		addf("Col.DARKBLUE"); 
		addf("Col.BLUE");
		addf("Col.LIGHTBLUE");
		addf("Col.MAGENTA");
		addf("");
		
		addf("Col:");
		addf("----------");
		
		addf("Text:");
		addf("----------");
		addf("Text.changefont(\"fontname\");");
		addf("Text.changesize(fontsize);");
		addf("Text.display(x, y, \"thing to display\", col, optional parameters);");
		addf("var waitforreply:Bool = Text.input(x, y, \"Question: \", question_colour, answer_colour)");
		addf("var response:String = Text.getinput()");
		
		
		addf("Music:");
		addf("----------");
		addf("Music.playsound(integer to generate from)");
		
		addf("Input:");
		addf("----------");
		addf("Key.A ... Key.Z");
		addf("Key.ZERO ... Key.NINE");
		addf("Key.F1 ... Key.F12");
		addf("Key.MINUS, Key.PLUS, Key.DELETE, Key.BACKSPACE, Key.LBRACKET");
		addf("Key.RBRACKET, Key.BACKSLASH, Key.CAPSLOCK, Key.SEMICOLON");
		addf("Key.QUOTE, Key.COMMA, Key.PERIOD, Key.SLASH");
		addf("Key.ESCAPE, Key.ENTER, Key.SHIFT");
		addf("Key.CONTROL, Key.ALT, Key.SPACE");
		addf("Key.UP, Key.DOWN, Key.LEFT, Key.RIGHT");
		addf("Input.justpressed(Key.ENTER);");
		addf("Input.pressed(Key.LEFT);");
		addf("Input.justreleased(Key.SPACE);");
		addf("Input.delaypressed(Key.Z, 5);");
		addf("Mouse.x");
		addf("Mouse.y");
		addf("Mouse.leftclick()");
		addf("Mouse.leftheld()");
		addf("Mouse.leftreleased()");
		addf("Mouse.middleclick()");
		addf("Mouse.middleheld()");
		addf("Mouse.middlereleased()");
		addf("Mouse.rightclick()");
		addf("Mouse.rightheld()");
		addf("Mouse.rightreleased()");
		addf("var wheel:Int = Mouse.mousewheel");
		
		addf("Other:");
		addf("----------");
		addf("Convert.tostring(1234);");
		addf("Convert.toint(\"15\");");
		addf("Convert.tofloat(\"3.1417826\");");
		addf("Random.int(from, to);");
		addf("Random.float(from, to);");
		addf("Random.string(length);");
		addf("Random.bool();");
		addf("Random.occasional();");
		addf("Random.rare();");
		addf("Random.pickstring(\"this one\", \"or this one?\", \"maybe this one?\");");
		addf("Random.pickint(5, 14, 72, 92, 1, -723, 8);");
		addf("Random.pickfloat(5.1, 14.2, 72.3, 92.4, 1.5, -723.6, 8.7);");
		addf("Debug.log(message)");
		
		return functionlist;
	}
}

class Main {
	public function new() {
		Webscript.init();
	}
	
	public function update() {
		Webscript.update();
  }
}