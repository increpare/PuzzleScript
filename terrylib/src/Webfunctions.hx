import terrylib.*;
import openfl.external.ExternalInterface;

class Webfunctions {
	//This class is shared with hscript. Any external functions you want to use
	//need to be implemented here.
	public static var runlocally:Bool = true;
	
	public static function cls() {
		if (runlocally) {
			Gfx.clearscreen();
		}else{
			ExternalInterface.call("cls");
		}
	}
	
	public static function random(start:Int, end:Int):Int {
		return Random.int(start, end);
		/*
		if (runlocally) {
			Gfx.clearscreen();
		}else{
			ExternalInterface.call("cls");
		}
		*/
	}
	
	public static function fillrect(x:Int, y:Int, w:Int, h:Int, col:Int) {
		if (runlocally) {
			Gfx.fillbox(x, y, w, h, col);
		}else{
			ExternalInterface.call("fillrect", x, y, w, h, col);
		}
	}
	
	
	public static function print(x:Int, y:Int, text:String, col:Int) {
		if (runlocally) {
			Text.display(x, y, text, col);
		}else{
			ExternalInterface.call("print", x, y, text, col);
		}
	}
}