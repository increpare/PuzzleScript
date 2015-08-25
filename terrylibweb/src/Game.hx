import openfl.external.ExternalInterface;

class Game {
	public static function title(t:String) {
		Webscript.title = t;
		ExternalInterface.call("settitle", t);
	}
	
	public static function homepage(p:String) {
		Webscript.homepage = p;
		ExternalInterface.call("sethomepage", p);
	}
	
	public static function background(c:Int) {
		Webscript.background_color = c;
		ExternalInterface.call("setbackgroundcolor", c);
	}
}