import openfl.external.ExternalInterface;

class Game {
	public static function title(t:String) {
		Webscript.title = t;
		#if !flash
		ExternalInterface.call("settitle", t);
		#end
	}
	
	public static function homepage(p:String) {
		Webscript.homepage = p;
		#if !flash
		ExternalInterface.call("sethomepage", p);
		#end
	}
	
	public static function background(c:Int) {
		Webscript.background_color = c;
		#if !flash
		ExternalInterface.call("setbackgroundcolor", c);
		#end
	}
}