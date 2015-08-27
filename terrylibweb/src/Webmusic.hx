import terrylib.*;
import openfl.external.ExternalInterface;

class Webmusic{
	public static function playsound(t:Int) {
	  #if !flash	
		ExternalInterface.call("playSound", t);
		#end
	}
	
	public static function playnote(seed:Int, freq:Float, length:Float, volume:Float) {
	  #if !flash	
		ExternalInterface.call("playNote", seed, freq, length, volume);
		#end
	}
}