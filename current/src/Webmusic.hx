import terrylib.*;
import openfl.external.ExternalInterface;

class Webmusic{
	public static function playsound(t:Int){
		ExternalInterface.call("playSound", t);
	}
}