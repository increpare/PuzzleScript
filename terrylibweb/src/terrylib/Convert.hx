package terrylib;

class Convert {
	public static function tostring(?value:Dynamic):String {
		return Std.string(value);
	}
	
	public static function toint(?value:Dynamic):Int {
		return Std.parseInt(value);
	}
	
	public static function tofloat(?value:Dynamic):Float {
		return Std.parseFloat(value);
	}
}