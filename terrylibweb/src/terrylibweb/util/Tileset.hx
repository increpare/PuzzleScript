package terrylibweb.util;

import openfl.display.*;

class Tileset {
	public function new(n:String, w:Int, h:Int) {
		name = n;
		width = w;
		height = h;
		
		animationspeed = 0;
		timethisframe = 0;
		currentframe = 0;
		
		startframe = 0;
		endframe = -1;
	}
	
	public var tiles:Array<BitmapData> = new Array<BitmapData>();
	public var name:String;
	public var width:Int;
	public var height:Int;
	
	public var animationspeed:Int;
	public var timethisframe:Int;
	public var currentframe:Int;
	
	public var startframe:Int;
	public var endframe:Int;
}