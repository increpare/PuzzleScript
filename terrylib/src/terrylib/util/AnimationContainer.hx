package terrylib.util;

@:access(terrylib.Gfx)
class AnimationContainer {
	public function new(_animationname:String, _tileset:String, _startframe:Int, _endframe:Int, _delayperframe:Int) {
		name = _animationname;
		tileset = _tileset;
		tilesetnum = Gfx.tilesetindex.get(tileset);
		startframe = _startframe;
		endframe = _endframe;
		delayperframe = _delayperframe;
		
		reset();
	}
	
	public function update() {
		timethisframe++;
		if (timethisframe > delayperframe) {
			timethisframe = 0;
	  	currentframe++;
			if (currentframe > endframe) {
				currentframe = startframe;
			}
		}
	}
	
	public function reset() {
		timethisframe = 0;
		currentframe = startframe;
	}
	
	public var name:String;
	
	public var tileset:String;
	public var tilesetnum:Int;
	public var startframe:Int;
	public var endframe:Int;
	
	public var delayperframe:Int;
	public var timethisframe:Int;
	public var currentframe:Int;
}