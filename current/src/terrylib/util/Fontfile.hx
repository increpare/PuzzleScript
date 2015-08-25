package terrylib.util;

import openfl.Assets;
import openfl.text.*;
import openfl.display.*;

class Fontfile {
	public function new(_file:String) {
		filename = "data/fonts/" + _file + ".ttf";
		font = Assets.getFont(filename);
		typename = font.fontName;
	}
	
	public var font:Font;
	public var filename:String;
	public var typename:String;
}