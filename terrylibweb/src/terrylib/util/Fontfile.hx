package terrylib.util;

import terrylib.bitmapFont.*;
import openfl.Assets;
import openfl.text.*;
import openfl.display.*;

class Fontfile {
	public function new(_file:String) {
		if (Assets.exists("data/fonts/" + _file + "/" + _file + ".fnt")) {
			type = "bitmap";
			fontxml = Xml.parse(Assets.getText("data/fonts/" + _file + "/" + _file + ".fnt"));
			var tempfontimage:BitmapData = Assets.getBitmapData("data/fonts/" + _file + "/" + _file + "_0.png");
			fontimage = new BitmapData(tempfontimage.width, tempfontimage.height, true, 0);
			for (j in 0 ... tempfontimage.height) {
				for (i in 0 ... tempfontimage.width) {
					var cpixel:Int = tempfontimage.getPixel(i, j);
					if (cpixel != 0x00000000 && cpixel != 0x000000) {
						fontimage.setPixel32(i, j, 0xFFFFFFFF);
					}
				}
			}
			
			bitmapfont = BitmapFont.fromAngelCode(fontimage, fontxml);
			typename = _file;
		}else if(Assets.exists("data/fonts/" + _file + "/" + _file + ".ttf")){		
			type = "ttf";
			filename = "data/fonts/" + _file + "/" + _file + ".ttf";
			font = Assets.getFont(filename);
			typename = font.fontName;
		}else {
			throw("ERROR: Cannot set font to \"" + _file + "\", no TTF or Bitmap Font found.");
		}
	}
	
	public var typename:String;
	
	public var bitmapfont:BitmapFont;
	public var fontxml:Xml;
	public var fontimage:BitmapData;
	
	public var font:Font;
	public var filename:String;
	public var type:String;
}