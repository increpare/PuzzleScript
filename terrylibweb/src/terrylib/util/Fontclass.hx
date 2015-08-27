package terrylib.util;

import terrylib.bitmapFont.*;
import openfl.Assets;
import openfl.text.*;
import openfl.display.*;
import openfl.geom.*;

@:access(terrylib.Text)
class Fontclass {
	public function new(_name:String, _size:Int) {
		type = Text.fontfile[Text.fontfileindex.get(_name)].type;
		if (type == "bitmap") {
			loadbitmapfont(_name, _size);
		}else if (type == "ttf") {
			loadttffont(_name, _size);
		}
	}
	
	public function loadbitmapfont(_name:String, _size:Int) {
		name = _name;
		size = _size;
		
		tf_bitmap = new BitmapTextField(Text.fontfile[Text.fontfileindex.get(_name)].bitmapfont);
		tf_bitmap.text = "???";
		
		tf_bitmap.background = false;
		
		tf_bitmap.size = _size;
		tfbitmap = new BitmapData(Gfx.screenwidth, Gfx.screenheight, true, 0);
	}
	
	public function loadttffont(_name:String, _size:Int) {
		name = _name;
		size = _size;
		
		tf_ttf = new TextField();
		tf_ttf.embedFonts = true;
		tf_ttf.defaultTextFormat = new TextFormat(Text.getfonttypename(_name), size, 0, false);
		tf_ttf.selectable = false;
		tf_ttf.width = Gfx.screenwidth; 
		tf_ttf.height = Gfx.screenheight;
		// Taking this out for consistancy: only works on flash
		//if (size <= 16) {
		//	tf.antiAliasType = AntiAliasType.ADVANCED; //Small fonts need proper antialiasing
		//}else {
		tf_ttf.antiAliasType = AntiAliasType.NORMAL;	
		//}
		
		
		tf_ttf.text = "???";
		tfbitmap = new BitmapData(Gfx.screenwidth, Gfx.screenheight, true, 0);
		tf_ttf.height = Gfx.screenheight;
	}
	
	public function clearbitmap() {
		tfbitmap.fillRect(tfbitmap.rect, 0);
	}
	
	public var tf_bitmap:BitmapTextField;
	public var tf_ttf:TextField;
	public var tfbitmap:BitmapData;
	
	public var name:String;
	public var type:String;
	public var size:Int;
}