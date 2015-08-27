package terrylib.bitmapFont;

/**
 * Possible BitmapTextField align modes.
 */
@:enum
abstract BitmapTextAlign(String) from String
{
	var LEFT = "left";
	var CENTER = "center";
	var RIGHT = "right";
}