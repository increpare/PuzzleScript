package terrylib.bitmapFont;

/**
 * Possible border styles for BitmapTextField. Default is NONE, which means no borders.
 */
enum TextBorderStyle
{
	NONE;
	/**
	 * A simple shadow to the lower-right
	 */
	SHADOW;
	/**
	 * Outline on all 8 sides
	 */
	OUTLINE;
	/**
	 * Outline, optimized using only 4 draw calls. (Might not work for narrow and/or 1-pixel fonts)
	 */
	OUTLINE_FAST;
}