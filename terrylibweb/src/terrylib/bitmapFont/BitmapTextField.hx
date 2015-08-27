package terrylib.bitmapFont;

import flash.display.Bitmap;
import flash.display.BitmapData;
import flash.display.Graphics;
import flash.display.Sprite;
import flash.events.Event;
import flash.geom.Matrix;
import flash.geom.Point;
import terrylib.bitmapFont.BitmapFont;
import haxe.Utf8;
import openfl.display.PixelSnapping;
import openfl.display.Tilesheet;

/**
 * Class for rendering text with provided bitmap font and some additional options.
 */
class BitmapTextField extends Sprite 
{
	/**
	 * Font for text rendering.
	 */
	public var font(default, set):BitmapFont;
	
	/**
	 * Text to display.
	 */
	public var text(default, set):String = "";
	
	/**
	 * Helper array which contains actual strings for rendering.
	 */
	private var _lines:Array<String> = [];
	/**
	 * Helper array which contains width of each displayed lines.
	 */
	private var _linesWidth:Array<Float> = [];
	
	/**
	 * Specifies how the text field should align text.
	 */
	public var alignment(default, set):BitmapTextAlign = BitmapTextAlign.LEFT;
	
	/**
	 * The distance to add between lines.
	 */
	public var lineSpacing(default, set):Int = 0;
	
	/**
	 * The distance to add between letters.
	 */
	public var letterSpacing(default, set):Int = 0;
	
	/**
	 * Whether to convert text to upper case or not.
	 */
	public var autoUpperCase(default, set):Bool = false;
	
	/**
	 * A Boolean value that indicates whether the text field has word wrap.
	 */
	public var wordWrap(default, set):Bool = true;
	
	/**
	 * Whether word wrapping algorithm should wrap lines by words or by single character.
	 * Default value is true.
	 */ 
	public var wrapByWord(default, set):Bool = true;
	
	/**
	 * Whether this text field have fixed width or not.
	 * Default value if true.
	 */
	public var autoSize(default, set):Bool = true;
	
	/**
	 * Number of pixels between text and text field border
	 */
	public var padding(default, set):Int = 0;
	
	/**
	 * Width of the text in this text field.
	 */
	public var textWidth(get, null):Float;
	
	/**
	 * Height of the text in this text field.
	 */
	public var textHeight(get, null):Float;
	
	/**
	 * Height of the single line of text (without lineSpacing).
	 */
	public var lineHeight(get, null):Float;
	
	/**
	 * Number of space characters in one tab.
	 */
	public var numSpacesInTab(default, set):Int = 4;
	private var _tabSpaces:String = "    ";
	
	/**
	 * The color of the text in 0xAARRGGBB format.
	 */
	public var textColor(default, set):UInt = 0xFFFFFFFF;
	
	/**
	 * Whether to use textColor while rendering or not.
	 */
	public var useTextColor(default, set):Bool = false;
	
	/**
	 * Use a border style
	 */	
	public var borderStyle(default, set):TextBorderStyle = NONE;
	
	/**
	 * The color of the border in 0xAARRGGBB format
	 */
	public var borderColor(default, set):UInt = 0xFF000000;
	
	/**
	 * The size of the border, in pixels.
	 */
	public var borderSize(default, set):Float = 1;
	
	/**
	 * How many iterations do use when drawing the border. 0: only 1 iteration, 1: one iteration for every pixel in borderSize
	 * A value of 1 will have the best quality for large border sizes, but might reduce performance when changing text. 
	 * NOTE: If the borderSize is 1, borderQuality of 0 or 1 will have the exact same effect (and performance).
	 */
	public var borderQuality(default, set):Float = 0;
	
	/**
	 * Offset that is applied to the shadow border style, if active. 
	 * x and y are multiplied by borderSize. Default is (1, 1), or lower-right corner.
	 */
	public var shadowOffset(default, null):Point;
	
	/**
	 * Specifies whether the text should have background
	 */
	public var background(default, set):Bool = false;
	
	/**
	 * Specifies the color of background in 0xAARRGGBB format.
	 */
	public var backgroundColor(default, set):UInt = 0x00000000;
	
	/**
	 * Specifies whether the text field will break into multiple lines or not on overflow.
	 */
	public var multiLine(default, set):Bool = true;
	
	/**
	 * Reflects how many lines have this text field.
	 */
	public var numLines(get, null):Int = 0;
	
	/**
	 * The "size" (scale) of the font.
	 */
	public var size(default, set):Float = 1;
	
	public var smoothing(default, set):Bool;
	
	/**
	 * Whether graphics/bitmapdata of this text field should be updated immediately after each setter call.
	 * Default value is true which means that graphics will be updated/regenerated after each setter call,
	 * which could be CPU-heavy.
	 * So if you want to save some CPU resources then you could set updateImmediately to false,
	 * make all operations with this text field (change text color, size, border style, etc.).
	 * and then set updateImmediately back to true which will immediately update graphics of this text field. 
	 */
	public var updateImmediately(default, set):Bool = true;
	
	private var _pendingTextChange:Bool = true;
	private var _pendingGraphicChange:Bool = true;
	
	private var _pendingTextGlyphsChange:Bool = true;
	private var _pendingBorderGlyphsChange:Bool = false;
	
	private var _fieldWidth:Int = 1;
	private var _fieldHeight:Int = 1;
	
	#if RENDER_BLIT
	private var _bitmap:Bitmap;
	private var _bitmapData:BitmapData;
	
	/**
	 * Glyphs for text rendering. Used only in blit render mode.
	 */
	private var textGlyphs:BitmapGlyphCollection;
	/**
	 * Glyphs for border (shadow or outline) rendering.
	 * Used only in blit render mode.
	 */
	private var borderGlyphs:BitmapGlyphCollection;
	
	private var _point:Point;
	#else
	private var _drawData:Array<Float>;
	#end
	
	/**
	 * Constructs a new text field component.
	 * @param font	optional parameter for component's font prop
	 * @param text	optional parameter for component's text
	 */
	public function new(?font:BitmapFont, text:String = "", pixelSnapping:PixelSnapping = null, smoothing:Bool = false)
	{
		super();
		
		shadowOffset = new Point(1, 1);
		
		#if RENDER_BLIT
		pixelSnapping = (pixelSnapping == null) ? PixelSnapping.AUTO : pixelSnapping;
		_bitmapData = new BitmapData(_fieldWidth, _fieldHeight, true, 0);
		_bitmap = new Bitmap(_bitmapData, pixelSnapping, smoothing);
		_bitmap.smoothing = false;
		addChild(_bitmap);
		_point = new Point();
		#else
		_drawData = [];
		#end
		
		if (font == null)
		{
			font = BitmapFont.getDefaultFont();
		}
		
		this.font = font;
		this.text = text;
		this.smoothing = smoothing;
	}
	
	/**
	 * Clears all resources used by this text field.
	 */
	public function dispose():Void 
	{
		updateImmediately = false;
		
		font = null;
		text = null;
		_lines = null;
		_linesWidth = null;
		shadowOffset = null;
		
		#if RENDER_BLIT
		_point = null;
		
		if (textGlyphs != null)
		{
			textGlyphs.dispose();
		}
		textGlyphs = null;
		
		if (borderGlyphs != null)
		{
			borderGlyphs.dispose();
		}
		borderGlyphs = null;
		
		if (_bitmap != null)
		{
			removeChild(_bitmap);
		}
		_bitmap = null;
		
		if (_bitmapData != null)
		{
			_bitmapData.dispose();
		}
		_bitmapData = null;
		#else
		_drawData = null;
		#end
	}
	
	/**
	 * Forces graphic regeneration for this text field immediately.
	 */
	public function forceGraphicUpdate():Void
	{
		_pendingGraphicChange = true;
		checkPendingChanges();
	}
	
	inline private function checkImmediateChanges():Void
	{
		if (updateImmediately)
		{
			checkPendingChanges();
		}
	}
	
	inline private function checkPendingChanges():Void
	{
		if (_pendingTextGlyphsChange)
		{
			updateTextGlyphs();
		}
		
		if (_pendingBorderGlyphsChange)
		{
			updateBorderGlyphs();
		}
		
		if (_pendingTextChange)
		{
			updateText();
			_pendingGraphicChange = true;
		}
		
		if (_pendingGraphicChange)
		{
			updateGraphic();
		}
	}
	
	private function set_textColor(value:UInt):UInt 
	{
		if (textColor != value)
		{
			textColor = value;
			_pendingTextGlyphsChange = true;
			checkImmediateChanges();
		}
		
		return value;
	}
	
	private function set_useTextColor(value:Bool):Bool 
	{
		if (useTextColor != value)
		{
			useTextColor = value;
			_pendingTextGlyphsChange = true;
			checkImmediateChanges();
		}
		
		return value;
	}
	
	private function set_text(value:String):String 
	{
		if (value != text && value != null)
		{
			text = value;
			_pendingTextChange = true;
			checkImmediateChanges();
		}
		
		return value;
	}
	
	private function updateText():Void 
	{
		var tmp:String = (autoUpperCase) ? text.toUpperCase() : text;
		
		_lines = tmp.split("\n");
		
		if (!autoSize)
		{
			if (wordWrap)
			{
				wrap();
			}
			else
			{
				cutLines();
			}
		}
		
		if (!multiLine)
		{
			_lines = [_lines[0]];
		}
		
		_pendingTextChange = false;
		_pendingGraphicChange = true;
	}
	
	/**
	 * Calculates the size of text field.
	 */
	private function computeTextSize():Void 
	{
		var txtWidth:Int = Math.ceil(_fieldWidth);
		var txtHeight:Int = Math.ceil(textHeight) + 2 * padding;
		
		var tw:Int = Math.ceil(textWidth);
		
		if (autoSize)
		{
			txtWidth = tw + 2 * padding;
		}
		else
		{
			txtWidth = Math.ceil(_fieldWidth);
		}
		
		_fieldWidth = (txtWidth == 0) ? 1 : txtWidth;
		_fieldHeight = (txtHeight == 0) ? 1 : txtHeight;
	}
	
	/**
	 * Calculates width of the line with provided index
	 * 
	 * @param	lineIndex	index of the line in _lines array
	 * @return	The width of the line
	 */
	public function getLineWidth(lineIndex:Int):Float
	{
		if (lineIndex < 0 || lineIndex >= _lines.length)
		{
			return 0;
		}
		
		return getStringWidth(_lines[lineIndex]);
	}
	
	/**
	 * Calculates width of provided string (for current font with fontScale).
	 * 
	 * @param	str	String to calculate width for
	 * @return	The width of result bitmap text.
	 */
	public function getStringWidth(str:String, fordrawing:Bool=true):Float
	{
		var spaceWidth:Float = Math.ceil(font.spaceWidth * size);
		var tabWidth:Float = Math.ceil(spaceWidth * numSpacesInTab);
		
		var lineLength:Int = Utf8.length(str);	// lenght of the current line
		var lineWidth:Float = Math.ceil(Math.abs(font.minOffsetX) * size);
		if(!fordrawing) lineWidth = 0;
		
		var charCode:Int;
		var charWidth:Float = 0;			// the width of current character
		
		var widthPlusOffset:Int = 0;
		var glyphFrame:BitmapGlyphFrame;
		
		for (c in 0...lineLength)
		{
			charCode = Utf8.charCodeAt(str, c);
			
			if (charCode == BitmapFont.spaceCode)
			{
				charWidth = spaceWidth;
			}
			else if (charCode == BitmapFont.tabCode)
			{
				charWidth = tabWidth;
			}
			else
			{
				if (font.glyphs.exists(charCode))
				{
					glyphFrame = font.glyphs.get(charCode);
					charWidth = Math.ceil(glyphFrame.xadvance * size);
					
					if (c == (lineLength - 1))
					{
						widthPlusOffset = Math.ceil((glyphFrame.xoffset + glyphFrame.bitmap.width) * size); 
						if (widthPlusOffset > charWidth)
						{
							charWidth = widthPlusOffset;
						}
					}
				}
				else
				{
					charWidth = 0;
				}
			}
			
			lineWidth += (charWidth + letterSpacing);
		}
		
		if (lineLength > 0)
		{
			lineWidth -= letterSpacing;
		}
		
		return lineWidth;
	}
	
	/**
	 * Just cuts the lines which are too long to fit in the field.
	 */
	private function cutLines():Void 
	{
		var newLines:Array<String> = [];
		
		var lineLength:Int;			// lenght of the current line
		
		var c:Int;					// char index
		var char:String; 			// current character in word
		var charCode:Int;
		var charWidth:Float = 0;	// the width of current character
		
		var subLine:Utf8;			// current subline to assemble
		var subLineWidth:Float;		// the width of current subline
		
		var spaceWidth:Float = font.spaceWidth * size;
		var tabWidth:Float = spaceWidth * numSpacesInTab;
		
		var startX:Float = Math.abs(font.minOffsetX) * size;
		
		for (line in _lines)
		{
			lineLength = Utf8.length(line);
			subLine = new Utf8();
			subLineWidth = startX;
			
			c = 0;
			while (c < lineLength)
			{
				charCode = Utf8.charCodeAt(line, c);
				
				if (charCode == BitmapFont.spaceCode)
				{
					charWidth = spaceWidth;
				}
				else if (charCode == BitmapFont.tabCode)
				{
					charWidth = tabWidth;
				}
				else
				{
					charWidth = (font.glyphs.exists(charCode)) ? font.glyphs.get(charCode).xadvance * size : 0;
				}
				charWidth += letterSpacing;
				
				if (subLineWidth + charWidth > _fieldWidth - 2 * padding)
				{
					subLine.addChar(charCode);
					newLines.push(subLine.toString());
					subLine = new Utf8();
					subLineWidth = startX;
					c = lineLength;
				}
				else
				{
					subLine.addChar(charCode);
					subLineWidth += charWidth;
				}
				
				c++;
			}
		}
		
		_lines = newLines;
	}
	
	/**
	 * Automatically wraps text by figuring out how many characters can fit on a
	 * single line, and splitting the remainder onto a new line.
	 */
	private function wrap():Void
	{
		// subdivide lines
		var newLines:Array<String> = [];
		var words:Array<String>;			// the array of words in the current line
		
		for (line in _lines)
		{
			words = [];
			// split this line into words
			splitLineIntoWords(line, words);
			
			if (wrapByWord)
			{
				wrapLineByWord(words, newLines);
			}
			else
			{
				wrapLineByCharacter(words, newLines);
			}
		}
		
		_lines = newLines;
	}
	
	/**
	 * Helper function for splitting line of text into separate words.
	 * 
	 * @param	line	line to split.
	 * @param	words	result array to fill with words.
	 */
	private function splitLineIntoWords(line:String, words:Array<String>):Void
	{
		var word:String = "";				// current word to process
		var wordUtf8:Utf8 = new Utf8();
		var isSpaceWord:Bool = false; 		// whether current word consists of spaces or not
		var lineLength:Int = Utf8.length(line);	// lenght of the current line
		
		var hyphenCode:Int = Utf8.charCodeAt('-', 0);
		
		var c:Int = 0;						// char index on the line
		var charCode:Int; 					// code for the current character in word
		var charUtf8:Utf8;
		
		while (c < lineLength)
		{
			charCode = Utf8.charCodeAt(line, c);
			word = wordUtf8.toString();
			
			if (charCode == BitmapFont.spaceCode || charCode == BitmapFont.tabCode)
			{
				if (!isSpaceWord)
				{
					isSpaceWord = true;
					
					if (word != "")
					{
						words.push(word);
						wordUtf8 = new Utf8();
					}
				}
				
				wordUtf8.addChar(charCode);
			}
			else if (charCode == hyphenCode)
			{
				if (isSpaceWord && word != "")
				{
					isSpaceWord = false;
					words.push(word);
					words.push('-');
				}
				else if (isSpaceWord == false)
				{
					charUtf8 = new Utf8();
					charUtf8.addChar(charCode);
					words.push(word + charUtf8.toString());
				}
				
				wordUtf8 = new Utf8();
			}
			else
			{
				if (isSpaceWord && word != "")
				{
					isSpaceWord = false;
					words.push(word);
					wordUtf8 = new Utf8();
				}
				
				wordUtf8.addChar(charCode);
			}
			
			c++;
		}
		
		word = wordUtf8.toString();
		if (word != "") words.push(word);
	}
	
	/**
	 * Wraps provided line by words.
	 * 
	 * @param	words		The array of words in the line to process.
	 * @param	newLines	Array to fill with result lines.
	 */
	private function wrapLineByWord(words:Array<String>, newLines:Array<String>):Void
	{
		var numWords:Int = words.length;	// number of words in the current line
		var w:Int;							// word index in the current line
		var word:String;					// current word to process
		var wordWidth:Float;				// total width of current word
		var wordLength:Int;					// number of letters in current word
		
		var isSpaceWord:Bool = false; 		// whether current word consists of spaces or not
		
		var charCode:Int;
		var charWidth:Float = 0;			// the width of current character
		
		var subLines:Array<String> = [];	// helper array for subdividing lines
		
		var subLine:String;					// current subline to assemble
		var subLineWidth:Float;				// the width of current subline
		
		var spaceWidth:Float = font.spaceWidth * size;
		var tabWidth:Float = spaceWidth * numSpacesInTab;
		
		var startX:Float = Math.abs(font.minOffsetX) * size;
		
		if (numWords > 0)
		{
			w = 0;
			subLineWidth = startX;
			subLine = "";
			
			while (w < numWords)
			{
				wordWidth = 0;
				word = words[w];
				wordLength = Utf8.length(word);
				
				charCode = Utf8.charCodeAt(word, 0);
				isSpaceWord = (charCode == BitmapFont.spaceCode || charCode == BitmapFont.tabCode);
				
				for (c in 0...wordLength)
				{
					charCode = Utf8.charCodeAt(word, c);
					
					if (charCode == BitmapFont.spaceCode)
					{
						charWidth = spaceWidth;
					}
					else if (charCode == BitmapFont.tabCode)
					{
						charWidth = tabWidth;
					}
					else
					{
						charWidth = (font.glyphs.exists(charCode)) ? font.glyphs.get(charCode).xadvance * size : 0;
					}
					
					wordWidth += charWidth;
				}
				
				wordWidth += ((wordLength - 1) * letterSpacing);
				
				if (subLineWidth + wordWidth > _fieldWidth - 2 * padding)
				{
					if (isSpaceWord)
					{
						subLines.push(subLine);
						subLine = "";
						subLineWidth = startX;
					}
					else if (subLine != "") // new line isn't empty so we should add it to sublines array and start another one
					{
						subLines.push(subLine);
						subLine = word;
						subLineWidth = startX + wordWidth + letterSpacing;
					}
					else					// the line is too tight to hold even one word
					{
						subLine = word;
						subLineWidth = startX + wordWidth + letterSpacing;
					}
				}
				else
				{
					subLine += word;
					subLineWidth += wordWidth + letterSpacing;
				}
				
				w++;
			}
			
			if (subLine != "")
			{
				subLines.push(subLine);
			}
		}
		
		for (subline in subLines)
		{
			newLines.push(subline);
		}
	}
	
	/**
	 * Wraps provided line by characters (as in standart flash text fields).
	 * 
	 * @param	words		The array of words in the line to process.
	 * @param	newLines	Array to fill with result lines.
	 */
	private function wrapLineByCharacter(words:Array<String>, newLines:Array<String>):Void
	{
		var numWords:Int = words.length;	// number of words in the current line
		var w:Int;							// word index in the current line
		var word:String;					// current word to process
		var wordLength:Int;					// number of letters in current word
		
		var isSpaceWord:Bool = false; 		// whether current word consists of spaces or not
		
		var char:String; 					// current character in word
		var charCode:Int;
		var c:Int;							// char index
		var charWidth:Float = 0;			// the width of current character
		
		var subLines:Array<String> = [];	// helper array for subdividing lines
		
		var subLine:String;					// current subline to assemble
		var subLineUtf8:Utf8;
		var subLineWidth:Float;				// the width of current subline
		
		var spaceWidth:Float = font.spaceWidth * size;
		var tabWidth:Float = spaceWidth * numSpacesInTab;
		
		var startX:Float = Math.abs(font.minOffsetX) * size;
		
		if (numWords > 0)
		{
			w = 0;
			subLineWidth = startX;
			subLineUtf8 = new Utf8();
			
			while (w < numWords)
			{
				word = words[w];
				wordLength = Utf8.length(word);
				
				charCode = Utf8.charCodeAt(word, 0);
				isSpaceWord = (charCode == BitmapFont.spaceCode || charCode == BitmapFont.tabCode);
				
				c = 0;
				
				while (c < wordLength)
				{
					charCode = Utf8.charCodeAt(word, c);
					
					if (charCode == BitmapFont.spaceCode)
					{
						charWidth = spaceWidth;
					}
					else if (charCode == BitmapFont.tabCode)
					{
						charWidth = tabWidth;
					}
					else
					{
						charWidth = (font.glyphs.exists(charCode)) ? font.glyphs.get(charCode).xadvance * size : 0;
					}
					
					if (subLineWidth + charWidth > _fieldWidth - 2 * padding)
					{
						subLine = subLineUtf8.toString();
						
						if (isSpaceWord) // new line ends with space / tab char, so we push it to sublines array, skip all the rest spaces and start another line
						{
							subLines.push(subLine);
							c = wordLength;
							subLineUtf8 = new Utf8();
							subLineWidth = startX;
						}
						else if (subLine != "") // new line isn't empty so we should add it to sublines array and start another one
						{
							subLines.push(subLine);
							subLineUtf8 = new Utf8();
							subLineUtf8.addChar(charCode);
							subLineWidth = startX + charWidth + letterSpacing;
						}
						else	// the line is too tight to hold even one glyph
						{
							subLineUtf8 = new Utf8();
							subLineUtf8.addChar(charCode);
							subLineWidth = startX + charWidth + letterSpacing;
						}
					}
					else
					{
						subLineUtf8.addChar(charCode);
						subLineWidth += (charWidth + letterSpacing);
					}
					
					c++;
				}
				
				w++;
			}
			
			subLine = subLineUtf8.toString();
			
			if (subLine != "")
			{
				subLines.push(subLine);
			}
		}
		
		for (subline in subLines)
		{
			newLines.push(subline);
		}
	}
	
	/**
	 * Internal method for updating the view of the text component
	 */
	private function updateGraphic():Void 
	{
		computeTextSize();
		var colorForFill:Int = (background) ? backgroundColor : 0x00000000;
		#if RENDER_BLIT
		if (_bitmapData == null || (_fieldWidth != _bitmapData.width || _fieldHeight != _bitmapData.height))
		{
			if (_bitmapData != null)
			{
				_bitmapData.dispose();
			}
			
			_bitmapData = new BitmapData(_fieldWidth, _fieldHeight, true, colorForFill);
			_bitmap.bitmapData = _bitmapData;
			_bitmap.smoothing = smoothing;
		}
		else 
		{
			_bitmapData.fillRect(_bitmapData.rect, colorForFill);
		}
		#else
		this.graphics.clear();
		
		if (colorForFill != 0x00000000)
		{
			this.graphics.beginFill(colorForFill & 0x00FFFFFF, ((colorForFill >> 24) & 0xFF) / 255);
			this.graphics.drawRect(0, 0, _fieldWidth, _fieldHeight);
			this.graphics.endFill();
		}
		
		var colorForBorder:UInt = (borderStyle != TextBorderStyle.NONE) ? borderColor : 0xFFFFFFFF;
		var colorForText:UInt = (useTextColor) ? textColor : 0xFFFFFFFF;
		
		_drawData.splice(0, _drawData.length);
		#end
		
		if (size > 0)
		{
			#if RENDER_BLIT
			_bitmapData.lock();
			#end
			
			var numLines:Int = _lines.length;
			var line:String;
			var lineWidth:Float;
			
			var ox:Int, oy:Int;
			
			var iterations:Int = Std.int(borderSize * borderQuality);
			iterations = (iterations <= 0) ? 1 : iterations; 
			
			var delta:Int = Std.int(borderSize / iterations);
			
			var iterationsX:Int = 1;
			var iterationsY:Int = 1;
			var deltaX:Int = 1;
			var deltaY:Int = 1;
			
			if (borderStyle == TextBorderStyle.SHADOW)
			{
				iterationsX = Math.round(Math.abs(shadowOffset.x) * borderQuality);
				iterationsX = (iterationsX <= 0) ? 1 : iterationsX;
				
				iterationsY = Math.round(Math.abs(shadowOffset.y) * borderQuality);
				iterationsY = (iterationsY <= 0) ? 1 : iterationsY;
				
				deltaX = Math.round(shadowOffset.x / iterationsX);
				deltaY = Math.round(shadowOffset.y / iterationsY);
			}
			
			// render border
			for (i in 0...numLines)
			{
				line = _lines[i];
				lineWidth = _linesWidth[i];
				
				// LEFT
				ox = Std.int(Math.abs(font.minOffsetX) * size);
				oy = Std.int(i * (font.lineHeight * size + lineSpacing)) + padding;
				
				if (alignment == BitmapTextAlign.CENTER) 
				{
					ox += Std.int((_fieldWidth - lineWidth) / 2) - padding;
				}
				if (alignment == BitmapTextAlign.RIGHT) 
				{
					ox += (_fieldWidth - Std.int(lineWidth)) - padding;
				}
				else	// LEFT
				{
					ox += padding;
				}
				
				switch (borderStyle)
				{
					case SHADOW:
						for (iterY in 0...iterationsY)
						{
							for (iterX in 0...iterationsX)
							{
								#if RENDER_BLIT
								blitLine(line, borderGlyphs, ox + deltaX * (iterX + 1), oy + deltaY * (iterY + 1));
								#else
								renderLine(line, colorForBorder, ox + deltaX * (iterX + 1), oy + deltaY * (iterY + 1));
								#end
							}
						}
					case OUTLINE:
						//Render an outline around the text
						//(do 8 offset draw calls)
						var itd:Int = 0;
						for (iter in 0...iterations)
						{
							itd = delta * (iter + 1);
							#if RENDER_BLIT
							//upper-left
							blitLine(line, borderGlyphs, ox - itd, oy - itd);
							//upper-middle
							blitLine(line, borderGlyphs, ox, oy - itd);
							//upper-right
							blitLine(line, borderGlyphs, ox + itd, oy - itd);
							//middle-left
							blitLine(line, borderGlyphs, ox - itd, oy);
							//middle-right
							blitLine(line, borderGlyphs, ox + itd, oy);
							//lower-left
							blitLine(line, borderGlyphs, ox - itd, oy + itd);
							//lower-middle
							blitLine(line, borderGlyphs, ox, oy + itd);
							//lower-right
							blitLine(line, borderGlyphs, ox + itd, oy + itd);
							#else
							//upper-left
							renderLine(line, colorForBorder, ox - itd, oy - itd);
							//upper-middle
							renderLine(line, colorForBorder, ox, oy - itd);
							//upper-right
							renderLine(line, colorForBorder, ox + itd, oy - itd);
							//middle-left
							renderLine(line, colorForBorder, ox - itd, oy);
							//middle-right
							renderLine(line, colorForBorder, ox + itd, oy);
							//lower-left
							renderLine(line, colorForBorder, ox - itd, oy + itd);
							//lower-middle
							renderLine(line, colorForBorder, ox, oy + itd);
							//lower-right
							renderLine(line, colorForBorder, ox + itd, oy + itd);
							#end
						}
					case OUTLINE_FAST:
						//Render an outline around the text
						//(do 4 diagonal offset draw calls)
						//(this method might not work with certain narrow fonts)
						var itd:Int = 0;
						for (iter in 0...iterations)
						{
							itd = delta * (iter + 1);
							#if RENDER_BLIT
							//upper-left
							blitLine(line, borderGlyphs, ox - itd, oy - itd);
							//upper-right
							blitLine(line, borderGlyphs, ox + itd, oy - itd);
							//lower-left
							blitLine(line, borderGlyphs, ox - itd, oy + itd);
							//lower-right
							blitLine(line, borderGlyphs, ox + itd, oy + itd);
							#else
							//upper-left
							renderLine(line, colorForBorder, ox - itd, oy - itd);
							//upper-right
							renderLine(line, colorForBorder, ox + itd, oy - itd);
							//lower-left
							renderLine(line, colorForBorder, ox - itd, oy + itd);
							//lower-right
							renderLine(line, colorForBorder, ox + itd, oy + itd);
							#end
						}	
					case NONE:
				}
			}
			
			// render text
			for (i in 0...numLines)
			{
				line = _lines[i];
				lineWidth = _linesWidth[i];
				
				// LEFT
				ox = Std.int(Math.abs(font.minOffsetX) * size);
				oy = Std.int(i * (font.lineHeight * size + lineSpacing)) + padding;
				
				if (alignment == BitmapTextAlign.CENTER) 
				{
					ox += Std.int((_fieldWidth - lineWidth) / 2) - padding;
				}
				if (alignment == BitmapTextAlign.RIGHT) 
				{
					ox += (_fieldWidth - Std.int(lineWidth)) - padding;
				}
				else	// LEFT
				{
					ox += padding;
				}
				
				#if RENDER_BLIT
				blitLine(line, textGlyphs, ox, oy);
				#else
				renderLine(line, colorForText, ox, oy);
				#end
			}
			
			#if RENDER_BLIT
			_bitmapData.unlock();
			#else
			font.tilesheet.drawTiles(this.graphics, _drawData, smoothing, Tilesheet.TILE_SCALE | Tilesheet.TILE_RGB | Tilesheet.TILE_ALPHA);
			#end
		}
		
		_pendingGraphicChange = false;
	}
	
	#if RENDER_BLIT
	private function blitLine(line:String, glyphs:BitmapGlyphCollection, startX:Int, startY:Int):Void
	{
		if (glyphs == null) return;
		
		var glyph:BitmapGlyph;
		var charCode:Int;
		var curX:Int = startX;
		var curY:Int = startY;
		
		var spaceWidth:Int = Std.int(font.spaceWidth * size);
		var tabWidth:Int = Std.int(spaceWidth * numSpacesInTab);
		
		var lineLength:Int = Utf8.length(line);
		
		for (i in 0...lineLength)
		{
			charCode = Utf8.charCodeAt(line, i);
			
			if (charCode == BitmapFont.spaceCode)
			{
				curX += spaceWidth;
			}
			else if (charCode == BitmapFont.tabCode)
			{
				curX += tabWidth;
			}
			else
			{
				glyph = glyphs.glyphMap.get(charCode);
				if (glyph != null)
				{
					_point.x = curX + glyph.offsetX;
					_point.y = curY + glyph.offsetY;
					_bitmapData.copyPixels(glyph.bitmap, glyph.rect, _point, null, null, true);
					curX += glyph.xAdvance;
				}				
			}
			
			curX += letterSpacing;
		}
	}
	#else
	private function renderLine(line:String, color:UInt, startX:Int, startY:Int):Void
	{
		var glyph:BitmapGlyphFrame;
		var charCode:Int;
		var curX:Float = startX;
		var curY:Int = startY;
		
		var spaceWidth:Int = Std.int(font.spaceWidth * size);
		var tabWidth:Int = Std.int(spaceWidth * numSpacesInTab);
		
		var r:Float = ((color >> 16) & 0xFF) / 255;
		var g:Float = ((color >> 8) & 0xFF) / 255;
		var b:Float = (color & 0xFF) / 255;
		var a:Float = ((color >> 24) & 0xFF) / 255;
		
		var pos:Int = _drawData.length;
		
		var lineLength:Int = Utf8.length(line);
		
		for (i in 0...lineLength)
		{
			charCode = Utf8.charCodeAt(line, i);
			
			if (charCode == BitmapFont.spaceCode)
			{
				curX += spaceWidth;
			}
			else if (charCode == BitmapFont.tabCode)
			{
				curX += tabWidth;
			}
			else
			{
				glyph = font.glyphs.get(charCode);
				if (glyph != null)
				{
					_drawData[pos++] = curX + glyph.xoffset * size;
					_drawData[pos++] = curY + glyph.yoffset * size;
				
					_drawData[pos++] = glyph.tileID;
					
					_drawData[pos++] = size;
					
					_drawData[pos++] = r;
					_drawData[pos++] = g;
					_drawData[pos++] = b;
					_drawData[pos++] = a;
					
					curX += glyph.xadvance * size;
				}				
			}
			
			curX += letterSpacing;
		}
	}
	#end
	
	/**
	 * Set border's style (shadow, outline, etc), color, and size all in one go!
	 * 
	 * @param	Style outline style
	 * @param	Color outline color in flash 0xAARRGGBB format
	 * @param	Size outline size in pixels
	 * @param	Quality outline quality - # of iterations to use when drawing. 0:just 1, 1:equal number to BorderSize
	 */
	public inline function setBorderStyle(Style:TextBorderStyle, Color:UInt = 0xFFFFFFFF, Size:Float = 1, Quality:Float = 1):Void 
	{
		borderStyle = Style;
		borderColor = Color;
		borderSize = Size;
		borderQuality = Quality;
		if (borderStyle == TextBorderStyle.SHADOW)
		{
			shadowOffset.setTo(borderSize, borderSize);
		}
		_pendingGraphicChange = true;
		checkImmediateChanges();
	}
	
	/**
	 * Sets the width of the text field. If the text does not fit, it will spread on multiple lines.
	 */
	#if !flash
	override function set_width(value:Float):Float
	#else
	@:setter(width) function set_width(value:Float):Void
	#end
	{
		value = Std.int(value);
		value = Math.max(1, value);
		
		if (value != width)
		{
			_fieldWidth = (value == 0) ? 1 : Std.int(value);
			_pendingTextChange = true;
			checkImmediateChanges();
		}
		#if !flash
		return value;
		#end
	}
	
	private function set_alignment(value:BitmapTextAlign):BitmapTextAlign 
	{
		if (alignment != value)
		{
			alignment = value;
			_pendingGraphicChange = true;
			checkImmediateChanges();
		}
		
		return value;
	}
	
	private function set_multiLine(value:Bool):Bool 
	{
		if (multiLine != value)
		{
			multiLine = value;
			_pendingTextChange = true;
			checkImmediateChanges();
		}
		
		return value;
	}
	
	private function set_font(value:BitmapFont):BitmapFont 
	{
		if (font != value && value != null)
		{
			font = value;
			_pendingTextChange = true;
			_pendingBorderGlyphsChange = true;
			checkImmediateChanges();
		}
		
		return value;
	}
	
	private function set_lineSpacing(value:Int):Int
	{
		if (lineSpacing != value)
		{
			lineSpacing = Std.int(Math.abs(value));
			_pendingGraphicChange = true;
			checkImmediateChanges();
		}
		
		return lineSpacing;
	}
	
	private function set_letterSpacing(value:Int):Int
	{
		var tmp:Int = Std.int(Math.abs(value));
		
		if (tmp != letterSpacing)
		{
			letterSpacing = tmp;
			_pendingTextChange = true;
			checkImmediateChanges();
		}
		
		return letterSpacing;
	}
	
	private function set_autoUpperCase(value:Bool):Bool 
	{
		if (autoUpperCase != value)
		{
			autoUpperCase = value;
			_pendingTextChange = true;
			checkImmediateChanges();
		}
		
		return autoUpperCase;
	}
	
	private function set_wordWrap(value:Bool):Bool 
	{
		if (wordWrap != value)
		{
			wordWrap = value;
			_pendingTextChange = true;
			checkImmediateChanges();
		}
		
		return wordWrap;
	}
	
	private function set_wrapByWord(value:Bool):Bool
	{
		if (wrapByWord != value)
		{
			wrapByWord = value;
			_pendingTextChange = true;
			checkImmediateChanges();
		}
		
		return value;
	}
	
	private function set_autoSize(value:Bool):Bool 
	{
		if (autoSize != value)
		{
			autoSize = value;
			_pendingTextChange = true;
			checkImmediateChanges();
		}
		
		return autoSize;
	}
	
	private function set_size(value:Float):Float
	{
		var tmp:Float = Math.abs(value);
		
		if (tmp != size)
		{
			size = tmp;
			_pendingTextGlyphsChange = true;
			_pendingBorderGlyphsChange = true;
			_pendingTextChange = true;
			checkImmediateChanges();
		}
		
		return value;
	}
	
	private function set_padding(value:Int):Int
	{
		if (value != padding)
		{
			padding = value;
			_pendingTextChange = true;
			checkImmediateChanges();
		}
		
		return value;
	}
	
	private function set_numSpacesInTab(value:Int):Int 
	{
		if (numSpacesInTab != value && value > 0)
		{
			numSpacesInTab = value;
			_tabSpaces = "";
			
			for (i in 0...value)
			{
				_tabSpaces += " ";
			}
			
			_pendingTextChange = true;
			checkImmediateChanges();
		}
		
		return value;
	}
	
	private function set_background(value:Bool):Bool
	{
		if (background != value)
		{
			background = value;
			_pendingGraphicChange = true;
			checkImmediateChanges();
		}
		
		return value;
	}
	
	private function set_backgroundColor(value:UInt):UInt 
	{
		if (backgroundColor != value)
		{
			backgroundColor = value;
			_pendingGraphicChange = true;
			checkImmediateChanges();
		}
		
		return value;
	}
	
	private function set_borderStyle(style:TextBorderStyle):TextBorderStyle
	{		
		if (style != borderStyle)
		{
			borderStyle = style;
			_pendingBorderGlyphsChange = true;
			checkImmediateChanges();
		}
		
		return borderStyle;
	}
	
	private function set_borderColor(value:UInt):UInt 
	{
		if (borderColor != value)
		{
			borderColor = value;
			_pendingBorderGlyphsChange = true;
			checkImmediateChanges();
		}
		
		return value;
	}
	
	private function set_borderSize(value:Float):Float
	{
		if (value != borderSize)
		{			
			borderSize = value;
			
			if (borderStyle != TextBorderStyle.NONE)
			{
				_pendingGraphicChange = true;
				checkImmediateChanges();
			}
		}
		
		return value;
	}
	
	private function set_borderQuality(value:Float):Float
	{
		value = Math.min(1, Math.max(0, value));
		
		if (value != borderQuality)
		{
			borderQuality = value;
			
			if (borderStyle != TextBorderStyle.NONE)
			{
				_pendingGraphicChange = true;
				checkImmediateChanges();
			}
		}
		
		return value;
	}
	
	private function get_numLines():Int
	{
		return _lines.length;
	}
	
	/**
	 * Calculates maximum width of the text.
	 * 
	 * @return	text width.
	 */
	private function get_textWidth():Float
	{
		var max:Float = 0;
		var numLines:Int = _lines.length;
		var lineWidth:Float;
		_linesWidth = [];
		
		for (i in 0...numLines)
		{
			lineWidth = getLineWidth(i);
			_linesWidth[i] = lineWidth;
			max = Math.max(max, lineWidth);
		}
		
		return max;
	}
	
	private function get_textHeight():Float
	{
		return (lineHeight + lineSpacing) * _lines.length - lineSpacing;
	}
	
	private function get_lineHeight():Float
	{
		return font.lineHeight * size;
	}
	
	private function set_updateImmediately(value:Bool):Bool
	{
		if (updateImmediately != value)
		{
			updateImmediately = value;
			if (value)
			{
				checkPendingChanges();
			}
		}
		
		return value;
	}
	
	private function set_smoothing(value:Bool):Bool
	{
		#if RENDER_BLIT
		_bitmap.smoothing = value;
		#else
		if (smoothing != value)
		{
			_pendingGraphicChange = true;
			checkImmediateChanges();
		}
		#end
		
		return smoothing = value;
	}
	
	private function updateTextGlyphs():Void
	{
		#if RENDER_BLIT
		if (font == null)	return;
		
		if (textGlyphs != null)
		{
			textGlyphs.dispose();
		}
		textGlyphs = font.prepareGlyphs(size, textColor, useTextColor, smoothing);
		#end
		
		_pendingTextGlyphsChange = false;
		_pendingGraphicChange = true;
	}
	
	private function updateBorderGlyphs():Void
	{
		#if RENDER_BLIT
		if (font != null && (borderGlyphs == null || borderColor != borderGlyphs.color || size != borderGlyphs.scale || font != borderGlyphs.font))
		{
			if (borderGlyphs != null)
			{
				borderGlyphs.dispose();
			}
			borderGlyphs = font.prepareGlyphs(size, borderColor, true, smoothing);
		}
		#end
		
		_pendingBorderGlyphsChange = false;
		_pendingGraphicChange = true;
	}
}