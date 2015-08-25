package terrylibweb;

import openfl.display.DisplayObject;
import openfl.events.KeyboardEvent;
import openfl.ui.Keyboard;

enum Keystate {
  justreleased;
	notpressed;
	pressed;
	justpressed;
}

class Input {
	public static function pressed(k:Key):Bool {
		return keyheld[keymap.get(k)];
	}
	
	public static function justpressed(k:Key):Bool { 
		if (current[keymap.get(k)] == Keystate.justpressed) {
			//current[keymap.get(k)] = Keystate.pressed;
			return true;
		}else {
			return false;
		}
	}
	
	public static function justreleased(k:Key):Bool { 
		if (current[keymap.get(k)] == Keystate.justreleased) {
			current[keymap.get(k)] = Keystate.notpressed;
			return true;
		}else {
			return false;
		}
	}
	
	public static function delaypressed(k:Key, delay:Int):Bool {
		keycode = keymap.get(k);
		if (keyheld[keycode]) {
			if (keydelay[keycode] <= 0) {
				keydelay[keycode] = delay;
				return true;
			}else {
		    keydelay[keycode]--;
				return false;
			}
		}else {
			keydelay[keycode] = 0;
		}
		return false;
	}
	
	private static function init(stage:DisplayObject){
		stage.addEventListener(KeyboardEvent.KEY_DOWN, handlekeydown);
		stage.addEventListener(KeyboardEvent.KEY_UP, handlekeyup);
		
		//BASIC STORAGE & TRACKING			
		var i:Int = 0;
		for(i in 0...numletters){
			current.push(Keystate.notpressed);
			last.push(Keystate.notpressed);
			keyheld.push(false);
		}
		
		//LETTERS
		addkey(Key.A, Keyboard.A);
		addkey(Key.B, Keyboard.B);
		addkey(Key.C, Keyboard.C);
		addkey(Key.D, Keyboard.D);
		addkey(Key.E, Keyboard.E);
		addkey(Key.F, Keyboard.F);
		addkey(Key.G, Keyboard.G);
		addkey(Key.H, Keyboard.H);
		addkey(Key.I, Keyboard.I);
		addkey(Key.J, Keyboard.J);
		addkey(Key.K, Keyboard.K);
		addkey(Key.L, Keyboard.L);
		addkey(Key.M, Keyboard.M);
		addkey(Key.N, Keyboard.N);
		addkey(Key.O, Keyboard.O);
		addkey(Key.P, Keyboard.P);
		addkey(Key.Q, Keyboard.Q);
		addkey(Key.R, Keyboard.R);
		addkey(Key.S, Keyboard.S);
		addkey(Key.T, Keyboard.T);
		addkey(Key.U, Keyboard.U);
		addkey(Key.V, Keyboard.V);
		addkey(Key.W, Keyboard.W);
		addkey(Key.X, Keyboard.X);
		addkey(Key.Y, Keyboard.Y);
		addkey(Key.Z, Keyboard.Z);
		
		//NUMBERS
		addkey(Key.ZERO,Keyboard.NUMBER_0);
		addkey(Key.ONE,Keyboard.NUMBER_1);
		addkey(Key.TWO,Keyboard.NUMBER_2);
		addkey(Key.THREE,Keyboard.NUMBER_3);
		addkey(Key.FOUR,Keyboard.NUMBER_4);
		addkey(Key.FIVE,Keyboard.NUMBER_5);
		addkey(Key.SIX,Keyboard.NUMBER_6);
		addkey(Key.SEVEN,Keyboard.NUMBER_7);
		addkey(Key.EIGHT,Keyboard.NUMBER_8);
		addkey(Key.NINE,Keyboard.NUMBER_9);
		
		//FUNCTION KEYS
		addkey(Key.F1,Keyboard.F1);
		addkey(Key.F2,Keyboard.F2);
		addkey(Key.F3,Keyboard.F3);
		addkey(Key.F4,Keyboard.F4);
		addkey(Key.F5,Keyboard.F5);
		addkey(Key.F6,Keyboard.F6);
		addkey(Key.F7,Keyboard.F7);
		addkey(Key.F8,Keyboard.F8);
		addkey(Key.F9,Keyboard.F9);
		addkey(Key.F10,Keyboard.F10);
		addkey(Key.F11,Keyboard.F11);
		addkey(Key.F12,Keyboard.F12);
		
		//SPECIAL KEYS + PUNCTUATION
		addkey(Key.ESCAPE,Keyboard.ESCAPE);
		addkey(Key.MINUS,Keyboard.MINUS);
		addkey(Key.PLUS,Keyboard.EQUAL);
		addkey(Key.DELETE,Keyboard.DELETE);
		addkey(Key.BACKSPACE,Keyboard.BACKSPACE);
		addkey(Key.LBRACKET,Keyboard.LEFTBRACKET);
		addkey(Key.RBRACKET,Keyboard.RIGHTBRACKET);
		addkey(Key.BACKSLASH,Keyboard.BACKSLASH);
		addkey(Key.CAPSLOCK,Keyboard.CAPS_LOCK);
		addkey(Key.SEMICOLON,Keyboard.SEMICOLON);
		addkey(Key.QUOTE,Keyboard.QUOTE);
		addkey(Key.ENTER,Keyboard.ENTER);
		addkey(Key.SHIFT,Keyboard.SHIFT);
		addkey(Key.COMMA,Keyboard.COMMA);
		addkey(Key.PERIOD,Keyboard.PERIOD);
		addkey(Key.SLASH,Keyboard.SLASH);
		addkey(Key.CONTROL,Keyboard.CONTROL);
		addkey(Key.ALT, 18);
		addkey(Key.SPACE,Keyboard.SPACE);
		addkey(Key.UP,Keyboard.UP);
		addkey(Key.DOWN,Keyboard.DOWN);
		addkey(Key.LEFT,Keyboard.LEFT);
		addkey(Key.RIGHT, Keyboard.RIGHT);
	}
	
	private static function update(){
		for (i in 0 ... numletters) {
			if (lookup.exists(i)) {
				if ((last[i] == Keystate.justreleased) && (current[i] == Keystate.justreleased)) current[i] = Keystate.notpressed;
				else if ((last[i] == Keystate.justpressed) && (current[i] == Keystate.justpressed)) current[i] = Keystate.pressed;
				last[i] = current[i];
			}
		}
	}
	
	private static function reset(){
		for (i in 0...numletters) {
			if (lookup.exists(i)) {
				current[i] = Keystate.notpressed;
				last[i] = Keystate.notpressed;
				keyheld[i] = false;
			}
		}
	}
	
	private static function iskeycodeheld(k:Keystate):Bool {
		if (k == Keystate.justpressed || k == Keystate.pressed) {
			return true;
		}
		return false;
	}
	
	private static function handlekeydown(event:KeyboardEvent) {
		keycode = event.keyCode;
		
		if (lookup.exists(keycode)) {
			if (iskeycodeheld(current[keycode])) {
				current[keycode] = Keystate.pressed;
			}else {
				current[keycode] = Keystate.justpressed;
				keydelay[keycode] = 0;
			}
			keyheld[keycode] = true;
		}
	}
	
	private static function handlekeyup(event:KeyboardEvent) {
		keycode = event.keyCode;
		if (lookup.exists(keycode)) {
			if (iskeycodeheld(current[keycode])) {
				current[keycode] = Keystate.justreleased;
			}else {
				current[keycode] = Keystate.notpressed;
			}
			keyheld[keycode] = false;
		}
	}
	
	private static function addkey(KeyName:Key, KeyCode:Int) {
		keymap.set(KeyName, KeyCode);
		lookup.set(KeyCode, KeyName);
		current[KeyCode] = Keystate.notpressed;
		last[KeyCode] = Keystate.notpressed;
		keyheld[KeyCode] = false;
	}
	
	private static var keymap:Map<Key, Int> = new Map<Key, Int>();
	private static var lookup:Map<Int, Key> = new Map<Int, Key>();
	private static var current:Array<Keystate> = new Array<Keystate>();
	private static var last:Array<Keystate> = new Array<Keystate>();
	private static var keydelay:Array<Int> = new Array<Int>();
	private static var keyheld:Array<Bool> = new Array<Bool>();
	
	private static var numletters:Int = 256;
	private static var keycode:Int;
}
