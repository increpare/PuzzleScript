package terrylib;

import openfl.display.*;          
import openfl.media.*; 
import openfl.events.*;
import openfl.Assets;

class Music {
	//Play a sound effect! There are 16 channels, which iterate
	public static function playsound(t:String, offset:Int = 0) {
		temptransform = new SoundTransform(volumelevels[Std.int(effectindex.get(t))] * globalsound);
		efchannel[currentefchan] = efchan[Std.int(effectindex.get(t))].play(offset);
		efchannel[currentefchan].soundTransform = temptransform;
		currentefchan++;
		if (currentefchan > 15) currentefchan -= 16;
	}
	
	public static function stopsound() {
		temptransform = new SoundTransform(0);
		
		for (i in 0 ... 16) {
			if (efchannel[i] != null) efchannel[i].soundTransform = temptransform;
		}
	}
	
	public static function loadsound(t:String, vol:Float = 1.0) {
		effectindex.set(t, numeffects);
		volumelevels.push(vol);
		#if flash
		efchan.push(Assets.getSound("data/sounds/" + t + ".mp3")); 
		#else
		efchan.push(Assets.getSound("data/sounds/" + t + ".ogg")); 
		#end
		numeffects++;
	}
	
	public static function loadsong(t:String, vol:Float = 1.0) {	
		songindex.set(t, numsongs);
		songvolumelevels.push(vol);
		#if flash
		musicchan.push(Assets.getMusic("data/music/" + t + ".mp3"));
		#else
		musicchan.push(Assets.getMusic("data/music/" + t + ".ogg"));
		#end
		numsongs++;
	}
	
	public static function play(t:String, time:Int = 0) {
		if (currentsong !=t) {
			if (currentsong != "nothing") {
				//Stop the old song first
				musicchannel.stop();
				musicchannel.removeEventListener(Event.SOUND_COMPLETE, loopmusic);
			}
			if (t != "nothing") {
				currentsong = t;
				
				musicchannel = musicchan[Std.int(songindex.get(t))].play((time * 1000) % musicchan[Std.int(songindex.get(t))].length);
				musicchannel.soundTransform = new SoundTransform(songvolumelevels[Std.int(songindex.get(t))] * globalsound);
				
				musicchannel.addEventListener(Event.SOUND_COMPLETE, loopmusic);
			}else {	
				currentsong = "nothing";
			}
		}
	}   
	
	public static function stop() { 
		musicchannel.removeEventListener(Event.SOUND_COMPLETE, stopmusic);
		musicchannel.stop();
		currentsong = "nothing";
	}
	
	public static function fadeout() { 
		if (musicfade == 0) {
			musicfade = 31;
		}
	}
	
	private static function init(){
		currentsong = "nothing"; musicfade = 0;//no music, no amb
		currentefchan = 0;
		usingtickertext = false;
		
		globalsound = 1; muted = false;
		
		numplays = 0;
		numeffects = 0;
		numsongs = 0;
	}
	
	private static function loopmusic(e:Event) { 
		musicchannel.removeEventListener(Event.SOUND_COMPLETE, loopmusic);
		if (currentsong != "nothing") {
			musicchannel = musicchan[Std.int(songindex.get(currentsong))].play();
			musicchannel.soundTransform = new SoundTransform(songvolumelevels[Std.int(songindex.get(currentsong))] * globalsound);
				
			musicchannel.addEventListener(Event.SOUND_COMPLETE, loopmusic);
		}
	}
	
	private static function stopmusic(e:Event) { 
		musicchannel.removeEventListener(Event.SOUND_COMPLETE, stopmusic);
		musicchannel.stop();
		currentsong = "nothing";
	}
	
	private static function processmusicfade() {
		musicfade--;
		if (musicfade > 0) {
			musicchannel.soundTransform = new SoundTransform((musicfade / 30) * globalsound);
		}else {
			musicchannel.stop();
			currentsong = "nothing";
		}
	}
	
	private static function processmusicfadein() {
		musicfadein--;
		if (musicfadein > 0) {
			musicchannel.soundTransform = new SoundTransform(((60-musicfadein) / 60 )*globalsound);
		}else {
			musicchannel.soundTransform = new SoundTransform(1.0 * globalsound);
		}
	}
	
	private static function processmusic() {
		if (musicfade > 0) processmusicfade();
		if (musicfadein > 0) processmusicfadein();
	}
	
	private static function updateallvolumes() {
		//Update the volume levels of all currently playing sounds.
		//Music:
		if(currentsong!="nothing"){
			musicchannel.soundTransform = new SoundTransform(songvolumelevels[Std.int(songindex.get(currentsong))] * globalsound);
		}
		//Sound effects
		//Figure this out someday I guess?
	}
	
	private static function processmute() {
		if (Text.input_show == 0) {
			if (Input.justpressed(Key.M) && mutebutton <= 0) {
				mutebutton = 2; if (muted) { muted = false; }else { muted = true;}
			}
			if (mutebutton > 0 && !Input.pressed(Key.M)) mutebutton--;
		}
		
		if (muted) {
			if (globalsound == 1) {
			  globalsound = 0;
				updateallvolumes();
			}
		}
		
		if (!muted && globalsound < 1) {
			globalsound += 0.05; 
			if (globalsound > 1.0) globalsound = 1.0;
			updateallvolumes();
		}
	}
	
	public static var musicchan:Array<Sound> = new Array<Sound>();	
	public static var musicchannel:SoundChannel;
	public static var currentsong:String;
	public static var musicfade:Int;
	public static var musicfadein:Int;
	
	public static var effectindex:Map<String, Int> = new Map<String, Int>();
	public static var volumelevels:Array<Float> = new Array<Float>();
	public static var numeffects:Int;
	
	public static var songindex:Map<String, Int> = new Map<String, Int>();
	public static var songvolumelevels:Array<Float> = new Array<Float>();
	public static var numsongs:Int;
	
	public static var currentefchan:Int;
	public static var efchannel:Array<SoundChannel> = new Array<SoundChannel>();
	public static var efchan:Array<Sound> = new Array<Sound>();
	public static var numplays:Int;
	
	public static var usingtickertext:Bool;
	
	public static var temptransform:SoundTransform;
	public static var globalsound:Float;
	public static var muted:Bool; 
	public static var mutebutton:Int;
}