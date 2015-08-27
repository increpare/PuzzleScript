import org.si.sion.*;

import openfl.display.*;
import openfl.events.*;

class Bosca extends Sprite {
	public function new(stage:Stage) {
		super();
		
		driver.addEventListener(Event.ADDED_TO_STAGE, onAddedToStage);
		driver.addEventListener(Event.REMOVED_FROM_STAGE, onRemovedFromStage);
		stage.addChild(driver);
	}
	
	private function onAddedToStage (event:Event):Void {		
		data = driver.compile("t134l16c-d+c-f+c-d+c-f+c-d+c-f+c-d+c-f+c-d+c-f+c-d+c-f+c-d+c-f+c-d+c-f+c-ec-gc-ec-gc-ec-gc-ec-gc-ec-gc-ec-gc-ec-gc-ec-gc-d+c-f+c-d+c-f+c-d+c-f+c-d+c-f+c-d+c-f+c-d+c-f+c-d+c-f+c-d+c-f+c-ec-gc-ec-gc-ec-");
		driver.play(data);
	}

	private function onRemovedFromStage (event:Event):Void {
		driver.stop();
	}
		
	public var driver:SiONDriver = new SiONDriver();
  public var data:SiONData;
}