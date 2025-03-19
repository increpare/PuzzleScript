let unitTesting=false;
let lazyFunctionGeneration=true;
let curlevel=0;
let curlevelTarget=null;
let hasUsedCheckpoint=false;
let levelEditorOpened=false;
let muted=0;
let runrulesonlevelstart_phase=false;
let ignoreNotJustPressedAction=true;
let textMode = true;

function doSetupTitleScreenLevelContinue(){
    try {
        if (storage_has(document.URL)) {
            if (storage_has(document.URL+'_checkpoint')){
                let backupStr = storage_get(document.URL+'_checkpoint');
                curlevelTarget = JSON.parse(backupStr);
                
                let arr = [];
                for(let p in Object.keys(curlevelTarget.dat)) {
                    arr[p] = curlevelTarget.dat[p];
                }
                curlevelTarget.dat = new Int32Array(arr);

            }
            curlevel = storage_get(document.URL); 
        }
    } catch(ex) {
    }
}

doSetupTitleScreenLevelContinue();


let verbose_logging=false;
let throttle_movement=false;
let cache_console_messages=false;
let quittingTitleScreen=false;
let quittingMessageScreen=false;
let deltatime=17; // this gets updated every frame; see loop()
let timer=0;
let repeatinterval=150;
let autotick=0;
let autotickinterval=0;
let winning=false;
let againing=false;
let againinterval=150;
let norepeat_action=false;
let oldflickscreendat=[];//used for buffering old flickscreen/scrollscreen positions, in case player vanishes
let keybuffer = [];

let restarting=false;

let messageselected=false;

let textImages={};

let level = new Level(); //just give it some starting state



var WORKLIST_OBJECTS_TO_GENERATE_FUNCTIONS_FOR = [];
function tick_lazy_function_generation(iterative_generation=false){
	if (WORKLIST_OBJECTS_TO_GENERATE_FUNCTIONS_FOR.length===0){
		return;
	}
	// spent a maximum of 10ms on lazy function generation
	let start = performance.now();
	var generated_count=0;
	while (performance.now() - start < 10 || !iterative_generation) {
		if (WORKLIST_OBJECTS_TO_GENERATE_FUNCTIONS_FOR.length > 0) {
			const object = WORKLIST_OBJECTS_TO_GENERATE_FUNCTIONS_FOR.shift();
            //depending on type of object
            //if CellPattern, call generateMatchFunction
            //if Rule, call generate_all_MatchFunctions
            if (object instanceof CellPattern) {
                object.matches = object.generateMatchFunction();
            } else if (object instanceof Rule) {
                object.generate_all_MatchFunctions();
            } else {
                throw new Error("Unknown object type: " + object);
            }
			generated_count++;
		}
	}
	window.console.log("generated "+generated_count+" match functions");
}

function lazy_function_generation_clear_backlog(){
    WORKLIST_OBJECTS_TO_GENERATE_FUNCTIONS_FOR = [];
}
