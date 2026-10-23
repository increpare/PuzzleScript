'use strict';

let unitTesting = false;
let lazyFunctionGeneration = true;
let curlevel = 0;
let curlevelTarget = null;
let hasUsedCheckpoint = false;
let levelEditorOpened = false;
let muted = 0;
let runrulesonlevelstart_phase = false;
let ignoreNotJustPressedAction = true;
let textMode = true;

function doSetupTitleScreenLevelContinue() {
    try {
        if (storage_has(document.URL)) {
            if (storage_has(document.URL + '_checkpoint')) {
                let backupStr = storage_get(document.URL + '_checkpoint');
                curlevelTarget = JSON.parse(backupStr);

                let arr = [];
                for (let p in Object.keys(curlevelTarget.dat)) {
                    arr[p] = curlevelTarget.dat[p];
                }
                curlevelTarget.dat = new Int32Array(arr);

            }
            curlevel = storage_get_int(document.URL, 0);
        }
    } catch (ex) {
        console.warn("Failed to restore checkpoint:", ex);
    }
}

doSetupTitleScreenLevelContinue();


let verbose_logging = false;
let throttle_movement = false;
let suppress_all_console_output = false;
let cache_console_messages = false;
let quittingTitleScreen = false;
let quittingMessageScreen = false;
let deltatime = 17; // this gets updated every frame; see loop()
let timer = 0;
let repeatinterval = 150;
let autotick = 0;
let autotickinterval = 0;
let winning = false;
let againing = false;
let againinterval = 150;
let norepeat_action = false;
let oldflickscreendat = [];//used for buffering old flickscreen/scrollscreen positions, in case player vanishes
let keybuffer = [];

let restarting = false;

let messageselected = false;

let textImages = {};

let level = new Level(); //just give it some starting state


function get_title_animation_frame() {
    return Math.floor(((timer / 1000) / 0.3) * 10)
}

var WORKLIST_OBJECTS_TO_GENERATE_FUNCTIONS_FOR = [];
var WORKLIST_GENERATION_INDEX = 0;
function tick_lazy_function_generation(iterative_generation = false) {
    if (WORKLIST_OBJECTS_TO_GENERATE_FUNCTIONS_FOR.length === 0) {
        return;
    }
    // Spend up to 20ms on lazy function generation before starting another item.
    let start = performance.now();
    while (
        ((performance.now() - start < 20) || !iterative_generation)
        && WORKLIST_GENERATION_INDEX < WORKLIST_OBJECTS_TO_GENERATE_FUNCTIONS_FOR.length) {
        const object = WORKLIST_OBJECTS_TO_GENERATE_FUNCTIONS_FOR[WORKLIST_GENERATION_INDEX++];
        // Clear before generating the last item so appends and errors see an empty queue.
        if (WORKLIST_GENERATION_INDEX === WORKLIST_OBJECTS_TO_GENERATE_FUNCTIONS_FOR.length) {
            lazy_function_generation_clear_backlog();
        }
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
    }
}

function lazy_function_generation_clear_backlog() {
    WORKLIST_OBJECTS_TO_GENERATE_FUNCTIONS_FOR = [];
    WORKLIST_GENERATION_INDEX = 0;
}

