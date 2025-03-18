let unitTesting=false;
let curlevel=0;
let curlevelTarget=null;
let hasUsedCheckpoint=false;
let levelEditorOpened=false;
let muted=0;
let runrulesonlevelstart_phase=false;
let ignoreNotJustPressedAction=true;

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
