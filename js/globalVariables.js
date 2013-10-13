var unitTesting=false;
var curlevel=0;
var levelEditorOpened=false;

if (localStorage[document.URL]!==undefined) {
	curlevel = localStorage[document.URL];
}

var verbose_logging=false;
var quittingTitleScreen=false;
var quittingMessageScreen=false;
var deltatime=17;
var timer=0;
var repeatinterval=150;
var autotick=0;
var autotickinterval=0;
var winning=false;
var againing=false;
var againinterval=150;
var oldflickscreendat=[];//used for buffering old flickscreen/scrollscreen positions, in case player vanishes
var keybuffer = [];


var messageselected=false;

var textImages={};
var initLevel = {
    width: 5,
    height: 5,
    layerCount: 2,
    dat: [
    1, 3, 3, 1, 1, 2, 2, 3, 3, 1,
    2, 1, 2, 2, 3, 3, 1, 1, 2, 2,
    3, 2, 1, 3, 2, 1, 3, 2, 1, 3,
    1, 3, 3, 1, 1, 2, 2, 3, 3, 1,
    2, 1, 2, 2, 3, 3, 1, 1, 2, 2
    ],
    movementMask:[
    1, 3, 3, 1, 1, 2, 2, 3, 3, 1,
    2, 1, 2, 2, 3, 3, 1, 1, 2, 2,
    3, 2, 1, 3, 2, 1, 3, 2, 1, 3,
    1, 3, 3, 1, 1, 2, 2, 3, 3, 1,
    2, 1, 2, 2, 3, 3, 1, 1, 2, 2
    ],
    rigidGroupIndexMask:[],//[indexgroupNumber, masked by layer arrays]
    rigidMovementAppliedMask:[],//[indexgroupNumber, masked by layer arrays]
    bannedGroup:[],
    colCellContents:[],
    rowCellContents:[]
};

var level = initLevel;
