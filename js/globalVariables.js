var unitTesting=false;
var curlevel=0;
var curlevelTarget=null;
var levelEditorOpened=false;


var compiling = false;
var errorStrings = [];
var errorCount=0;

var canv = 
document.oncontextmenu = function (e) {
    if (e.target.tagName=="CANVAS"){
        e.preventDefault();
    }
};

try {
 	if (!!window.localStorage) { 
		if (localStorage[document.URL]!==undefined) {
            if (localStorage[document.URL+'_checkpoint']!==undefined){
                curlevelTarget = JSON.parse(localStorage[document.URL+'_checkpoint']);
                
                var arr = [];
                for(var p in Object.getOwnPropertyNames(curlevelTarget.dat)) {
                    arr[p] = curlevelTarget.dat[p];
                }
                curlevelTarget.dat = new Int32Array(arr);

            }
	        curlevel = localStorage[document.URL];            
		}
	}		 
} catch(ex) {
}



var verbose_logging=false;
var throttle_movement=false;
var cache_console_messages=false;
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
var norepeat_action=false;
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



function getTitle(){
    if (metaData.title==""){
        metaData.title="terrylib game";
    }   
    return metaData.title;
}

var metaData = {
    title:"terrylib game",
    homepage:"",
    bgCol:"#000000"
}
function settitle(t){
    metaData.title=t;
    window.console.log("st " +t);

    if (canSetHTMLColors){        
        var link = document.getElementById ("gametitle");
        link.innerText=t;
    }
    document.title = t;
}

function strip_http(url) {
   url = url.replace(/^https?:\/\//,'');
   url = url.replace(/\/*$/,'');
   return url;
}

function qualifyURL(url) {
    var a = document.createElement('a');
    a.href = url;
    return a.href;
}

function sethomepage(t){
    metaData.homepage=t;
    window.console.log("sh " +t);
    if (canSetHTMLColors){        
        var link = document.getElementById ("homepagelink");
        link.href=qualifyURL(metaData.homepage);
        link.innerText=strip_http(link.href);
    }
}

function decimalToHex(d) {
  var hex = Number(d).toString(16);
  hex = "000000".substr(0, 6 - hex.length) + hex; 
  return hex;
}


function setbackgroundcolor(t){
    metaData.bgCol="#"+decimalToHex(t);
    if (canSetHTMLColors){
        var meta = document.getElementById ("openfl-content");
        meta.style.backgroundColor=metaData.bgCol;
        document.body.style.backgroundColor=metaData.bgCol;
    } else {
        var meta = document.getElementById ("openfl-content");
        meta.style.backgroundColor=metaData.bgCol;   
    }
}
