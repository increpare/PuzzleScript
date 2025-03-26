// CodeMirror, copyright (c) by Marijn Haverbeke and others
// Distributed under an MIT license: http://codemirror.net/LICENSE
(function(mod) {
    if (typeof exports == "object" && typeof module == "object") // CommonJS
        mod(require("../../lib/codemirror"));
    else if (typeof define == "function" && define.amd) // AMD
        define(["../../lib/codemirror"], mod);
    else // Plain browser env
        mod(CodeMirror);
})(function(CodeMirror) {
        "use strict";

        var WORD = /[\w$#>-]+/,
            RANGE = 500;

        var PRELUDE_COMMAND_WORDS = [
            "METADATA",//tag
            ["again_interval", "0.1"],
            ["author", "Gill Bloggs"],
            ["background_color", "blue"],
            ["color_palette", "arne"],
            ["debug", ""],
            ["flickscreen", "8x5"],
            ["homepage", "www.puzzlescript.net"],
            ["key_repeat_interval", "0.1"],
            ["noaction", ""],
            ["norepeat_action", ""],
            ["norestart", ""],
            ["noundo", ""],
            ["realtime_interval", ""],
            ["require_player_movement", ""],
            ["run_rules_on_level_start", ""],
            ["scanline", ""],
            ["text_color", "orange"],
            ["throttle_movement", ""],
            ["title", "My Amazing Puzzle Game"],
            ["verbose_logging", ""],
            ["zoomscreen", "WxH"]                    
        ];

        var COLOR_WORDS = [
            "COLOR",//special tag
            "black", "blue", "brown", "darkblue", "darkbrown", "darkgray", "darkgreen", "darkred", "gray", "green", "lightblue", "lightbrown", "lightgray", "lightgreen", "lightred", "orange", "pink", "purple", "red", "transparent", "white", "yellow"];
        var RULE_COMMAND_WORDS = [
            "COMMAND",
            //sfx added manually based on definitions
            "again", "cancel", "checkpoint", "message", "restart", "win"];
        var SFX_COMMAND_LIST = ["sfx0", "sfx1", "sfx2", "sfx3", "sfx4", "sfx5", "sfx6", "sfx7", "sfx8", "sfx9", "sfx10"];

        var CARDINAL_DIRECTION_WORDS = [
            "DIRECTION",
            "up","down","left","right","horizontal","vertical"]

        var RULE_DIRECTION_WORDS = [
            "DIRECTION",//tag
            "up", "down", "left", "right", "random", "horizontal", "vertical","late","rigid"]

        var LOOP_WORDS = [
            "BRACKET",//tag
            "startloop","endloop"]
            
        var PATTERN_DIRECTION_WORDS = [
            "DIRECTION",
            "up", "down", "left", "right", "moving", "stationary", "no", "randomdir", "random", "horizontal", "vertical", "orthogonal", "perpendicular", "parallel", "action"]


        var SOUND_EVENTS = [
            "SOUNDEVENT",
            "cancel", "closemessage", "endgame", "endlevel", "restart", "showmessage", "startgame", "startlevel", "titlescreen", "undo", "sfx0", "sfx1", "sfx2", "sfx3", "sfx4", "sfx5", "sfx6", "sfx7", "sfx8", "sfx9", "sfx10"
        ];

        var SOUND_VERBS = [
            "SOUNDVERB",
            "action", "cantmove", "create", "destroy", "move"
        ];

        var SOUND_DIRECTIONS = [
            "DIRECTION",
            "up","down","left","right","horizontal","vertical","orthogonal"]

        var WINCONDITION_WORDS = [
            "LOGICWORD",
            "some", "on", "no", "all"]

        var LEGEND_LOGICWORDS = [
                "LOGICWORD",
                "and","or"
            ]

        var PRELUDE_COLOR_PALETTE_WORDS = [
            "amiga", "amstrad", "arnecolors", "atari", "c64", "ega", "famicom", "gameboycolour", "mastersystem", "pastel", "proteus_mellow", "proteus_night", "proteus_rich", "whitingjp"
        ]

        function renderHint(elt,data,cur){
            var t1=cur.text;
            var t2=cur.extra;
            var tag=cur.tag;
            if (t1.length==0){
                t1=cur.extra;
                t2=cur.text;
            }
            var wrapper = document.createElement("span")
            wrapper.className += " cm-s-midnight ";

            var h = document.createElement("span")                // Create a <h1> element
            // h.style.color="white";
            var t = document.createTextNode(t1);     // Create a text node

            h.appendChild(t);   
            wrapper.appendChild(h); 

            if (tag!=null){
                h.className += "cm-" + tag;
            }

            elt.appendChild(wrapper);//document.createTextNode(cur.displayText || getText(cur)));

            if (t2.length>0){
                var h2 = document.createElement("span")                // Create a <h1> element
                h2.style.color="orange";
                var t2 = document.createTextNode(" "+t2);     // Create a text node
                h2.appendChild(t2);  
                h2.style.color="orange";
                elt.appendChild(t2);
            }
        }

        //rotates clockwise 90 degrees n times
        function rotate_2d_array(array,amount){
            amount = amount %4;
            switch(amount){                    
                case 0:{
                    return array;
                }
                case 1:{
                    let result = [];
                    for (let i=0;i<array.length;i++) {
                        let new_row = "";
                        for (let j=0;j<array[i].length;j++) {
                            new_row+=array[j][i];
                        }
                        result.push(new_row);
                    }
                    return result;
                }
                case 2:{
                    //reverse both array and all the strings it contains
                    let result = array.reverse().map(row => row.split('').reverse().join(''));
                    return result;
                }
                case 3:{
                    //rotate 3 times
                    let result = [];
                    for (let i=0;i<array.length;i++) {
                        let new_row = "";
                        for (let j=0;j<array[i].length;j++) {
                            new_row+=array[array.length-1-j][array.length-1-i];
                        }
                        result.push(new_row);
                    }
                    return result;
                }
            }
        }

        CodeMirror.registerHelper("hint", "anyword", function(editor, options) {

            var word = options && options.word || WORD;
            var range = options && options.range || RANGE;
            var cur = editor.getCursor(),
                curLine = editor.getLine(cur.line);

            var end = cur.ch,
                start = end;

            var lineToCursor = curLine.substr(0,end);

            while (start && word.test(curLine.charAt(start - 1))) --start;
            var curWord = start != end && curLine.slice(start, end);

            var tok = editor.getTokenAt(cur);
            var state = tok.state;

            // ignore empty word
            if (!curWord || state.commentLevel>0) {
                // if ( 
                //         ( state.section=="" && curLine.trim()=="")  
                //         // || ( state.section=="objects" && state.objects_section==2 ) 
                //     ) {
                //     curWord="";
                // } else {
                    return {
                        list: []
                    };
                // }            
            }

            // case insensitive
            curWord = curWord.toLowerCase();

            var list = options && options.list || [],
                seen = {};

            var addObjects = false;
            var excludeProperties = false;
            var excludeAggregates = false;
            var candlists = [];
            var toexclude = [];
            switch (state.section) {
                case 'objects':
                    {
                        // if objects.section==1 
                        // * and objects.candname is start of last declared object,
                        // * and the last declared object name ends with Up/UP/up/_u 
                        // then suggest as autocomplete the declaration of the Down/Left/Right objects
                        if (state.objects_section==1){
                            

                            // STEP 1, find name of last declared object
                            // Original_line_numbers being a dictionary[Name,LineNumber] 
                            // makes this a bit indirect...
                            var object_names_lowercase = Object.keys(state.original_line_numbers);
                            if (object_names_lowercase.length==0){
                                break;
                            }
                            //in state.original_linenumbers, find the line number of the last declared object
                            let max_line_number_name_lowercase = object_names_lowercase[0];
                            let max_line_number = state.original_line_numbers[max_line_number_name_lowercase];
                            for (var i=1;i<object_names_lowercase.length;i++){
                                const this_name = object_names_lowercase[i]
                                const this_line_number = state.original_line_numbers[this_name];
                                if (this_line_number!==state.lineNumber && this_line_number>max_line_number){
                                    max_line_number_name_lowercase = this_name;
                                    max_line_number = this_line_number;
                                }
                            }

                            var previous_object_data = state.objects[max_line_number_name_lowercase];
                            var previous_object_casename = state.original_case_names[max_line_number_name_lowercase];

                            // only bother with all this if the curword is a prefix 
                            // of max_line_number_name_lowercase
                            if (!max_line_number_name_lowercase.startsWith(curWord)){
                                break;
                            }

                            const pairings = [ ["Up",["Down","Left","Right"]],
                                                ["UP",["DOWN","LEFT","RIGHT"]],
                                                ["up",["down","left","right"]],
                                                ["_u",["_d","_l","_r"]],
                                                ["_U",["_D","_L","_R"]] ];
                            for (var i=0;i<pairings.length;i++){
                                const suffix = pairings[i][0];
                                // STEP 2, if casename ends with suffix, suggest the corresonding pairings
                                if (previous_object_casename.endsWith(suffix)){
                                    //FOUND a match.
                                    let to_suggest = "";
                                    const stem = previous_object_casename.slice(0,-suffix.length);
                                    const further_endings = pairings[i][1];
                                    const previous_object = state.objects[max_line_number_name_lowercase];

                                    //generate rotations of previous_object
                                    const rotations = [ 
                                        //DOWN
                                        rotate_2d_array(previous_object.spritematrix,2),
                                        //LEFT
                                        rotate_2d_array(previous_object.spritematrix,3),
                                        //RIGHT
                                        rotate_2d_array(previous_object.spritematrix,1),
                                    ];

                                    for (let j=0;j<further_endings.length;j++){
                                        const pairing = further_endings[j];
                                        to_suggest += stem+pairing+"\n";
                                        //print colros
                                        to_suggest += previous_object.colors.join(" ")+"\n";

                                        if (previous_object.spritematrix.length>0){
                                            to_suggest += rotations[j].join("\n");
                                            to_suggest += "\n";
                                        } 
                                        to_suggest += "\n";
                                    }
                                    candlists.push(["comment",to_suggest]);
                                }                                    
                            }                                                                                                                    
                        } else if (state.objects_section==2){
                            candlists.push(COLOR_WORDS);
                        }
                        break;
                    }
                case 'legend':
                    {
                        var splits = lineToCursor.toLowerCase().split(/[\p{Z}\s]/u).filter(function(v) {
                            return v !== '';
                        });
                        toexclude=splits.filter(a => LEGEND_LOGICWORDS.indexOf(a)===-1);//don't filter out and or or
                        if (lineToCursor.indexOf('=')>=0){
                            if ((lineToCursor.trim().split(/\s+/ ).length%2)===1){
                                addObjects=true;
                            } else {
                                candlists.push(LEGEND_LOGICWORDS);                      
                            }
                        } //no hints before equals
                        break;
                    }
                case 'sounds':
                    {
                        /*
                        SOUNDEVENT SOUND 
                        NAME
                            SOUNDVERB <SOUND>
                            SOUNDVERB
                                <SOUND>
                                DIRECTION+ <SOUND>
                                */
                        var last_idx = state.current_line_wip_array.length-1;
                        if (last_idx>0 && state.current_line_wip_array[last_idx]==="ERROR"){
                            //if there's an error, just try to match greedily
                            candlists.push(SOUND_VERBS);
                            candlists.push(SOUND_DIRECTIONS);
                            candlists.push(SOUND_EVENTS);
                            addObjects=true;
                            excludeAggregates=true;       
                        } else if (state.current_line_wip_array.length<=1 ){
                            candlists.push(SOUND_EVENTS);
                            addObjects=true;
                            excludeAggregates=true;                            
                        } else  {
                            var lastType =  state.current_line_wip_array[last_idx][1];
                            switch (lastType){
                                case "SOUNDEVENT":
                                    {
                                        break;
                                    }
                                case "NAME":
                                    {
                                        candlists.push(SOUND_VERBS);
                                        break;
                                    }
                                case "SOUNDVERB":
                                case "DIRECTION":
                                    {
                                        candlists.push(SOUND_DIRECTIONS);
                                        break;
                                    }
                                case "SOUND":
                                    {
                                    }
                            }                                                 
                        }
                        break;
                    }
                case 'collisionlayers':
                    {
                        var splits = lineToCursor.toLowerCase().split(/[,\p{Z}\s]/u).filter(function(v) {
                            return v !== '';
                        });
                        toexclude=splits;
                        addObjects=true;
                        excludeAggregates=true;
                        break;
                    }
                case 'rules':
                    {   

                        /* rules look like this: 
                           blah blah [ a | b | c d ] [ e f | g ] -> [ a | b | c d ] [ e f | g ]
                            so, if the curword is ->, we want to show the commands that can follow that, mirroring the first half o the line
                        */

                        if (curWord==="->" || curWord==="-"){
                            const line_until_curword = curLine.substring(0,start);
                            var previous_arrow_idx = line_until_curword.indexOf("->");
                            //if there is a previous arrow on the line, don't suggest anything.
                            if (previous_arrow_idx>=0){
                                //nothing
                            } else {
                                
                                //check there's nother other than whitespace to the right of the cursor
                                var right_of_cursor = curLine.substring(cur.ch);
                                if (right_of_cursor.trim().length===0){
                                    //ignore first half until the [
                                    var first_half_start = lineToCursor.indexOf("[");
                                    var first_half_end = lineToCursor.lastIndexOf("]");
                                    var excerpt = lineToCursor.substring(first_half_start,first_half_end+1);
                                    //we should strip all substrings of the form "no XYZ" (case insensitive), removing both the "no" and the word that follows it
                                    var no_words = excerpt.match(/\bno\s+[^\s]+\s*/gi);
                                    if (no_words){
                                        for (var i=0;i<no_words.length;i++){
                                            var no_word = no_words[i];
                                            //repace the whole kaboodle with an empty string
                                            excerpt = excerpt.replace(no_word, "");
                                        }
                                    }
                                    //if we have 'stationary X' on the lhs, we can remove stationary from the rhs
                                    var stationary_words = excerpt.match(/\bstationary\s+/gi);
                                    if (stationary_words){
                                        for (var i=0;i<stationary_words.length;i++){
                                            var stationary_word = stationary_words[i];
                                            //repace the whole kaboodle with an empty string
                                            excerpt = excerpt.replace(stationary_word, "");
                                        }
                                    }
                                    //stripped excerpt - strip everything except for []|.
                                    var stripped_excerpt = excerpt.replace(/[^\[\]\.\|]/g, " ");
                                    //in both excerpt and stripped excerpt, reduce all whitespace to a single space
                                    excerpt = excerpt.replace(/\s+/g, " ");
                                    stripped_excerpt = stripped_excerpt.replace(/\s+/g, " ");
                                    var results = ["LOGICWORD","-> "+excerpt];
                                    if (excerpt!==stripped_excerpt){
                                        results.push("-> "+stripped_excerpt);
                                    }
                                    candlists.push(results);
                                }
                            }
                        }
                        //if inside of roles,can use some extra directions
                        if (lineToCursor.indexOf("[")==-1) {
                            candlists.push(RULE_DIRECTION_WORDS);
                            candlists.push(LOOP_WORDS);
                        } else {
                            candlists.push(PATTERN_DIRECTION_WORDS);                            
                        }
                        
                        if (lineToCursor.indexOf("->")>=0) {
                            var my_commands = RULE_COMMAND_WORDS;
                            for (var i=0;i<SFX_COMMAND_LIST.length;i++){
                                var sfxcommand = SFX_COMMAND_LIST[i];
                                for (var j=0;j<state.sounds.length;j++){
                                    var sfx = state.sounds[j][0][0];
                                    if (sfxcommand===sfx){
                                        my_commands.push(sfxcommand);
                                    }
                                }
                            }
                            candlists.push(RULE_COMMAND_WORDS);
                        }
                        addObjects=true;
                        break;
                    }
                case 'winconditions':
                    {
                        if ((lineToCursor.trim().split(/\s+/ ).length%2)===0){
                            addObjects=true;
                        }
                        candlists.push(WINCONDITION_WORDS);
                        break;
                    }
                case 'levels':
                    {
                        if ("message".indexOf(lineToCursor.trim())===0) {
                            candlists.push(["MESSAGE_VERB","message"]);
                        }
                        break;
                    }
                default: //preamble
                    {
                        var lc = lineToCursor.toLowerCase();
                        if (lc.indexOf("background_color")>=0 ||
                            lc.indexOf("text_color")>=0) {
                            candlists.push(COLOR_WORDS);
                        } else {
                            var linewords =lineToCursor.trim().split(/\s+/ );

                            if (linewords.length<2) {
                                candlists.push(PRELUDE_COMMAND_WORDS);
                            } else if (linewords.length==2 && linewords[0].toLowerCase()=='color_palette'){
                                candlists.push(PRELUDE_COLOR_PALETTE_WORDS);
                            }
                        }

                        break;
                    }
            }

            //first, add objects if needed
            if (addObjects){
                var obs = state.objects;
                for (var key in obs) {
                    if (obs.hasOwnProperty(key)) {
                        var w = key;
                        var matchWord = w.toLowerCase();
                        // if (matchWord === curWord) continue;
                        if ((!curWord || matchWord.lastIndexOf(curWord, 0) == 0) && !Object.prototype.hasOwnProperty.call(seen, matchWord)) {
                            seen[matchWord] = true;
                            var hint = state.original_case_names[w]; 
                            list.push({text:hint,extra:"",tag:"NAME",render:renderHint});
                        }
                    }
                }

                var legendbits = [state.legend_synonyms];
                if (!excludeProperties){
                    legendbits.push(state.legend_properties);
                }
                if (!excludeAggregates){
                    legendbits.push(state.legend_aggregates);
                }

                //go throuhg all derived objects
                for (var i=0;i<legendbits.length;i++){
                    var lr = legendbits[i];
                    for (var j=0;j<lr.length;j++){
                        var w = lr[j][0];
                        var matchWord = w.toLowerCase();
                        // if (matchWord === curWord) continue;
                        if ((!curWord || matchWord.lastIndexOf(curWord, 0) == 0) && !Object.prototype.hasOwnProperty.call(seen, matchWord)) {
                            seen[matchWord] = true;
                            var hint = state.original_case_names[w]; 
                            list.push({text:hint,extra:"",tag:"NAME",render:renderHint});
                        }
                    }
                }

            }

            // go through random names
            for (var i = 0; i < candlists.length; i++) {
                var candlist = candlists[i]
                var tag = candlist[0];
                for (var j = 1; j < candlist.length; j++) {
                    var m = candlist[j];
                    var orig = m;
                    var extra=""
                    if (typeof m !== 'string'){
                        if (m.length>1){
                            extra=m[1]
                        }
                        m=m[0];
                    }
                    var matchWord=m;
                    var matchWord = matchWord.toLowerCase();
                    // if (matchWord === curWord) continue;
                    if ((!curWord || matchWord.lastIndexOf(curWord, 0) == 0) && !Object.prototype.hasOwnProperty.call(seen, matchWord)) {
                        seen[matchWord] = true;

                        var mytag = tag;
                        if (mytag==="COLOR"){
                            mytag = "COLOR-"+m.toUpperCase();
                        }                    

                        list.push({text:m,extra:extra,tag:mytag,render:renderHint});
                    }
                }
            }
            

            //state.legend_aggregates
            //state.legend_synonyms
            //state.legend_properties
            //state.objects

            //remove words from the toexclude list

            
            if (toexclude.length>0){
                if (toexclude[toexclude.length-1]===curWord){
                    splits.pop();
                }
                for (var i=0;i<list.length;i++){
                    var lc = list[i].text.toLowerCase();
                    if (toexclude.indexOf(lc)>=0){
                        list.splice(i,1);
                        i--;
                    }
                }
            }
                    //if list is a single word and that matches what the current word is, don't show hint
            if (list.length===1 && list[0].text.toLowerCase()===curWord){
                list=[];
            }
            //if list contains the word that you've typed, put it to top of autocomplete list
            for (var i=1;i<list.length;i++){
                if (list[i].text.toLowerCase()===curWord){
                    var newhead=list[i];
                    list.splice(i,1);
                    list.unshift(newhead);
                    break;
                }
            }
            //if you're editing mid-word rather than at the end, no hints.
            if (tok.string.trim().length>curWord.length){
                list=[];
            }
            return {
                list: list,
                from: CodeMirror.Pos(cur.line, start),
                to: CodeMirror.Pos(cur.line, end)
            };
        });

    // https://statetackoverflow.com/questions/13744176/codemirror-autocomplete-after-any-keyup
    CodeMirror.ExcludedIntelliSenseTriggerKeys = {
        "9": "tab",
        "13": "enter",
        "16": "shift",
        "17": "ctrl",
        "18": "alt",
        "19": "pause",
        "20": "capslock",
        "27": "escape",
        "33": "pageup",
        "34": "pagedown",
        "35": "end",
        "36": "home",
        "37": "left",
        "38": "up",
        "39": "right",
        "40": "down",
        "45": "insert",
        "91": "left window key",
        "92": "right window key",
        "93": "select",
        "107": "add",
        "109": "subtract",
        "110": "decimal point",
        "111": "divide",
        "112": "f1",
        "113": "f2",
        "114": "f3",
        "115": "f4",
        "116": "f5",
        "117": "f6",
        "118": "f7",
        "119": "f8",
        "120": "f9",
        "121": "f10",
        "122": "f11",
        "123": "f12",
        "144": "numlock",
        "145": "scrolllock",
        "186": "semicolon",
        "187": "equalsign",
        "188": "comma",
        // "189": "dash",
        // "190": "period", on UK/US keyboard . is shift+>, which wants to trigger autocomplete
        "191": "slash",
        "192": "graveaccent",
        "220": "backslash",
        // "222": "quote" -used on german keyboard for > I think...
    }
});
