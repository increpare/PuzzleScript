var edn = window.jsedn;

addDumpTraceHook(function(title, curlevel, inputHistory) {
    consolePrint("Complete:");
    consolePrint(edn.encode(ednize_sequence(inputHistory)));
    consolePrint("Sliced:");
    consolePrint(edn.encode(ednize_sequences(slice_inputs(inputHistory))));
    // post to localhost:8000/title/curlevel/{complete,sliced}
    //     with data { inputs=EDN-INPUT-SEQ }
    //     and headers including Content-Type: application/edn
});

//SLICING:

//OK, puzzlescript's undo/restart/checkpoint semantics are a little crazy.
//Undo will undo a restart action, but undoing past a checkpoint won't
//change the current checkpoint until you end up causing another checkpoint to fire.
//The fancy code below handles that stuff, and it also collapses multiple "undo" actions
//into a single "undo:N" action.

function slice_inputs(inputs) {
    return filterInside(function(i) { return !Array.isArray(i); }, slice_inputs_(0,inputs,[],[],[]));
}

function slice_inputs_(i,inputs,trace,restartTrace,traces) {
    if(i >= inputs.length) { return [trace].concat(traces); }
    var first = inputs[i];
    //If we're about to restart, pop back to the restartTrace but store the current trace
    //in it in case we undo. The stored trace will get filtered out later.
    //Commit the current trace to the trace set.
    if(first == "restart") {
        return slice_inputs_(i+1,inputs,restartTrace.concat([["restarted",trace]]),restartTrace,[trace.concat(["restart"])].concat(traces));
    //If we're hitting a checkpoint, replace the restartTrace with the current trace.
    } else if(first == "checkpoint") {
        return slice_inputs_(i+1,inputs,trace,trace,traces);
    //If we're about to undo, count how many undos we'll do and then undo that many steps.
    //Commit the current trace to the trace set.
    } else if(first == "undo") {
        var undos = count_undos(i,inputs);
        var undone_trace = undo_n(trace,undos);
        return slice_inputs_(i+undos,inputs,undone_trace,restartTrace,[trace.concat(["undo:"+undos])].concat(traces));
    } else {
    //Otherwise, just add this input to the current trace.
        return slice_inputs_(i+1,inputs,trace.concat([first]),restartTrace,traces);
    }
}

//Count how many undo operations we will perform.
function count_undos(i,inputs) {
    if(i < inputs.length && inputs[i] == "undo") {
        return 1 + count_undos(i+1,inputs);
    }
    return 0;
}

//Undo the last N operations of trace. Returns a new trace.
//If undoing goes through a `["restarted", trace]` term, then
//continue undoing from the associated "prior trace".
function undo_n(trace,n) {
    if(n <= 0) { return trace; }
    if(trace.length == 0) { return trace; }
    var last = trace[trace.length-1];
    if(Array.isArray(last) && last[0] == "restarted") {
        return undo_n(last[1], n-1);
    }
    return undo_n(trace.slice(0,trace.length-1), n-1);
}

//Filters the leaf contents of a list-of-lists by function f.
function filterInside(f, ll) { 
    console.log("filtering"); console.log(JSON.stringify(ll));
    return ll.map(function (l) { return l.filter(f); }); 
}

//EDN EXPORT:

// Thanks to http://jsfiddle.net/jcward/7hyaC/3/
var lut = []; for (var i=0; i<256; i++) { lut[i] = (i<16?'0':'')+(i).toString(16); }
function makeUUID() {
    var d0 = Math.random()*0xffffffff|0;
    var d1 = Math.random()*0xffffffff|0;
    var d2 = Math.random()*0xffffffff|0;
    var d3 = Math.random()*0xffffffff|0;
    return lut[d0&0xff]+lut[d0>>8&0xff]+lut[d0>>16&0xff]+lut[d0>>24&0xff]+'-'+
      lut[d1&0xff]+lut[d1>>8&0xff]+'-'+lut[d1>>16&0x0f|0x40]+lut[d1>>24&0xff]+'-'+
      lut[d2&0x3f|0x80]+lut[d2>>8&0xff]+'-'+lut[d2>>16&0xff]+lut[d2>>24&0xff]+
      lut[d3&0xff]+lut[d3>>8&0xff]+lut[d3>>16&0xff]+lut[d3>>24&0xff];
}

function ednize_sequence(ih) {
    var uuid = makeUUID();
    return new edn.Vector(
        [new edn.Vector([edn.kw(":log_start"), uuid])].concat(
        ih.map(ednize_input)).concat(
        [edn.kw(":log_end")])
    );
}

function ednize_sequences(ihs) {
    return ihs.map(ednize_sequence);
}

function ednize_input(i) {
    var parts = i.toString().split(":");
    var player = edn.kw(":player");
    var detA, detB, vals;
    //[:i :player :game (:move) :place (3 [6 4])]
    //system actions:
    //randomEntIdx:Idx
    //randomDir:dir
    //randomRuleIdx:idx
    //checkpoint
    //win
    if(parts[0] == "randomEntIdx") {
        player = edn.kw(":random");
        detA = new edn.List([edn.kw(":resolve")]);
        detB = edn.kw(":select_entity");
        vals = new edn.List([parseInt(parts[1])]);
    } else if(parts[0] == "randomDir") {
        player = edn.kw(":random");        
        detA = new edn.List([edn.kw(":resolve")]);
        detB = edn.kw(":select_direction");
        var direction = dirMaskName[parseInt(parts[1])];
        vals = new edn.List([direction]);
    } else if(parts[0] == "randomRuleIdx") {
        player = edn.kw(":random");
        detA = new edn.List([edn.kw(":resolve")]);
        detB = edn.kw(":select_rule");
        vals = new edn.List([parseInt(parts[1])]);
    } else if(parts[0] == "checkpoint") {
        player = edn.kw(":system");
        detA = new edn.List([edn.kw(":resolve")]);
        detB = edn.kw(":checkpoint");
        vals = new edn.List([])
    } else if(parts[0] == "win") {
        player = edn.kw(":system");
        detA = new edn.List([edn.kw(":resolve")]);
        detB = edn.kw(":win");
        vals = new edn.List([]);
    //player actions:
    //quit
    //undo
    //restart
    //wait (autotick)
    //0 up
    //1 left
    //2 down
    //3 right 
    //4 act
    } else if(parts[0] == "quit") {
        player = edn.kw(":player");
        detA = new edn.List([edn.kw(":move")]);
        detB = edn.kw(":quit");
        vals = new edn.List([edn.kw(":quit")]);
    } else if(parts[0] == "undo") {
        player = edn.kw(":player");
        detA = new edn.List([edn.kw(":move")]);
        detB = edn.kw(":undo");
        vals = new edn.List([parts.length > 1 ? parseInt(parts[1]) : 1]);
    } else if(parts[0] == "restart") {
        player = edn.kw(":player");
        detA = new edn.List([edn.kw(":move")]);
        detB = edn.kw(":restart");
        vals = new edn.List([edn.kw(":restart")]);
    } else if(parts[0] == "wait") {
        player = edn.kw(":player");
        detA = new edn.List([edn.kw(":move")]);
        detB = edn.kw(":move");
        vals = new edn.List([edn.kw(":wait")]);
    } else if(parts[0] == "0") {
        player = edn.kw(":player");
        detA = new edn.List([edn.kw(":move")]);
        detB = edn.kw(":move");
        vals = new edn.List([edn.kw(":up")]);
    } else if(parts[0] == "1") {
        player = edn.kw(":player");
        detA = new edn.List([edn.kw(":move")]);
        detB = edn.kw(":move");
        vals = new edn.List([edn.kw(":left")]);
    } else if(parts[0] == "2") {
        player = edn.kw(":player");
        detA = new edn.List([edn.kw(":move")]);
        detB = edn.kw(":move");
        vals = new edn.List([edn.kw(":down")]);
    } else if(parts[0] == "3") {
        player = edn.kw(":player");
        detA = new edn.List([edn.kw(":move")]);
        detB = edn.kw(":move");
        vals = new edn.List([edn.kw(":right")]);
    } else if(parts[0] == "4") {
        player = edn.kw(":player");
        detA = new edn.List([edn.kw(":move")]);
        detB = edn.kw(":move");
        vals = new edn.List([edn.kw(":act")]);
    } else {
        throw "Unrecognized input "+parts;
    }
    return new edn.Vector([edn.kw(":i"), player, edn.kw(":game"), detA, detB, vals]);
}

