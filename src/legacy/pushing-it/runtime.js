/*
 * Pushing It, by Jack Lance. Preserved from gist
 * 2fe3172d2b9fe684977d184f1b6226d5, revision
 * 922c1a2fbf1fb887b3903f350c0af6cd0fbc8b9c.
 * See README.md for provenance and the small host compatibility changes.
 * This is intentionally a classic, non-strict script: the original game
 * replaces the PuzzleScript player hooks through global assignments.
 */
var pushingItEngineRedraw = redraw;
var pushingItEngineUpdate = update;

function syncPushingItControls() {
    if (!window.Mobile || !Mobile.syncPlayerControls) return;
    Mobile.syncPlayerControls();
    var undoButton = document.getElementById('undoButton');
    if (undoButton) {
        undoButton.disabled = textMode || typeof undostack === 'undefined' || undostack.length < 2;
    }
}

/*I know my code is awful/hacky, don't judge me.*/ canvas = document.getElementById('gameCanvas');
myctx = canvas.getContext('2d');
mysprites = {
    player: '.aaa.|ababa|aaaaa|aaaaa|.aaa.',
    wmyall: 'cccdd|ccddc|cddcc|ddccc|dcccd',
    box: 'ee.ee|eefee|.fff.|eefee|ee.ee',
    button: '.....|.ggg.|.ghg.|.ggg.|.....',
    pusherup: 'iiiii|ijjji|iijii|.iji.|.iii.',
    pusherdown: '.iii.|.iji.|iijii|ijjji|iiiii',
    pusherleft: 'iii..|ijiii|ijjji|ijiii|iii..',
    pusherright: '..iii|iiiji|ijjji|iiiji|..iii',
    Apusherup: 'iiiii|ijjji|..j..|..j..|..i..',
    Apusherdown: '..i..|..j..|..j..|ijjji|iiiii',
    Apusherleft: 'ii...|ij...|ijjji|ij...|ii...',
    Apusherright: '...ii|...ji|ijjji|...ji|...ii',
    target: 'kk.kk|k...k|.....|k...k|kk.kk',
    upborder: 'lllll|.....|.....|.....|.....',
    downborder: '.....|.....|.....|.....|lllll',
    leftborder: 'l....|l....|l....|l....|l....',
    rightborder: '....l|....l|....l|....l|....l'
};
cols = {
    a: 'medium1',
    b: 'activated',
    c: 'dark1',
    d: 'dark2',
    e: 'medium1',
    f: 'deactivated',
    g: 'light1',
    h: 'deactivated',
    i: 'deactivated',
    j: 'medium1',
    k: 'light1',
    l: 'verydark',
    '.': 'verylight1'
};
colscheme = {
    verylight1: '#C1FBC6',
    verylight2: '#CCCCCC',
    light1: '#668866',
    medium1: '#555555',
    deactivated: '#A20021',
    activated: '#FF0021',
    dark1: '#153A29',
    dark2: '#222222',
    verydark: '#01130A'
};
for (i of Object.keys(cols)){
    cols[i] = colscheme[cols[i]];
}
chars = {
    p: 'player',
    V: 'target',
    P: 'player',
    O: 'box',
    x: 'wmyall',
    D: 'pusherdown',
    U: 'pusherup',
    R: 'pusherright',
    z: 'wmyall',
    y: 'wmyall',
    o: 'box',
    l: 'pusherleft',
    r: 'pusherright',
    d: 'pusherdown',
    u: 'pusherup',
    b: 'button',
    t: 'target',
    T: 'target',
    X: 'wmyall',
    '#': 'wmyall'
};
sortOrder = {
    player: 1,
    wmyall: 1,
    box: 1,
    pusherup: 1.1,
    pusherleft: 1,
    pusherdown: 1.1,
    pusherright: 1,
    button: 0,
    target: 0,
    leftborder: 2,
    rightborder: 2,
    upborder: 2,
    downborder: 2
};
things = [];
grid = [
    'xxxxxxxxxxx|xpx...d...x|x.x..xbb.xx|x.xx.x.x..x|x..x.x.xxox|xxbx.b.xxbx|x..xr.xxx.x|x.xxxbxxx.x|rb....xxt.l|xxxxxxxxxxx',
    'xxxxxxxxxxx|xx...x..xxx|x....x.bxxx|xlxx.....xx|xpx.b.x..xx|x.xx..x..xx|xbxx......x|xxxx.x.b..x|xxxx....xxx|xxxxb.ootxx|xxxxxxxxxxx',
    'xxxxxxxxxxx|xxx.....xxx|xxx..u..xxx|x..x...x..x|x...x.x...x|x.t..p..b.x|x...x.x...x|x..x...x..x|xxx..o..xxx|xxx.....xxx|xxxxxxxxxxx',
    'xxxxxxxxxxxxx|xxxxx.xxxxxxx|xxx...o.d.xxx|xxx...b...xxx|xxxxx...xxxxx|xxx..x..xxxxx|xx.......xxxx|x...uxx.....x|xx..x..x..b.x|xtr.x.b.x...x|xxx....pxxxxx|xxxxxxxxxxxxx',
    'xxxxxxxxxxxxxx|xxxx.x......xx|xxxtb.......xx|xxx.ub....x.xx|xxx..x..o.d.xx|xxxxxxx.x.xxxx|xxxxxxxxx.xxxx|x.rrrr...bp.xx|xxxxxxxxxx...x|x......b.....x|x.xx.....or..x|xlll...xxxx.xx|x.x..xxx....lx|x....xxx....xx|xxxxxxxxxxxxxx',
    '...xxxxxxxxx..|.xxx.......x..|.x....xxx..x..|.x.p..xxx..xxx|.x..o..xxrb..x|xxb....xx.x..x|x..x..Txx..l.x|xd..xxxx.....x|x.....xxxx..xx|xox.x.dx.xxxx.|xuxo...x......|x.x...bx......|xxxx..ux......|...xxxxx......',
    '.....xxxxxxx.....|..tor......x.....|....x......x.....|.xxxx...x.xxd....|xx..x.....x......|x..ox...xxx.ur.l.|x...x...xpx.or.l.|xx.o.....b.......|.xx.........xxxxx|..x.............x|..xx.xx.........x|..x.blx..xxxxxxdx|..xx.xxx........x|xxxxo..x..xx...bx|x......xxxxrbo..x|x......x..xx..xxx|x...x..x...x..x..|xxxxxxxx...xxxx..|',
    '..x............|x..r.l..xxxxxxx|.x.xxxxxx....xx|.x........xrb.x|.x.....xxxxx.xx|.rb....xxxx..x.|.xo....xxx...x|.x.xxxxxx...xx|.......x.o.xx.|..t....x.o.x..|xrrrr....blx..|xxxxxxxxx.xx|xrb.llllx.x|x....obxx.x|xxxx..u...x|...x..pxxxx.|...xxxxx',
    '.....xxx.|xxxxxxtx|x.....lx|x..r.o.x|x.b....xx|x.......x|xxxxxxx.x|xb....l.x|x..r.p.xx|x......x|x......x|xxxxxxxx',
    'xxx........|xt.xxxxxxx.|x.p..x....x|xx.....b..x|.xxxob....x|...x....b.x|..x.b.b...x|..x.......x|..x.....b.x|..xxxx.bxxx|.....x..x..|....x...dx.|....xr..lx.|....xu...x.|....xx.bxx.|.....xxxx..',
    '.......xxxx....|.......xD.xx|.......x.b.x|......xx.U.x|..xxxxx...xx|xxx....o..x|xt......xxx|xxx.....xxxxxx|..x..xxx..p..xx|..x...........x|..x.......xx..x|..xxxxxxxxxx..x|.xxxxxx...xx..x|xxd........x..x|x.b.....x..x..x|x....xx....x..x|x....xx.......x|xxx..xx......xx|..xxxxxxxxxxxx.',
    'xxxxxxxxxxx|xpx...d...x|x.x..xbb.xx|x.xx.x.x.tx|x..x.x.xxox|xxbx.b.xxbx|x..xr.xxx.x|x.xxxbxxx.x|rb....xx..l|xxxxxxxxxxx'
];
dirs = {
    r: [
        1,
        0
    ],
    d: [
        0,
        1
    ],
    u: [
        0,
        -1
    ],
    l: [
        -1,
        0
    ],
    R: [
        1,
        0
    ],
    U: [
        0,
        -1
    ],
    D: [
        0,
        1
    ]
};
cellsize = 10;
cellsize2 = 100;
myall = {};
clone = (x)=>{
    return JSON.parse(JSON.stringify(x));
};
getstate = ()=>{
    return {
        things: clone(things),
        activated: activated,
        psprite: mysprites['player']
    };
};
fromState = (s)=>{
    things = s.things;
    activated = s.activated;
    mysprites['player'] = s.psprite;
    myall = {};
    player = {};
    for (thing of things){
        if (thing.name == 'player') player = thing;
        myall[thing.name] ||= [];
        myall[thing.name].push(thing);
        if (thing.name.startsWith('pusher')) {
            myall['pusher'] ||= [];
            myall['pusher'].push(thing);
        }
    }
    activated = s.activated;
    cols['h'] = activated ? colscheme['activated'] : colscheme['deactivated'];
    cols['i'] = activated ? colscheme['activated'] : colscheme['deactivated'];
    againing = false;
};
undo = ()=>{
    if (undostack.length < 2) return;
    undostack.pop();
    fromState(clone(undostack[undostack.length - 1]));
    draw();
};
loadmylevel = ()=>{
    curlevel = Number(mylevel);
    cellsize = Math.floor(Math.min(canvas.height / (grid[mylevel].split('|').length * 5), canvas.width / (grid[mylevel].split('|')[0].length * 5))) * 5;
    vpad = Math.floor((canvas.width - cellsize * grid[mylevel].split('|')[0].length) / 2);
    hpad = Math.floor((canvas.height - cellsize * grid[mylevel].split('|').length) / 2);
    things = [];
    activated = false;
    grid[mylevel].split('|').map((s, i1)=>{
        s.split('').map((c, j)=>{
            if (chars[c] != undefined) {
                var thing = {
                    name: chars[c],
                    x: j * cellsize2,
                    y: i1 * cellsize2,
                    dx: cellsize2,
                    dy: cellsize2,
                    angle: 0
                };
                things.push(thing);
                if (chars[c].startsWith('pusher')) {
                    [thing.dirx, thing.diry] = dirs[c];
                    thing.realx = thing.x;
                    thing.realy = thing.y;
                }
                if (chars[c] == 'wmyall') {
                    things.push({
                        name: 'upborder',
                        x: j * cellsize2,
                        y: i1 * cellsize2,
                        dx: cellsize2,
                        dy: cellsize2,
                        angle: 0
                    });
                    things.push({
                        name: 'downborder',
                        x: j * cellsize2,
                        y: i1 * cellsize2,
                        dx: cellsize2,
                        dy: cellsize2,
                        angle: 0
                    });
                    things.push({
                        name: 'leftborder',
                        x: j * cellsize2,
                        y: i1 * cellsize2,
                        dx: cellsize2,
                        dy: cellsize2,
                        angle: 0
                    });
                    things.push({
                        name: 'rightborder',
                        x: j * cellsize2,
                        y: i1 * cellsize2,
                        dx: cellsize2,
                        dy: cellsize2,
                        angle: 0
                    });
                }
                if (c == 'T') {
                    thing.x -= cellsize2 / 2;
                    thing.y -= cellsize2 / 2;
                }
                if (c == 'V') {
                    thing.x -= cellsize2 / 2;
                    thing.y -= cellsize2 / 2;
                    thing.angle = Math.PI / 8;
                }
                if (c == 'X') {
                    thing.y -= cellsize2 / 2;
                }
                if (c == 'O') {
                    thing.y += cellsize2 / 20;
                }
                if (c == 'R') {
                    thing.y += 3 * cellsize2 / 5;
                    thing.realy += 3 * cellsize2 / 5;
                }
                if (c == 'U') {
                    thing.x += 3.8 * cellsize2 / 5;
                    thing.realx += 3.8 * cellsize2 / 5;
                }
                if (c == 'D') {
                    thing.x += 1.2 * cellsize2 / 5;
                    thing.realx += 1.2 * cellsize2 / 5;
                }
                if (c == 'y') thing.x -= cellsize2 / 5;
                if (c == 'z') thing.x += cellsize2 / 5;
                if (c == '#') {
                    var thing21 = {
                        name: chars[c],
                        x: j * cellsize2,
                        y: i1 * cellsize2 - cellsize2 / 2,
                        dx: cellsize2,
                        dy: cellsize2
                    };
                    things.push(thing21);
                }
                if (c == 'P') {}
            }
        });
    });
    if (mylevel == 3) sortOrder['pusherright'] = 1.2;
    else sortOrder['pusherright'] = 1;
    things.sort((a, b)=>{
        return sortOrder[a.name] - sortOrder[b.name];
    });
    for(var i1 = 0; i1 < things.length; i1++){
        for(var j = 0; j < things.length; j++){
            if (things[i1].name == 'upborder' && things[j].name == 'downborder' && things[i1].x == things[j].x && things[i1].y - cellsize2 == things[j].y) {
                things.splice(Math.max(i1, j), 1);
                things.splice(Math.min(i1, j), 1);
                i1 -= 2;
                j -= 2;
            }
            if (things[i1].name == 'leftborder' && things[j].name == 'rightborder' && things[i1].x - cellsize2 == things[j].x && things[i1].y == things[j].y) {
                things.splice(Math.max(i1, j), 1);
                things.splice(Math.min(i1, j), 1);
                i1 -= 2;
                j -= 2;
            }
        }
    }
    fromState(getstate());
    undostack.push(getstate());
};
controls = [
    {
        key: 'ArrowUp',
        f: ()=>{
            moveDir(dirs['u']);
        }
    },
    {
        key: 'ArrowLeft',
        f: ()=>{
            moveDir(dirs['l']);
        }
    },
    {
        key: 'ArrowDown',
        f: ()=>{
            moveDir(dirs['d']);
        }
    },
    {
        key: 'ArrowRight',
        f: ()=>{
            moveDir(dirs['r']);
        }
    }
];
moveDir = (d)=>{
    moveTimes(cellsize2 * d[0], cellsize2 * d[1], player);
};
onTarget = (thing)=>{
    for (other of myall['target']){
        if (other.x - cellsize2 / 2 <= roundish(thing.x) && other.x + cellsize2 / 2 >= roundish(thing.x) && other.y - cellsize2 / 2 <= roundish(thing.y) && other.y + cellsize2 / 2 >= roundish(thing.y)) {
            return true;
        }
    }
    return false;
};
tmpcanvas = document.createElement('canvas');
tmpmyctx = tmpcanvas.getContext('2d');
drawSquare = (x, y)=>{
    tmpmyctx.fillRect(x, y, cellsize / 5, cellsize / 5);
};
drawSquareold = (x, y)=>{
    myctx.fillRect(Math.round(vpad + x), Math.round(hpad + y), Math.round(vpad + x + cellsize / 5) - Math.round(vpad + x), Math.round(hpad + y + cellsize / 5) - Math.round(hpad + y));
};
wincounter = 0;
drawthing = (thing)=>{
    if (thing.angle == 0) {
        drawthingold(thing);
        return;
    }
    if (!thing.name.startsWith('pusher')) {
        tmpmyctx.translate(cellsize / 2, cellsize / 2);
        tmpmyctx.rotate(-thing.angle);
        tmpmyctx.translate(-cellsize / 2, -cellsize / 2);
        tmpmyctx.drawImage(canvas, Math.floor(vpad + thing.x + cellsize / 2) - Math.floor(cellsize / 2) - cellsize, Math.floor(hpad + thing.y + cellsize / 2) - Math.floor(cellsize / 2) - cellsize, cellsize * 3, cellsize * 3, -cellsize, -cellsize, cellsize * 3, cellsize * 3);
        tmpmyctx.translate(cellsize / 2, cellsize / 2);
        tmpmyctx.rotate(thing.angle);
        tmpmyctx.translate(-cellsize / 2, -cellsize / 2);
        for(var i1 = 0; i1 < 5; i1++){
            for(var j = 0; j < 5; j++){
                tmpmyctx.fillStyle = cols[mysprites[thing.name][i1 + 6 * j]];
                if (mysprites[thing.name][i1 + 6 * j] == '.') continue;
                if (mysprites[thing.name][i1 + 6 * j] == 'f') {
                    xamt = cellsize / 5;
                    yamt = cellsize / 5;
                    if (onTarget({
                        x: thing.x / ratio + ((i1 - 2) * Math.cos(thing.angle) - (j - 2) * Math.sin(thing.angle)) * cellsize2 / 5,
                        y: thing.y / ratio + ((i1 - 2) * Math.sin(thing.angle) + (j - 2) * Math.cos(thing.angle)) * cellsize2 / 5
                    })) {
                        tmpmyctx.fillStyle = colscheme['activated'];
                        wincounter++;
                    }
                }
                drawSquare(cellsize / 5 * i1, cellsize / 5 * j);
            }
        }
        myctx.translate(Math.floor(vpad + thing.x + cellsize / 2), Math.floor(hpad + thing.y + cellsize / 2));
        myctx.rotate(thing.angle);
        myctx.drawImage(tmpcanvas, 0, 0, cellsize, cellsize, -Math.floor(cellsize / 2), -Math.floor(cellsize / 2), cellsize, cellsize);
        myctx.rotate(-thing.angle);
        myctx.translate(-Math.floor(vpad + thing.x + cellsize / 2), -Math.floor(hpad + thing.y + cellsize / 2));
    } else {
        tmpmyctx.translate(cellsize / 2, cellsize / 2);
        tmpmyctx.rotate(-thing.angle);
        tmpmyctx.translate(-cellsize / 2, -cellsize / 2);
        tmpmyctx.drawImage(canvas, Math.floor(vpad + thing.realx + cellsize / 2) - Math.floor(cellsize / 2) - 2 * (thing.realx - thing.x) - cellsize, Math.floor(hpad + thing.realy + cellsize / 2) - Math.floor(cellsize / 2) - 2 * (thing.realy - thing.y) - cellsize, cellsize * 3, cellsize * 3, -cellsize, -cellsize, cellsize * 3, cellsize * 3);
        tmpmyctx.translate(cellsize / 2, cellsize / 2);
        tmpmyctx.rotate(thing.angle);
        tmpmyctx.translate(-cellsize / 2, -cellsize / 2);
        tmpmyctx.fillStyle = activated ? colscheme['activated'] : colscheme['deactivated'];
        for(var i1 = 0; i1 < 5; i1++){
            for(var j = 0; j < 5; j++){
                if (thing.dirx == 0 && i1 == 2 || thing.diry == 0 && j == 2 || thing.dirx == 1 && i1 > 2 || thing.dirx == -1 && i1 < 2 || thing.diry == 1 && j > 2 || thing.diry == -1 && j < 2) drawSquare(cellsize / 5 * i1, cellsize / 5 * j);
            }
        }
        myctx.translate(Math.floor(vpad + thing.realx + cellsize / 2), Math.floor(hpad + thing.realy + cellsize / 2));
        myctx.rotate(thing.angle);
        myctx.drawImage(tmpcanvas, 0, 0, cellsize, cellsize, -Math.floor(cellsize / 2) - 2 * (thing.realx - thing.x), -Math.floor(cellsize / 2) - 2 * (thing.realy - thing.y), cellsize, cellsize);
        myctx.rotate(-thing.angle);
        myctx.translate(-Math.floor(vpad + thing.realx + cellsize / 2), -Math.floor(hpad + thing.realy + cellsize / 2));
        for(var i1 = 0; i1 < 5; i1++){
            for(var j = 0; j < 5; j++){
                tmpmyctx.fillStyle = cols[mysprites[thing.name][i1 + 6 * j]];
                drawSquare(cellsize / 5 * i1, cellsize / 5 * j);
            }
        }
        myctx.translate(Math.floor(vpad + thing.realx + cellsize / 2), Math.floor(hpad + thing.realy + cellsize / 2));
        myctx.rotate(thing.angle);
        myctx.drawImage(tmpcanvas, 0, 0, cellsize, cellsize, -Math.floor(cellsize / 2), -Math.floor(cellsize / 2), cellsize, cellsize);
        myctx.rotate(-thing.angle);
        myctx.translate(-Math.floor(vpad + thing.realx + cellsize / 2), -Math.floor(hpad + thing.realy + cellsize / 2));
    }
};
drawthingold = (thing)=>{
    if (!thing.name.startsWith('pusher')) {
        for(var i1 = 0; i1 < 5; i1++){
            for(var j = 0; j < 5; j++){
                if (mysprites[thing.name][i1 + 6 * j] == '.') continue;
                myctx.fillStyle = cols[mysprites[thing.name][i1 + 6 * j]];
                if (mysprites[thing.name][i1 + 6 * j] == 'f') {
                    if (onTarget({
                        x: thing.x / ratio - cellsize2 / 2 + cellsize2 / 10 + cellsize2 / 5 * i1,
                        y: thing.y / ratio - cellsize2 / 2 + cellsize2 / 10 + cellsize2 / 5 * j
                    })) {
                        myctx.fillStyle = colscheme['activated'];
                        wincounter++;
                    }
                }
                drawSquareold(thing.x + cellsize / 5 * i1, thing.y + cellsize / 5 * j);
            }
        }
    } else {
        myctx.fillStyle = activated ? colscheme['activated'] : colscheme['deactivated'];
        for(var i1 = 0; i1 < 5; i1++){
            for(var j = 0; j < 5; j++){
                if (thing.dirx == 0 && i1 == 2 || thing.diry == 0 && j == 2 || thing.dirx == 1 && i1 > 2 || thing.dirx == -1 && i1 < 2 || thing.diry == 1 && j > 2 || thing.diry == -1 && j < 2) drawSquareold(thing.realx + cellsize / 5 * i1 - 2 * (thing.realx - thing.x), thing.realy + cellsize / 5 * j - 2 * (thing.realy - thing.y));
            }
        }
        for(var i1 = 0; i1 < 5; i1++){
            for(var j = 0; j < 5; j++){
                if (mysprites[thing.name][i1 + 6 * j] == '.') continue;
                myctx.fillStyle = cols[mysprites[thing.name][i1 + 6 * j]];
                drawSquareold(cellsize / 5 * i1 + thing.realx, cellsize / 5 * j + thing.realy);
            }
        }
    }
};
draw = ()=>{
    syncPushingItControls();
    wincounter = 0;
    cellsize = Math.floor(Math.min(canvas.height / (grid[mylevel].split('|').length * 5), canvas.width / (grid[mylevel].split('|')[0].length * 5))) * 5;
    vpad = Math.floor((canvas.width - cellsize * grid[mylevel].split('|')[0].length) / 2);
    hpad = Math.floor((canvas.height - cellsize * grid[mylevel].split('|').length) / 2);
    myctx.fillStyle = '#C1FBC6';
    myctx.fillRect(0, 0, canvas.width, canvas.height);
    for(var i1 = 0; i1 < grid[mylevel].split('|')[0].length; i1++){
        for(var j = 0; j < grid[mylevel].split('|').length; j++){
            myctx.fillStyle = (i1 + j) % 2 ? colscheme['verylight1'] : colscheme['verylight2'];
            myctx.fillRect(i1 * cellsize + vpad, j * cellsize + hpad, cellsize, cellsize);
        }
    }
    ratio = cellsize / cellsize2;
    for (thing of things){
        thing2 = clone(thing);
        thing2.x *= ratio;
        thing2.y *= ratio;
        thing2.dx *= ratio;
        thing2.dy *= ratio;
        thing2.realx *= ratio;
        thing2.realy *= ratio;
        drawthing(thing2);
    }
    winCheck();
};
clockwise = (p, q, r)=>{
    z = (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x);
    if (z > 0) return 1;
    if (z < 0) return -1;
    return 0;
};
getCorners = (a)=>{
    tox = a.dx / 2 * Math.cos(a.angle) - a.dy / 2 * Math.sin(a.angle);
    toy = a.dy / 2 * Math.cos(a.angle) + a.dx / 2 * Math.sin(a.angle);
    return [
        {
            x: a.x - tox,
            y: a.y - toy
        },
        {
            x: a.x - toy,
            y: a.y + tox
        },
        {
            x: a.x + tox,
            y: a.y + toy
        },
        {
            x: a.x + toy,
            y: a.y - tox
        }
    ];
};
rectsIntersect = (a, b)=>{
    if (a.angle == 0 && b.angle == 0) {
        return roundish(Math.abs(a.x - b.x)) < roundish((a.dx + b.dx) / 2) && roundish(Math.abs(a.y - b.y)) < roundish((a.dy + b.dy) / 2);
    }
    if ((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) < a.dx * a.dx) return true;
    if (Math.abs(a.x - b.x) > a.dx + b.dx || Math.abs(a.y - b.y) > a.dy + b.dy) return false;
    acorns = getCorners(a);
    bcorns = getCorners(b);
    for (ac of acorns)if (clockwise(ac, bcorns[0], bcorns[1]) * clockwise(ac, bcorns[2], bcorns[3]) == 1 && clockwise(ac, bcorns[1], bcorns[2]) * clockwise(ac, bcorns[3], bcorns[0]) == 1) return true;
    for (bc of bcorns)if (clockwise(bc, acorns[0], acorns[1]) * clockwise(bc, acorns[2], acorns[3]) == 1 && clockwise(bc, acorns[1], acorns[2]) * clockwise(bc, acorns[3], acorns[0]) == 1) return true;
    return false;
};
wouldCollide = (obj, thing)=>{
    if (obj.name.startsWith('pusher') && (obj.dx > cellsize2 || obj.dy > cellsize2)) {
        return wouldCollide({
            x: obj.realx,
            y: obj.realy,
            dx: cellsize2,
            dy: cellsize2,
            name: obj.name,
            angle: obj.angle
        }, thing) || wouldCollide({
            x: obj.x * 2 - obj.realx,
            y: obj.y * 2 - obj.realy,
            dx: cellsize2,
            dy: cellsize2,
            name: 'A' + obj.name,
            angle: obj.angle
        }, thing);
    }
    if (thing.name.startsWith('pusher') && (thing.dx > cellsize2 || thing.dy > cellsize2)) {
        return wouldCollide(obj, {
            x: thing.realx,
            y: thing.realy,
            dx: cellsize2,
            dy: cellsize2,
            name: thing.name,
            angle: thing.angle
        }) || wouldCollide(obj, {
            x: thing.x * 2 - thing.realx,
            y: thing.y * 2 - thing.realy,
            dx: cellsize2,
            dy: cellsize2,
            name: 'A' + thing.name,
            angle: thing.angle
        });
    }
    if (!rectsIntersect(obj, thing)) {
        return false;
    }
    if (obj.angle == 0 && thing.angle == 0 && (roundish(Math.abs(thing.x - obj.x)) <= cellsize2 / 5 || roundish(Math.abs(thing.y - obj.y)) <= cellsize2 / 5) && !obj.name.startsWith('A') && !thing.name.startsWith('A')) return true;
    var tox1 = obj.dx / 2 * Math.cos(obj.angle) - obj.dy / 2 * Math.sin(obj.angle);
    var toy1 = obj.dy / 2 * Math.cos(obj.angle) + obj.dx / 2 * Math.sin(obj.angle);
    var tox2 = thing.dx / 2 * Math.cos(thing.angle) - thing.dy / 2 * Math.sin(thing.angle);
    var toy2 = thing.dy / 2 * Math.cos(thing.angle) + thing.dx / 2 * Math.sin(thing.angle);
    for(var i1 = 0; i1 < 5; i1++){
        for(var j = 0; j < 5; j++){
            if (mysprites[obj.name][i1 + 6 * j] == '.') continue;
            if (!rectsIntersect({
                x: obj.x - tox1 + (tox1 + toy1) / 5 * (i1 + 0.5) + (tox1 - toy1) / 5 * (j + 0.5),
                y: obj.y - toy1 + (toy1 - tox1) / 5 * (i1 + 0.5) + (toy1 + tox1) / 5 * (j + 0.5),
                dx: cellsize2 / 5,
                dy: cellsize2 / 5,
                angle: obj.angle
            }, thing)) continue;
            for(var i2 = 0; i2 < 5; i2++){
                for(var j2 = 0; j2 < 5; j2++){
                    if (mysprites[thing.name][i2 + 6 * j2] == '.') continue;
                    if (rectsIntersect({
                        x: obj.x - tox1 + (tox1 + toy1) / 5 * (i1 + 0.5) + (tox1 - toy1) / 5 * (j + 0.5),
                        y: obj.y - toy1 + (toy1 - tox1) / 5 * (i1 + 0.5) + (toy1 + tox1) / 5 * (j + 0.5),
                        dx: cellsize2 / 5,
                        dy: cellsize2 / 5,
                        angle: obj.angle
                    }, {
                        x: thing.x - tox2 + (tox2 + toy2) / 5 * (i2 + 0.5) + (tox2 - toy2) / 5 * (j2 + 0.5),
                        y: thing.y - toy2 + (toy2 - tox2) / 5 * (i2 + 0.5) + (toy2 + tox2) / 5 * (j2 + 0.5),
                        dx: cellsize2 / 5,
                        dy: cellsize2 / 5,
                        angle: thing.angle
                    })) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
};
roundish = (a)=>{
    return Math.floor((a + 0.0000001) * 10000) / 10000;
};
powerpower = 10;
pushcontents = [];
move = (dx, dy, obj, power = 10, ang = 0, expandx = 0, expandy = 0, shrinkx = 0, shrinky = 0)=>{
    if (pushcontents.length == 0) sh = shrinkx == 0 && shrinky == 0;
    pushcontents.push(obj);
    powerpower = power;
    var obj2 = clone(obj);
    pushx = dx;
    pushy = dy;
    if (expandx != 0 || expandy != 0) {
        obj2.dx += Math.abs(expandx);
        obj2.dy += Math.abs(expandy);
        obj2.realx += dx;
        obj2.realy += dy;
        obj2.x += dx + expandx / 2;
        obj2.y += dy + expandy / 2;
        pushx = 2 * dx + expandx;
        pushy = 2 * dy + expandy;
    } else if (shrinkx != 0 || shrinky != 0) {
        obj2.dx += -Math.abs(shrinkx);
        obj2.dy += -Math.abs(shrinky);
        obj2.realx += dx;
        obj2.realy += dy;
        obj2.x += dx - shrinkx / 2;
        obj2.y += dy - shrinky / 2;
        pushx = 2 * dx - shrinkx;
        pushy = 2 * dy - shrinky;
    } else if (ang == 0) {
        obj2.x += dx;
        obj2.y += dy;
        obj2.realx += dx;
        obj2.realy += dy;
    } else {
        if (obj.name.startsWith('pusher')) {
            pushcontents.pop();
            return false;
        }
        obj2.x += dx;
        obj2.y += dy;
        obj2.realx += dx;
        obj2.realy += dy;
        obj2.angle += ang;
    }
    obj2.x = roundish(obj2.x);
    obj2.y = roundish(obj2.y);
    obj2.dx = roundish(obj2.dx);
    obj2.dy = roundish(obj2.dy);
    obj2.realx = roundish(obj2.realx);
    obj2.realy = roundish(obj2.realy);
    for (var thing of things){
        if (thing.name == 'wmyall' && Math.abs(thing.x - obj2.x) < cellsize2 * 2 && Math.abs(thing.y - obj2.y) < cellsize2 * 2) {
            if (wouldCollide(obj2, thing)) {
                pushcontents.pop();
                return false;
            }
        }
        if ((thing.name == 'box' || thing.name.startsWith('pusher') || thing.name == 'player') && Math.abs(thing.x - obj2.x) < 3 * cellsize2 && Math.abs(thing.y - obj2.y) < 3 * cellsize2) {
            if (thing == obj) continue;
            if (!wouldCollide(obj2, thing)) {
                continue;
            }
            if (pushcontents.includes(thing) && ang == 0) {
                if (sh) continue;
                else {
                    pushcontents.pop();
                    return false;
                }
            }
            if (power == 0) {
                pushcontents.pop();
                return false;
            }
            if (!move(pushx, pushy, thing, power - 1)) {
                if (thing.x == obj.x || thing.y == obj.y) {
                    pushcontents.pop();
                    return false;
                }
                if (thing.name.startsWith('pusher')) {
                    pushcontents.pop();
                    return false;
                }
                var ok = false;
                if (mylevel >= 9) {
                    obj3 = clone(obj);
                    obj.x = obj2.x;
                    obj.y = obj2.y;
                    obj.dx = obj2.dx;
                    obj.dy = obj2.dy;
                    obj.realx = obj2.realx;
                    obj.realy = obj2.realy;
                    obj.angle = obj2.angle;
                    for(var i1 = 1; i1 <= 15; i1++){
                        if (move(pushx, pushy, thing, 0, i1 * Math.PI / 600)) {
                            ok = true;
                            break;
                        }
                    }
                    for(var i1 = -1; i1 >= -15; i1--){
                        if (move(pushx, pushy, thing, 0, i1 * Math.PI / 600)) {
                            ok = true;
                            break;
                        }
                    }
                }
                if (!ok) {
                    if (mylevel >= 9) {
                        obj.x = obj3.x;
                        obj.y = obj3.y;
                        obj.dx = obj3.dx;
                        obj.dy = obj3.dy;
                        obj.realx = obj3.realx;
                        obj.realy = obj3.realy;
                        obj.angle = obj3.angle;
                    }
                    pushcontents.pop();
                    return false;
                }
            }
        }
    }
    obj.x = obj2.x;
    obj.y = obj2.y;
    obj.dx = obj2.dx;
    obj.dy = obj2.dy;
    obj.realx = obj2.realx;
    obj.realy = obj2.realy;
    obj.angle = obj2.angle;
    if (Math.round(obj.angle / (Math.PI / 2)) == obj.angle / (Math.PI / 2)) obj.angle = 0;
    obj.done = false;
    pushcontents.pop();
    return true;
};
activated = false;
activatePushers = (state1)=>{
    activated = state1;
    cols['h'] = activated ? colscheme['activated'] : colscheme['deactivated'];
    cols['i'] = activated ? colscheme['activated'] : colscheme['deactivated'];
    if (state1) return expandTimes();
    else return shrinkTimes();
};
anim = 200;
buttonCheck = ()=>{
    anyDid = false;
    for (button of myall['button']){
        for (thing of things){
            if (!(thing.name == 'box' || thing.name == 'player' || thing.name.startsWith('pusher'))) continue;
            if (thing.name.startsWith('pusher')) {
                if (Math.abs(button.x - thing.realx) <= cellsize2 / 2 && Math.abs(button.y - thing.realy) <= cellsize2 / 2) {
                    anyDid = true;
                }
            } else if (Math.abs(button.x - thing.x) <= cellsize2 / 2 && Math.abs(button.y - thing.y) <= cellsize2 / 2) {
                anyDid = true;
            }
        }
    }
    againing = true;
    if (activatePushers(anyDid)) {
        setTimeout(buttonCheck, anim);
    } else {
        againing = false;
        undostack.push(getstate());
        syncPushingItControls();
        winCheck();
        for (thing of things){
            if (Math.abs(Math.round(thing.angle / (Math.PI / 2)) * (Math.PI / 2) - thing.angle) < Math.PI / 199) {
                if (Math.round(thing.angle / (Math.PI / 2)) * (Math.PI / 2) == thing.angle && Math.round(thing.x / (cellsize2 / 20)) * (cellsize2 / 20) == thing.x && Math.round(thing.y / (cellsize2 / 20)) * (cellsize2 / 20) == thing.y) continue;
                move(Math.round(thing.x / (cellsize2 / 20)) * (cellsize2 / 20) - thing.x, Math.round(thing.y / (cellsize2 / 20)) * (cellsize2 / 20) - thing.y, thing, 0, Math.round(thing.angle / (Math.PI / 2)) * (Math.PI / 2) - thing.angle);
            }
        }
    }
};
mywinning = false;
winCheck = ()=>{
    if (mywinning || wincounter < 5) return;
    mywinning = true;
    win();
};
win = ()=>{
    againing = true;
    setTimeout(()=>{
        againing = false;
        mylevel++;
        playSound(90748503);
        if (!!window.localStorage) {
            localStorage[document.URL] = mylevel;
        }
        mywinning = false;
        undostack = [];
        if (mylevel == 12) {
            curlevel = 0;
            curlevelTarget = null;
            titleScreen = true;
            titleSelected = false;
            curlevel = 0;
            localStorage.removeItem(document.URL);
            textMode = true;
            quittingMessageScreen = false;
            goToTitleScreen();
            titleImage = [
                '                                  ',
                '            Pushing It            ',
                '                                  ',
                '                     by Jack Lance',
                '                                  ',
                '                                  ',
                '         # start game #           ',
                '                                  ',
                '                                  ',
                ' arrow keys to move               ',
                ' X to action                      ',
                ' Z to undo, R to restart          ',
                '                                  '
            ];
            redraw();
            canvasResize();
        } else {
            loadmylevel();
            draw();
        }
    }, 400);
};
expand = (dx, dy, obj)=>{
    if (Math.abs(dx) + obj.dx > cellsize2 * 2 || Math.abs(dy) + obj.dy > cellsize2 * 2) {
        obj.done = true;
        return false;
    }
    if (move(0, 0, obj, 10, 0, dx, dy)) {
        return true;
    } else if (move(-dx, -dy, obj, 10, 0, dx, dy)) {
        return true;
    } else {
        obj.done = true;
        return false;
    }
};
shrink = (dx, dy, obj)=>{
    if (obj.dx == cellsize2 && obj.dy == cellsize2) return false;
    if (dx != 0) dx = dx / Math.abs(dx) * Math.max(0, Math.min(Math.abs(dx), obj.dx - cellsize2));
    if (dy != 0) dy = dy / Math.abs(dy) * Math.max(0, Math.min(Math.abs(dy), obj.dy - cellsize2));
    if (move(0, 0, obj, 10, 0, 0, 0, dx, dy)) {
        return true;
    } else if (move(dx, dy, obj, 10, 0, 0, 0, dx, dy)) {
        return true;
    } else {
        obj.done = true;
        return false;
    }
};
expandTimes = (t = 100)=>{
    for (pusher of myall['pusher']){
        pusher.done = false;
    }
    var returnval = false;
    for(var j = 0; j < t / 5; j++){
        if (mylevel != 8 || j % 2 == 0) {
            for (pusher of myall['pusher']){
                var b = expand(cellsize2 * pusher.dirx / t, cellsize2 * pusher.diry / t, pusher);
                returnval = returnval || b;
            }
        } else {
            for (pusher of myall['pusher'].reverse()){
                var b = expand(cellsize2 * pusher.dirx / t, cellsize2 * pusher.diry / t, pusher);
                returnval = returnval || b;
            }
        }
    }
    if (!returnval) return false;
    draw();
    playSound(71682106);
    for(var i1 = 1; i1 < 5; i1++){
        setTimeout(()=>{
            playSound(71682106);
        }, i1 * anim / 5);
        setTimeout(()=>{
            for(var j = 0; j < t / 5; j++){
                for (pusher of myall['pusher']){
                    if (pusher.done) continue;
                    expand(cellsize2 * pusher.dirx / t, cellsize2 * pusher.diry / t, pusher);
                }
            }
            draw();
        }, i1 * anim / 5);
    }
    setTimeout(draw, anim);
    return true;
};
shrinkTimes = (t = 100)=>{
    for (pusher of myall['pusher']){
        pusher.done = false;
    }
    var returnval = false;
    for(var j = 0; j < t / 5; j++){
        for (pusher of myall['pusher']){
            if (pusher.done) continue;
            var b = shrink(cellsize2 * pusher.dirx / t, cellsize2 * pusher.diry / t, pusher);
            returnval = returnval || b;
        }
    }
    if (!returnval) return false;
    draw();
    playSound(46796706);
    for(var i1 = 1; i1 < 5; i1++){
        setTimeout(()=>{
            playSound(46796706);
        }, i1 * anim / 5);
        setTimeout(()=>{
            for(var j = 0; j < t / 5; j++){
                for (pusher of myall['pusher']){
                    if (pusher.done) continue;
                    shrink(cellsize2 * pusher.dirx / t, cellsize2 * pusher.diry / t, pusher);
                }
            }
            draw();
        }, i1 * anim / 5);
    }
    setTimeout(draw, anim);
    return true;
};
moveTimes = (dx, dy, obj, t = 100)=>{
    newdx = dx * Math.cos(obj.angle) - dy * Math.sin(obj.angle);
    newdy = dy * Math.cos(obj.angle) + dx * Math.sin(obj.angle);
    dx = newdx;
    dy = newdy;
    for(var i1 = 0; i1 < t; i1++){
        if (i1 / (t / 5) == Math.floor(i1 / (t / 5))) sstate = getstate();
        b1 = move(dx / t, 0, obj, 1);
        b2 = move(0, dy / t, obj, 1);
        if (dy == 0 && !b1 || dx == 0 && !b2) {
            if (i1 / (t / 5) == Math.floor(i1 / (t / 5))) fromState(sstate);
            break;
        }
        ;
    }
    buttonCheck();
    draw();
};
playersprites = [
    '.aaa.|ababa|aaaaa|aaaaa|.aaa.',
    '.aaa.|aaaaa|babaa|aaaaa|.aaa.',
    '.aaa.|aaaaa|aaaaa|ababa|.aaa.',
    '.aaa.|aaaaa|aabab|aaaaa|.aaa.'
];
DoRestart = ()=>{
    if (againing) return;
    loadmylevel();
    draw();
};
DoUndo = ()=>{
    if (!againing) undo();
    againing = false;
    draw();
};
processInput = (dir)=>{
    ULBS();
    if (dir >= 0 && dir <= 3) {
        againing = true;
        mysprites['player'] = playersprites[dir];
        controls[dir].f();
    }
    draw();
};
nextLevel = ()=>{
    titleScreen = false;
    titleSelected = false;
    undostack = [];
    textMode = false;
    mylevel = 0;
    if (!!window.localStorage) {
        if (titleSelection == 1 && localStorage[document.URL] != undefined) mylevel = localStorage[document.URL];
        localStorage[document.URL] = mylevel;
    }
    loadmylevel();
    draw();
};
update = ()=>{
    if (textMode) return pushingItEngineUpdate();
    pollGamepads();
    timer += deltatime;
    input_throttle_timer += deltatime;
    if (quittingTitleScreen) {
        if (timer / 1000 > 0.3) {
            quittingTitleScreen = false;
            nextLevel();
        }
    }
    if (againing) {
        if (timer > againinterval && messagetext.length == 0) {
            if (processInput(-1)) {
                redraw();
                keyRepeatTimer = 0;
                autotick = 0;
            }
        }
    }
    if (quittingMessageScreen) {
        if (timer / 1000 > 0.15) {
            quittingMessageScreen = false;
            if (messagetext === '') {
                nextLevel();
            } else {
                messagetext = '';
                textMode = false;
                titleScreen = false;
                titleMode = curlevel > 0 || curlevelTarget !== null ? 1 : 0;
                titleSelected = false;
                titleSelection = 0;
                canvasResize();
                checkWin();
            }
        }
    }
    if (keybuffer.length > 0) {
        keyRepeatTimer += deltatime;
        var ticklength = throttle_movement ? repeatinterval : repeatinterval / Math.sqrt(keybuffer.length);
        if (keyRepeatTimer > ticklength) {
            keyRepeatTimer = 0;
            keyRepeatIndex = (keyRepeatIndex + 1) % keybuffer.length;
            var key = keybuffer[keyRepeatIndex];
            checkKey({
                keyCode: key
            }, false);
        }
    }
};
redraw = ()=>{
    if (textMode) {
        pushingItEngineRedraw();
    } else {
        draw();
    }
};
if (!titleScreen) {
    messagetext = 'POWER: ON';
    drawMessageScreen();
    drawMessageScreenOld = drawMessageScreen;
    drawMessageScreen = ()=>{
        drawMessageScreenOld();
        nextLevel();
    };
}
document.getElementById('homepagelink').href = '//www.puzzlescript.net';
document.getElementById('homepagelink').textContent = 'www.puzzlescript.net';
document.getElementById('hacklink').href = 'editor.html?hack=76ce230398172e7b0cbb8719b8c4f302';

// The native title menu sees one PuzzleScript board; this game owns 12 levels.
showContinueOptionOnTitleScreen = ()=>curlevel > 0 && curlevel < grid.length;
tick_lazy_function_generation(false);
if (titleScreen) {
    titleSelection = showContinueOptionOnTitleScreen() ? 1 : 0;
    generateTitleScreen();
}
canvasResize();
