/*
 * Add gesture support for mobile devices.
 */

window.Mobile = {};

//stolen from https://github.com/Modernizr/Modernizr/blob/master/feature-detects/touchevents.js
Mobile.hasTouch = function() {
    var bool;
    if(('ontouchstart' in window) || window.DocumentTouch && document instanceof DocumentTouch)     {
      bool = true;
    } else {
      /*
      //don't know what's happening with this, so commented it out
      var query = ['@media (',prefixes.join('touch-enabled),    ('),'heartz',')','{#modernizr{top:9px;position:absolute}}'].join('');
      testStyles(query, function( node ) {
        bool = node.offsetTop === 9;
      });*/
    }
    return bool;
}

Mobile.enable = function (force) {
    if (force || Mobile.hasTouch() && !Mobile._instance) {
        Mobile._instance = new Mobile.GestureHandler();
        Mobile._instance.bindEvents();
        Mobile._instance.bootstrap();
    }
    return Mobile._instance;
};

window.Mobile.GestureHandler = function () {
    this.initialize.apply(this, arguments);
};

Mobile.log = function (message) {
    var h1;
    h1 = document.getElementsByTagName('h1')[0];
    h1.innerHTML = "" + Math.random().toString().substring(4, 1) + "-" + message;
};

Mobile.debugDot = function (event) {
    var dot, body, style

    style = 'border-radius: 50px;' +
        'width: 5px;' +
        'height: 5px;' +
        'background: red;' +
        'position: absolute;' +
        'left: ' + event.touches[0].clientX + 'px;' +
        'top: ' + event.touches[0].clientY + 'px;';
    dot = document.createElement('div');
    dot.setAttribute('style', style);
    body = document.getElementsByTagName('body')[0];
    body.appendChild(dot);
};

(function (proto) {
    'use strict';

    // Minimum range to begin looking at the swipe direction, in pixels
    var SWIPE_THRESHOLD = 10;
    // Distance in pixels required to complete a swipe gesture.
    var SWIPE_DISTANCE = 50;
    // Time in milliseconds to complete the gesture.
    var SWIPE_TIMEOUT = 1000;
    // Time in milliseconds to repeat a motion if still holding down,
    // ... and not specified in state.metadata.key_repeat_interval.
    var DEFAULT_REPEAT_INTERVAL = 150;

    // Lookup table mapping action to keyCode.
    var CODE = {
        action:  88, // x
        left:    37, // left arrow
        right:   39, // right arrow
        up:      38, // up arrow
        down:    40, // down arrow
        undo:    85, // u
        restart: 82, // r
        quit:    27 // escape
    }

    var TAB_STRING = [
        '<div class="tab">',
        '  <div class="tab-affordance"></div>',
        '  <div class="tab-icon">',
        '    <div class="slice"></div>',
        '    <div class="slice"></div>',
        '  </div>',
        '</div>'
    ].join("\n");

    /** Bootstrap Methods **/

    proto.initialize = function () {
        this.firstPos = { x: 0, y: 0 };
        this.setTabAnimationRatio = this.setTabAnimationRatio.bind(this);
        this.setMenuAnimationRatio = this.setMenuAnimationRatio.bind(this);
        this.repeatTick = this.repeatTick.bind(this);
        this.isFocused = true;
    };

    // assign the element that will allow tapping to toggle focus.
    proto.setFocusElement = function (focusElement) {
        this.focusElement = focusElement;
        this.isFocused = false;
        this.buildFocusIndicator();
    };

    proto.bindEvents = function () {
        window.addEventListener('touchstart', this.onTouchStart.bind(this));
        window.addEventListener('touchend', this.onTouchEnd.bind(this));
        window.addEventListener('touchmove', this.onTouchMove.bind(this));
    };

    proto.bootstrap = function () {
        this.showTab();
        this.disableScrolling();
        if (!this.isAudioSupported()) {
            this.disableAudio();
        }
        this.disableSelection();
    };

    /** Event Handlers **/

    proto.onTouchStart = function (event) {
        if (this.isTouching) {
            return;
        }

        // Handle focus changes used in editor.
        this.handleFocusChange(event);
        if (!this.isFocused) {
            return;
        }

        if (event.target.tagName.toUpperCase() === 'A') {
            return;
        }
        this.isTouching = true;

        this.mayBeSwiping = true;
        this.gestured = false;

        this.swipeDirection = undefined;
        this.swipeDistance = 0;
        this.startTime = new Date().getTime();

        this.firstPos.x = event.touches[0].clientX;
        this.firstPos.y = event.touches[0].clientY;
    };

    proto.onTouchEnd = function (event) {
        if (!this.isFocused) {
            return;
        }
        if (!this.isTouching) {
            // If we're here, the menu event handlers had probably
            // canceled the touchstart event.
            return;
        }
        if (!this.gestured) {
            if (event.touches.length === 0 && event.target.id!=="unMuteButton" && event.target.id!=="muteButton" ) {
                this.handleTap();
            }
        }

        // The last finger to be removed from the screen lets us know
        // we aren't tracking anything.
        if (event.touches.length === 0) {
            this.isTouching = false;
            this.endRepeatWatcher();
        }
    };

    proto.onTouchMove = function (event) {
        if (!this.isFocused) {
            return;
        }
        if (levelEditorOpened){
            return;
        }
        if (this.isSuccessfulSwipe()) {
            this.handleSwipe(this.swipeDirection, this.touchCount);
            this.gestured = true;
            this.mayBeSwiping = false;
            this.beginRepeatWatcher(event);
        } else if (this.mayBeSwiping) {
            this.swipeStep(event);
        } else if (this.isRepeating) {
            this.repeatStep(event);
        }

        prevent(event);
        return false;
    };

    proto.handleFocusChange = function (event) {
        if (!this.focusElement) {
            return;
        }

        this.isFocused = this.isTouchInsideFocusElement(event);
        this.setFocusIndicatorVisibility(this.isFocused);
        
        canvas.focus();
        editor.display.input.blur();
    };

    proto.isTouchInsideFocusElement = function (event) {
        var canvasPosition;

        if (!event.touches || !event.touches[0]) {
            return false;
        }
        canvasPosition = this.absoluteElementPosition(this.focusElement);

        if (event.touches[0].clientX < canvasPosition.left ||
            event.touches[0].clientY < canvasPosition.top) {
            return false;
        }

        if (event.touches[0].clientX > canvasPosition.left + this.focusElement.clientWidth ||
            event.touches[0].clientY > canvasPosition.top + this.focusElement.clientHeight) {
            return false;
        }

        return true;
    };

    proto.setFocusIndicatorVisibility = function (isVisible) {
        var visibility;

        visibility = 'visible';
        if (!isVisible) {
            visibility = 'hidden';
        }
        // this.focusIndicator.setAttribute('style', 'visibility: ' + visibility + ';');
    };

    proto.absoluteElementPosition = function (element) {
        var position, body;

        position = {
            top: element.offsetTop || 0,
            left: element.offsetLeft || 0
        };
        body = document.getElementsByTagName('body')[0];
        position.top -= body.scrollTop || 0;

        while (true) {
            element = element.offsetParent;
            if (!element) {
                break;
            }
            position.top += element.offsetTop || 0;
            position.left += element.offsetLeft || 0;
        }

        return position;
    };

    proto.beginRepeatWatcher = function (event) {
        var repeatIntervalMilliseconds;
        if (this.repeatInterval) {
            return;
        }
        this.isRepeating = true;
        repeatIntervalMilliseconds = state.metadata.key_repeat_interval * 1000;
        if (isNaN(repeatIntervalMilliseconds) || !repeatIntervalMilliseconds) {
            repeatIntervalMilliseconds = DEFAULT_REPEAT_INTERVAL;
        }
        this.repeatInterval = setInterval(this.repeatTick, repeatIntervalMilliseconds);
        this.recenter(event);
    };

    proto.endRepeatWatcher = function () {
        if (this.repeatInterval) {
            clearInterval(this.repeatInterval);
            delete this.repeatInterval;
            this.isRepeating = false;
        }
    };

    proto.repeatTick = function () {
        if (this.isTouching) {
            this.handleSwipe(this.direction, this.touchCount);
        }
    };

    // Capture the location to consider the gamepad center.
    proto.recenter = function (event) {
        this.firstPos.x = event.touches[0].clientX;
        this.firstPos.y = event.touches[0].clientY;
    }

    /** Detection Helper Methods **/

    proto.isSuccessfulSwipe = function () {
        var isSuccessful;

        if (this.mayBeSwiping &&
            this.swipeDirection !== undefined &&
            this.swipeDistance >= SWIPE_DISTANCE) {
            isSuccessful = true;
        }

        return isSuccessful;
    };

    // Examine the current state to see what direction they're swiping and
    // if the gesture can still be considered a swipe.
    proto.swipeStep = function (event) {
        var currentPos, distance, currentTime;
        var touchCount;

        if (!this.mayBeSwiping) {
            return;
        }

        currentPos = {
            x: event.touches[0].clientX,
            y: event.touches[0].clientY
        };
        currentTime = new Date().getTime();
        touchCount = event.touches.length;

        this.swipeDistance = this.cardinalDistance(this.firstPos, currentPos);
        if (!this.swipeDirection) {
            if (this.swipeDistance > SWIPE_THRESHOLD) {
                // We've swiped far enough to decide what direction we're swiping in.
                this.swipeDirection = this.dominantDirection(this.firstPos, currentPos);
                this.touchCount = touchCount;
            }
        } else if (distance < SWIPE_DISTANCE) {
            // Now that they've committed to the swipe, look for misfires...

            direction = this.dominantDirection(this.firstPos, currentPos);
            // Cancel the swipe if the direction changes.
            if (direction !== this.swipeDirection) {
                this.mayBeSwiping = false;
            }
            // If they're changing touch count at this point, it's a misfire.
            if (touchCount < this.touchCount) {
                this.mayBeSwiping = false;
            }
        } else if (currentTime - this.startTime > SWIPE_TIMEOUT) {
            // Cancel the swipe if they took too long to finish.
            this.mayBeSwiping = false;
        }
    };

    proto.repeatStep = function (event) {
        var currentPos, distance, currentTime;
        var newDistance, direction;

        currentPos = {
            x: event.touches[0].clientX,
            y: event.touches[0].clientY
        };

        newDistance = this.cardinalDistance(this.firstPos, currentPos);

        if (newDistance >= SWIPE_DISTANCE) {
            this.swipeDirection = this.dominantDirection(this.firstPos, currentPos);
            this.recenter(event);
        }
    };

    // Find the distance traveled by the swipe along compass directions.
    proto.cardinalDistance = function (firstPos, currentPos) {
        var xDist, yDist;

        xDist = Math.abs(firstPos.x - currentPos.x);
        yDist = Math.abs(firstPos.y - currentPos.y);

        return Math.max(xDist, yDist);
    };

    // Decide which direction the touch has moved farthest.
    proto.dominantDirection = function (firstPos, currentPos) {
        var dx, dy;
        var dominantAxis, dominantDirection;

        dx = currentPos.x - firstPos.x;
        dy = currentPos.y - firstPos.y;

        dominantAxis = 'x';
        if (Math.abs(dy) > Math.abs(dx)) {
            dominantAxis = 'y';
        }

        if (dominantAxis === 'x') {
            if (dx > 0) {
                dominantDirection = 'right';
            } else {
                dominantDirection = 'left';
            }
        } else {
            if (dy > 0) {
                dominantDirection = 'down';
            } else {
                dominantDirection = 'up';
            }
        }

        return dominantDirection;
    };

    /** Action Methods **/

    // Method to be called when we've detected a swipe and some action
    // is called for.
    proto.handleSwipe = function (direction, touchCount) {
        if (touchCount === 1) {
            this.emitKeydown(this.swipeDirection);
        } else if (touchCount > 1) {
            // Since this was a multitouch gesture, open the menu.
            this.toggleMenu();
        }
    };

    proto.handleTap = function () {
        this.emitKeydown('action');
    };

    // Fake out keypresses to acheive the desired effect.
    proto.emitKeydown = function (input) {
        var event;

        event = { keyCode: CODE[input] };

        this.fakeCanvasFocus();
        // Press, then release key.
        onKeyDown(event);
        onKeyUp(event);
    };

    proto.fakeCanvasFocus = function () {
        var canvas;

        canvas = document.getElementById('gameCanvas');
        onMouseDown({
            button: 0,
            target: canvas
        });
    };

    proto.toggleMenu = function () {
        if (this.isMenuVisible) {
            this.hideMenu();
        } else {
            this.showMenu();
        }
    };

    proto.showMenu = function () {
        if (!this.menuElem) {
            this.buildMenu();
        }
        this.getAnimatables().menu.animateUp();
        this.isMenuVisible = true;
        this.hideTab();
    };

    proto.hideMenu = function () {
        if (this.menuElem) {
            this.getAnimatables().menu.animateDown();
        }
        this.isMenuVisible = false;
        this.showTab();
    };

    proto.getAnimatables = function () {
        var self = this;
        if (!this._animatables) {
            this._animatables = {
                tab: Animatable('tab', 0.1, self.setTabAnimationRatio),
                menu: Animatable('menu', 0.1, self.setMenuAnimationRatio)
            }
        }
        return this._animatables;
    };

    proto.showTab = function () {
        if (!this.tabElem) {
            this.buildTab();
        }
        this.getAnimatables().tab.animateDown();
    };

    proto.hideTab = function () {
        if (this.tabElem) {
            this.tabElem.setAttribute('style', 'display: none;');
        }
        this.getAnimatables().tab.animateUp();
    };

    proto.buildTab = function () {
        var self = this;
        var tempElem, body;
        var openCallback;
        var tabElem;
        var assemblyElem;

        tempElem = document.createElement('div');
        tempElem.innerHTML = TAB_STRING;
        assemblyElem = tempElem.children[0];

        openCallback = function (event) {
            event.stopPropagation();
            self.showMenu();
        };
        this.tabAffordance = assemblyElem.getElementsByClassName('tab-affordance')[0];
        this.tabElem = assemblyElem.getElementsByClassName('tab-icon')[0];

        this.tabAffordance.addEventListener('touchstart', openCallback);
        this.tabElem.addEventListener('touchstart', openCallback);

        body = document.getElementsByTagName('body')[0];
        body.appendChild(assemblyElem);
    };

    proto.buildMenu = function () {
        var self = this;
        var tempElem, body;
        var undo, restart, quit;
        var closeTab;
        var closeCallback;

        tempElem = document.createElement('div');
        tempElem.innerHTML = this.buildMenuString(state);
        this.menuElem = tempElem.children[0];
        this.closeElem = this.menuElem.getElementsByClassName('close')[0];

        closeCallback = function (event) {
            event.stopPropagation();
            self.hideMenu();
        };
        this.closeAffordance = this.menuElem.getElementsByClassName('close-affordance')[0];
        closeTab = this.menuElem.getElementsByClassName('close')[0];
        this.closeAffordance.addEventListener('touchstart', closeCallback);
        closeTab.addEventListener('touchstart', closeCallback);

        undo = this.menuElem.getElementsByClassName('undo')[0];
        if (undo) {
            undo.addEventListener('touchstart', function (event) {
                event.stopPropagation();
                self.emitKeydown('undo');
            });
        }
        restart = this.menuElem.getElementsByClassName('restart')[0];
        if (restart) {
            restart.addEventListener('touchstart', function (event) {
                event.stopPropagation();
                self.emitKeydown('restart');
            });
        }

        quit = this.menuElem.getElementsByClassName('quit')[0];
        quit.addEventListener('touchstart', function (event) {
            event.stopPropagation();
            self.emitKeydown('quit');
        });

        body = document.getElementsByTagName('body')[0];
        body.appendChild(this.menuElem);
    };

    proto.buildMenuString = function (state) {
    // Template for the menu.
        var itemCount, menuLines;
        var noUndo, noRestart;

        noUndo = state.metadata.noundo;
        noRestart = state.metadata.norestart;

        itemCount = 3;
        if (noUndo) {
            itemCount -= 1;
        }
        if (noRestart) {
            itemCount -= 1;
        }

        menuLines = [
            '<div class="mobile-menu item-count-' + itemCount + '">',
            '  <div class="close-affordance"></div>',
            '  <div class="close">',
            '    <div class="slice"></div>',
            '    <div class="slice"></div>',
            '  </div>'
        ];

        if (!noUndo) {
            menuLines.push('  <div class="undo button">Undo</div>');
        }
        if (!noRestart) {
            menuLines.push('  <div class="restart button">Restart</div>');
        }
        menuLines = menuLines.concat([
            '  <div class="quit button">Quit to Menu</div>',
            '  <div class="clear"></div>',
            '</div>'
        ]);

        return menuLines.join("\n");
    };

    proto.buildFocusIndicator = function () {
        var focusElementParent;
        this.focusIndicator = document.createElement('DIV');
        this.focusIndicator.setAttribute('class', 'tapFocusIndicator');
        this.focusIndicator.setAttribute('style', 'visibility: hidden;');

        focusElementParent = this.focusElement.parentNode;
        focusElementParent.appendChild(this.focusIndicator);
    };

    proto.setTabAnimationRatio = function (ratio) {
        var LEFT = 18;
        var RIGHT = 48 + 18;
        var size, opacityString;
        var style;

        // Round away any exponents that might appear.
        ratio = Math.round((ratio) * 1000) / 1000;
        if (ratio >= 0.999) {
            this.tabAffordance.setAttribute('style', 'display: none;');
        } else {
            this.tabAffordance.setAttribute('style', 'display: block;');
        }
        size = RIGHT * ratio + LEFT * (1 - ratio);
        opacityString = 'opacity: ' + (1 - ratio) + ';';
        style = opacityString + ' ' +
            'width: ' + size + 'px;';
        this.tabElem.setAttribute('style', style);
    };

    proto.setMenuAnimationRatio = function (ratio) {
        var LEFT = -48 - 18;
        var RIGHT = -18;
        var size, opacityString;
        var style;

        // Round away any exponents that might appear.
        ratio = Math.round((ratio) * 1000) / 1000;

        size = RIGHT * ratio + LEFT * (1 - ratio);
        opacityString = 'opacity: ' + ratio + ';';
        style = 'left: ' + (size - 4) + 'px; ' +
            opacityString + ' ' +
            'width: ' + (-size) + 'px;';
        ratio = Math.round((ratio) * 1000) / 1000;

        if (ratio <= 0.001) {
            this.closeAffordance.setAttribute('style', 'display: none;');
            opacityString="display:none;"
        } else {
            this.closeAffordance.setAttribute('style', 'display: block;');
        }

        this.closeElem.setAttribute('style', style);

        this.menuElem.setAttribute('style', opacityString);
    };

    proto.disableScrolling = function() {
        var style = {
            height: "100%",
            overflow: "hidden",
            position: "fixed",
            width: "100%"
        }
        
        var styleString = "";
        for (var key in style) {
            styleString += key + ": " + style[key] + "; ";
        }

        document.body.setAttribute('style', styleString)
    }

    /** Audio Methods **/

    proto.disableAudio = function () {
        // Overwrite the playseed function to disable it.
        window.playSeed = function () {};
    };

    proto.isAudioSupported = function () {
        var isAudioSupported = true;

        if (typeof webkitAudioContext !== 'undefined') {
            // We may be on Mobile Safari, which throws up
            // 'Operation not Supported' alerts when we attempt to
            // play Audio elements with "data:audio/wav;base64"
            // encoded HTML5 Audio elements.
            //
            // Switching to MP3 encoded audio may be the way we have
            // to go to get Audio working on mobile devices.
            //
            // e.g. https://github.com/rioleo/webaudio-api-synthesizer
            isAudioSupported = false;
        }

        return isAudioSupported;
    };

    /** Other HTML5 Stuff **/

    proto.disableSelection = function () {
        var body;
        body = document.getElementsByTagName('body')[0];
        body.setAttribute('class', body.getAttribute('class') + ' disable-select');
    };

}(window.Mobile.GestureHandler.prototype));

window.Animator = function () {
    this.initialize.apply(this, arguments);
};

(function (proto) {
    proto.initialize = function () {
        this._animations = {};
        this.tick = this.tick.bind(this);
    };

    proto.animate = function (key, tick) {
        this._animations[key] = tick;
        this.wakeup();
    };

    proto.wakeup = function () {
        if (this._isAnimating) {
            return;
        }
        this._isAnimating = true;
        this.tick();
    };

    proto.tick = function () {
        var key;
        var isFinished, allFinished;
        var toRemove, index;

        toRemove = [];
        allFinished = true;
        for (key in this._animations) {
            if (!this._animations.hasOwnProperty(key)) {
                return;
            }
            isFinished = this._animations[key]();
            if (!isFinished) {
                allFinished = false;
            } else {
                toRemove.push(key);
            }
        }

        if (!allFinished) {
            requestAnimationFrame(this.tick);
        } else {
            for (index = 0; index < toRemove.length; toRemove++) {
                delete this._isAnimating[toRemove[index]];
            }
            this._isAnimating = false;
        }
    };

}(window.Animator.prototype));

window.Animator.getInstance = function () {
    if (!window.Animator._instance) {
        window.Animator._instance = new window.Animator();
    }
    return window.Animator._instance;
};

function Animatable(key, increment, update) {
    var ratio;
    var handles;

    handles = {
        animateUp: function () {
            Animator.getInstance().animate(key, tickUp);
        },
        animateDown: function () {
            Animator.getInstance().animate(key, tickDown);
        }
    };

    ratio = 0;

    function tickUp () {
        var isFinished;
        ratio += increment;
        if (ratio >= 1.0) {
            isFinished = true;
            ratio = 1;
        }
        update(ratio);
        return isFinished;
    };

    function tickDown () {
        var isFinished;
        ratio -= increment;
        if (ratio <= 0.0) {
            isFinished = true;
            ratio = 0;
        }
        update(ratio);
        return isFinished;
    };

    return handles;
};


// http://paulirish.com/2011/requestanimationframe-for-smart-animating/
// http://my.opera.com/emoller/blog/2011/12/20/requestanimationframe-for-smart-er-animating

// requestAnimationFrame polyfill by Erik Möller. fixes from Paul Irish and Tino Zijdel

// MIT license

(function() {
    'use strict';

    var VENDORS = ['ms', 'moz', 'webkit', 'o'];
    var index, lastTime;

    for (index = 0; index < VENDORS.length && !window.requestAnimationFrame; index++) {
        window.requestAnimationFrame = window[VENDORS[index] + 'RequestAnimationFrame'];
        window.cancelAnimationFrame = window[VENDORS[index] + 'CancelAnimationFrame'];
        if (!window.cancelAnimationFrame) {
            window.cancelAnimationFrame = window[VENDORS[index] + 'CancelRequestAnimationFrame'];
        }
    }

    if (!window.requestAnimationFrame) {
        lastTime = 0;
        window.requestAnimationFrame = function(callback, element) {
            var currTime, timeToCall, id;

            currTime = new Date().getTime();
            timeToCall = Math.max(0, 16 - (currTime - lastTime));
            id = window.setTimeout(function() {
                callback(currTime + timeToCall);
            }, timeToCall);
            lastTime = currTime + timeToCall;

            return id;
        };
    }

    if (!window.cancelAnimationFrame) {
        window.cancelAnimationFrame = function(id) {
            clearTimeout(id);
        };
    }

    Mobile.enable();
}());
