'use strict';

/*
 * Add gesture support for mobile devices.
 */

window.Mobile = {};

//stolen from https://github.com/Modernizr/Modernizr/blob/master/feature-detects/touchevents.js
Mobile.hasTouch = function () {
    var bool;
    if (navigator.maxTouchPoints > 0 || ('ontouchstart' in window) || window.DocumentTouch && document instanceof DocumentTouch) {
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
    if ((force || Mobile.hasTouch()) && !Mobile._instance) {
        Mobile._instance = new Mobile.GestureHandler();
        Mobile._instance.bindEvents();
        Mobile._instance.bootstrap();
    }
    return Mobile._instance;
};

// Shared by the web player, exported games and the editor preview.
Mobile.setupPlayerControls = function () {
    var toolbar = document.getElementById('playerToolbar');
    if (!toolbar) {
        return;
    }
    Mobile.playerToolbar = toolbar;
    var root = Mobile.playerRoot = toolbar.parentNode;
    var canvas = document.getElementById('gameCanvas');
    if (!IDE || hasLoadedGame()) { canvas.setAttribute('tabindex', '-1'); }
    var releaseButtons = [];
    // Consume the compatibility click even if activation hides its button
    // (mute/unmute) or opens a menu under the pointer before click dispatch.
    var skipPointerClick = false;
    document.addEventListener('pointerdown', function () { skipPointerClick = false; }, true);
    document.addEventListener('click', function (event) {
        if (skipPointerClick && event.detail > 0) {
            skipPointerClick = false;
            event.preventDefault();
            event.stopImmediatePropagation();
        }
    }, true);
    Mobile.releasePlayerButtons = function () {
        releaseButtons.forEach(function (release) { release(); });
    };
    var icons = {
        expand: '<path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5"/>',
        collapse: '<path d="M3 8h5V3m13 5h-5V3M8 21v-5H3m13 5v-5h5"/>',
        menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
        sound: '<path d="M11 4 6 8H3v8h3l5 4Z"/><path d="M15 8a6 6 0 0 1 0 8m3-11a10 10 0 0 1 0 14"/>',
        muted: '<path d="M11 4 6 8H3v8h3l5 4Z"/><path d="m16 9 6 6m0-6-6 6"/>',
        restart: '<path d="M4 10a8 8 0 1 1 1 7M4 4v6h6"/>',
        undo: '<path d="m8 4-5 5 5 5M3 9h11a6 6 0 0 1 0 12h-2"/>',
        action: '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="2"/>',
        up: '<path d="m7 14 5-5 5 5"/>',
        down: '<path d="m7 10 5 5 5-5"/>',
        left: '<path d="m14 7-5 5 5 5"/>',
        right: '<path d="m10 7 5 5-5 5"/>'
    };
    function input(code) {
        Mobile.releasePlayerButtons();
        lastDownTarget = canvas;
        onKeyDown({ keyCode: code });
        onKeyUp({ keyCode: code });
    }
    function button(id, label, icon, callback, heldCode, caption) {
        var element = document.createElement('button');
        element.type = 'button';
        element.id = id;
        element.className = 'player-button';
        element.setAttribute('aria-label', label);
        element.title = label;
        element.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">' + icons[icon] + '</svg>';
        if (caption) {
            var text = document.createElement('span');
            text.textContent = caption;
            element.appendChild(text);
        }
        function activate(keyboard) {
            callback(keyboard);
            if (keyboard && (id === 'muteButton' || id === 'unMuteButton')) {
                document.getElementById(muted ? 'unMuteButton' : 'muteButton').focus();
            } else if (!keyboard && !Mobile.playerMenuOpen) {
                canvas.focus({ preventScroll: true });
            }
        }
        var pointer = null;
        var pointerUsed = false;
        function release() {
            if (pointer && pointer.ownsKey) {
                onKeyUp({ keyCode: heldCode });
            }
            pointer = null;
            element.classList.remove('is-held');
        }
        releaseButtons.push(release);
        function inside(event) {
            var bounds = element.getBoundingClientRect();
            return event.clientX >= bounds.left && event.clientX <= bounds.right &&
                event.clientY >= bounds.top && event.clientY <= bounds.bottom;
        }
        if (window.PointerEvent) {
            element.addEventListener('pointerdown', function (event) {
                if (event.button !== 0 || element.disabled || pointer) { return; }
                event.preventDefault();
                pointerUsed = true;
                skipPointerClick = true;
                pointer = { id: event.pointerId, ownsKey: heldCode && keybuffer.indexOf(heldCode) < 0 };
                element.setPointerCapture(event.pointerId);
                element.classList.add('is-held');
                if (heldCode) {
                    // Use the engine's repeat timing, throttling and norepeat_action.
                    // Setting the input target directly also allows two-thumb input.
                    lastDownTarget = canvas;
                    onKeyDown({ keyCode: heldCode });
                }
            });
            element.addEventListener('pointermove', function (event) {
                if (pointer && event.pointerId === pointer.id && !inside(event)) { release(); }
            });
            element.addEventListener('pointerup', function (event) {
                if (!pointer || event.pointerId !== pointer.id) { return; }
                var tap = !heldCode && !element.disabled && inside(event);
                release();
                if (tap) { activate(false); }
            });
            element.addEventListener('pointercancel', release);
            element.addEventListener('lostpointercapture', release);
        } else {
            // Older touch browsers can suppress the first click after a swipe.
            var touchStart;
            element.addEventListener('touchstart', function (event) {
                touchStart = event.touches.length === 1 ? event.touches[0] : null;
            });
            element.addEventListener('touchmove', function (event) {
                if (touchStart && (event.touches.length !== 1 ||
                    Math.abs(event.touches[0].clientX - touchStart.clientX) > 10 ||
                    Math.abs(event.touches[0].clientY - touchStart.clientY) > 10)) { touchStart = null; }
            });
            element.addEventListener('touchcancel', function () { touchStart = null; });
            element.addEventListener('touchend', function (event) {
                var tap = touchStart && !element.disabled && !event.touches.length && inside(event.changedTouches[0]);
                touchStart = null;
                event.preventDefault();
                if (tap) { activate(false); }
            });
        }
        element.addEventListener('click', function (event) {
            if (!pointerUsed || event.detail === 0) { activate(event.detail === 0); }
            pointerUsed = false;
        });
        return element;
    }
    var leading = document.createElement('div');
    var trailing = document.createElement('div');
    leading.className = trailing.className = 'player-controls';
    var menuButton = button('gameMenuButton', 'Game menu', 'menu', function (keyboard) { setMenu(!Mobile.playerMenuOpen, keyboard); });
    menuButton.classList.add('touch-control');
    menuButton.setAttribute('aria-expanded', 'false');
    menuButton.setAttribute('aria-controls', 'playerMenu');
    leading.appendChild(menuButton);
    trailing.appendChild(button('muteButton', IDE ? 'Mute sound' : 'Mute sound (M)', 'sound', muteAudio));
    trailing.appendChild(button('unMuteButton', IDE ? 'Unmute sound' : 'Unmute sound (M)', 'muted', unMuteAudio));
    toolbar.insertBefore(leading, toolbar.firstChild);
    toolbar.appendChild(trailing);
    if (IDE) {
        var inertPanels = [];
        var expand = button('expandPreviewButton', 'Expand game preview', 'expand', function () {
            setExpanded(!root.classList.contains('preview-expanded'));
        });
        trailing.appendChild(expand);
        function setExpanded(expanded) {
            Mobile.releasePlayerButtons();
            if (Mobile._instance) { Mobile._instance.cancelTouch(); }
            keybuffer = [];
            root.classList.toggle('preview-expanded', expanded);
            expand.setAttribute('aria-expanded', String(expanded));
            expand.setAttribute('aria-label', expanded ? 'Return to editor' : 'Expand game preview');
            expand.title = expanded ? 'Return to editor' : 'Expand game preview';
            expand.querySelector('svg').innerHTML = icons[expanded ? 'collapse' : 'expand'];
            if (expanded) {
                inertPanels = Array.from(document.body.children).filter(function (element) {
                    return !element.contains(root) && !element.inert;
                });
                inertPanels.forEach(function (element) { element.inert = true; });
            } else {
                inertPanels.forEach(function (element) { element.inert = false; });
                inertPanels = [];
            }
            Mobile.layoutPlayerControls();
            canvasResize();
        }
        expand.setAttribute('aria-expanded', 'false');
        root.addEventListener('keydown', function (event) {
            if (!root.classList.contains('preview-expanded') || Mobile.playerMenuOpen) { return; }
            if (event.key === 'Tab') {
                var controls = Array.from(root.querySelectorAll('button:not(:disabled), input, select, a[href]')).filter(function (element) { return element.getClientRects().length; });
                var first = controls[0], last = controls[controls.length - 1];
                if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
                else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
                event.stopPropagation();
            }
            if (event.key === 'Escape') {
                event.preventDefault();
                event.stopPropagation();
                setExpanded(false);
                expand.focus({ preventScroll: true });
            }
        });
    }
    killAudioButton();

    var deck = document.createElement('div');
    deck.className = 'player-deck';
    deck.id = 'playerDeck';
    var movement = document.createElement('div');
    movement.className = 'player-movement';
    movement.setAttribute('role', 'group');
    movement.setAttribute('aria-label', 'Movement');
    // A single quiet silhouette groups the directions; the native buttons
    // retain their full touch targets and separate keyboard focus.
    movement.innerHTML = '<svg class="movement-surface movement-surface-portrait" aria-hidden="true" focusable="false"><path/></svg>' +
        '<svg class="movement-surface movement-surface-landscape" aria-hidden="true" focusable="false"><path/></svg>';
    [['up',38],['left',37],['down',40],['right',39]].forEach(function (direction) {
        var element = button('move' + direction[0], 'Move ' + direction[0], direction[0], function () { input(direction[1]); }, direction[1]);
        element.classList.add('move-' + direction[0]);
        movement.appendChild(element);
    });
    var actions = document.createElement('div');
    actions.className = 'player-actions';
    actions.setAttribute('role', 'group');
    actions.setAttribute('aria-label', 'Game actions');
    actions.appendChild(button('restartButton', 'Restart level (R)', 'restart', function () { input(82); }, null, 'Restart'));
    actions.appendChild(button('undoButton', 'Undo move (Z)', 'undo', function () { input(90); }, 90, 'Undo'));
    actions.appendChild(button('actionButton', 'Action (X)', 'action', function () { input(88); }, 88, 'Action'));
    deck.appendChild(movement);
    deck.appendChild(actions);
    document.querySelector('.gameContainer').after(deck);

    var menu = document.createElement('div');
    menu.id = 'playerMenu';
    menu.className = 'player-menu';
    menu.hidden = true;
    menu.innerHTML = '<div class="player-menu-panel" role="dialog" aria-modal="true" aria-label="Game menu">' +
        '<div class="player-menu-header"><h2>Game menu</h2><button type="button" id="closeMenuButton" class="player-button" aria-label="Close menu" title="Close menu"><svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="m6 6 12 12M18 6 6 18"/></svg></button></div>' +
        '<label><span>Button size</span><select id="controlSize"><option value="48">Compact</option><option value="56" selected>Comfortable</option><option value="64">Large</option></select></label>' +
        '<label><span>Position</span><select id="controlPosition"><option value="12">Near edges</option><option value="24" selected>Inset</option><option value="36">Further in</option></select></label>' +
        '<div class="player-menu-toggles"><label><span>Button labels</span><input type="checkbox" id="buttonLabels" checked></label>' +
        '<label><span>Swap control sides</span><input type="checkbox" id="swapControls"></label>' +
        '<label><span>Overlay controls</span><input type="checkbox" id="overlayControls" checked></label></div>' +
        '<div class="player-menu-footer"><button type="button" id="backToTitleButton">Back to title</button></div></div>';
    root.appendChild(menu);
    var panel = menu.firstChild;
    panel.tabIndex = -1;
    var links = document.getElementById('playerLinks');
    if (links) {
        links.hidden = false;
        if (Mobile.hasTouch()) { panel.querySelector('.player-menu-footer').appendChild(links); }
    }
    function setMenu(open, keyboard) {
        Mobile.releasePlayerButtons();
        if (Mobile._instance) { Mobile._instance.cancelTouch(); }
        keybuffer = [];
        Mobile.playerMenuOpen = open;
        menu.hidden = !open;
        menuButton.setAttribute('aria-expanded', String(open));
        if (open) {
            lastDownTarget = null;
            // Pointer users enter the dialog without a keyboard focus ring.
            (keyboard ? document.getElementById('closeMenuButton') : panel).focus({ preventScroll: true });
        } else {
            (keyboard ? menuButton : canvas).focus({ preventScroll: true });
            lastDownTarget = canvas;
        }
    }
    Mobile.setPlayerMenu = setMenu;
    document.getElementById('backToTitleButton').addEventListener('click', function (event) { setMenu(false, event.detail === 0); input(27); });
    document.getElementById('closeMenuButton').addEventListener('click', function (event) { setMenu(false, event.detail === 0); });
    menu.addEventListener('click', function (event) { if (event.target === menu) { setMenu(false); } });
    menu.addEventListener('keydown', function (event) {
        if (event.key === 'Escape') { event.preventDefault(); setMenu(false, true); }
        if (event.key === 'Tab') {
            var focusable = Array.from(panel.querySelectorAll('button:not(:disabled), input, select, a[href]'));
            var first = focusable[0], last = focusable[focusable.length - 1];
            if (event.shiftKey && (document.activeElement === first || document.activeElement === panel)) { event.preventDefault(); last.focus(); }
            else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
        }
        event.stopPropagation();
    });
    menu.addEventListener('keyup', function (event) { event.stopPropagation(); });
    [['buttonLabels','controls-labels'],['swapControls','controls-swapped'],['overlayControls','controls-overlay']].forEach(function (setting) {
        var control = document.getElementById(setting[0]);
        var key = 'puzzlescript:' + setting[0];
        try {
            var saved = storage_get(key);
            if (saved === 'true' || saved === 'false') { control.checked = saved === 'true'; }
        } catch (error) { /* Keep the default when storage is unavailable. */ }
        root.classList.toggle(setting[1], control.checked);
        control.addEventListener('change', function () {
            Mobile.releasePlayerButtons();
            root.classList.toggle(setting[1], control.checked);
            try { storage_set(key, String(control.checked)); } catch (error) { /* The preference still works for this session. */ }
            canvasResize();
        });
    });


    var sizeControl = document.getElementById('controlSize');
    var positionControl = document.getElementById('controlPosition');
    var movementSurfaceSize;
    function layoutMovementSurface(size) {
        if (movementSurfaceSize === size) { return; }
        movementSurfaceSize = size;
        // Match the buttons' 14px circular corners and centre the 1px stroke
        // half a pixel inside their border boxes. Keep the grid's 4px gaps.
        var outerRadius = 13.5;
        var innerRadius = Math.min(14.5, (size + 4) / 2 - outerRadius);
        function draw(selector, width, height, points) {
            var surface = movement.querySelector(selector);
            surface.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
            var path = '';
            points.forEach(function (point, index) {
                var previous = points[(index + points.length - 1) % points.length];
                var next = points[(index + 1) % points.length];
                var dx = Math.sign(point[0] - previous[0]), dy = Math.sign(point[1] - previous[1]);
                var nx = Math.sign(next[0] - point[0]), ny = Math.sign(next[1] - point[1]);
                var convex = dx * ny - dy * nx > 0;
                var radius = convex ? outerRadius : innerRadius;
                path += (index ? 'L' : 'M') + (point[0] - dx * radius) + ' ' + (point[1] - dy * radius) +
                    'A' + radius + ' ' + radius + ' 0 0 ' + (convex ? 1 : 0) + ' ' +
                    (point[0] + nx * radius) + ' ' + (point[1] + ny * radius);
            });
            surface.firstElementChild.setAttribute('d', path + 'Z');
        }
        var wide = 3 * size + 8, short = 2 * size + 4;
        var left = size + 4.5, right = 2 * size + 3.5, row = size + 4.5;
        draw('.movement-surface-portrait', wide, short, [
            [left, .5], [right, .5], [right, row], [wide - .5, row],
            [wide - .5, short - .5], [.5, short - .5], [.5, row], [left, row]
        ]);
        left = (size + 4) / 2 + .5;
        right = left + size - 1;
        var bottomRow = 2 * size + 3.5;
        draw('.movement-surface-landscape', short, wide, [
            [left, .5], [right, .5], [right, row], [short - .5, row],
            [short - .5, bottomRow], [right, bottomRow], [right, wide - .5],
            [left, wide - .5], [left, bottomRow], [.5, bottomRow], [.5, row], [left, row]
        ]);
    }
    Mobile.layoutPlayerControls = function () {
        var size = Number(sizeControl.value);
        var inset = Number(positionControl.value);
        var actionWidth = Math.max(60, size);
        var landscape = window.matchMedia('(orientation: landscape)').matches;
        var movementWidth = landscape ? 2 * size + 4 : 3 * size + 8;
        var actionsWidth = 2 * actionWidth + 4;
        var style = root.style;
        style.setProperty('--control-size', size + 'px');
        layoutMovementSurface(size);
        style.setProperty('--action-width', actionWidth + 'px');
        style.setProperty('--movement-width', movementWidth + 'px');
        style.setProperty('--control-lift', (inset + 8) + 'px');
        style.setProperty('--deck-inset', inset + 'px');
        var padding = getComputedStyle(deck);
        var safeWidth = Math.max(0, parseFloat(padding.paddingLeft) - inset) +
            Math.max(0, parseFloat(padding.paddingRight) - inset);
        var width = root.clientWidth - safeWidth;
        // Keep the chosen target size: stack actions when the thumb groups
        // would collide, and limit only the edge inset on very narrow screens.
        var stacked = movementWidth + actionsWidth + 12 + 2 * inset > width;
        if (stacked) { actionsWidth = actionWidth; }
        root.classList.toggle('controls-stacked', stacked);
        root.classList.toggle('controls-too-small', IDE && (width < movementWidth + actionsWidth + 28 || root.clientHeight < 300));
        style.setProperty('--actions-width', actionsWidth + 'px');
        style.setProperty('--deck-inset', Math.max(8, Math.min(inset,
            Math.floor((width - movementWidth - actionsWidth - 12) / 2))) + 'px');
    };
    [[sizeControl, ['48','56','64']], [positionControl, ['12','24','36']]].forEach(function (setting) {
        var control = setting[0];
        var key = 'puzzlescript:' + control.id;
        try {
            var saved = storage_get(key);
            if (setting[1].indexOf(saved) >= 0) { control.value = saved; }
        } catch (error) { /* Use the comfortable default without storage. */ }
        control.addEventListener('change', function () {
            Mobile.releasePlayerButtons();
            Mobile.layoutPlayerControls();
            try { storage_set(key, control.value); } catch (error) { /* Session preference still applies. */ }
            canvasResize();
        });
    });
    Mobile.layoutPlayerControls();

    // Keep the existing pixel scale in overlay mode. If the board can fit clear
    // of the visible buttons, use the nearest such position to its usual centre.
    Mobile.positionPlayerBoard = function (width, height, x, y) {
        if (!root.classList.contains('touch-enabled') ||
            !root.classList.contains('controls-overlay')) { return null; }
        Mobile.syncPlayerControls();
        var maxX = Math.floor(canvas.width - width);
        var maxY = Math.floor(canvas.height - height);
        if (maxX < 0 || maxY < 0) { return null; }
        var origin = canvas.getBoundingClientRect();
        var obstacles = Array.from(deck.querySelectorAll('button')).filter(function (element) {
            return element.getClientRects().length > 0;
        }).map(function (element) {
            var bounds = element.getBoundingClientRect();
            return { left: bounds.left - origin.left, right: bounds.right - origin.left,
                top: bounds.top - origin.top, bottom: bounds.bottom - origin.top };
        });
        // Every closest clear placement lies at the centre, a canvas edge, or
        // a button edge. Try a little breathing room first, then a flush fit.
        function place(gap) {
            var columns = [Math.floor(x), 0, maxX];
            var rows = [Math.floor(y), 0, maxY];
            obstacles.forEach(function (bounds) {
                columns.push(Math.floor(bounds.left - width - gap), Math.ceil(bounds.right + gap));
                rows.push(Math.floor(bounds.top - height - gap), Math.ceil(bounds.bottom + gap));
            });
            var best = null;
            var distance = Infinity;
            columns.forEach(function (left) {
                if (left < 0 || left > maxX) { return; }
                rows.forEach(function (top) {
                    if (top < 0 || top > maxY) { return; }
                    var shift = (left - x) * (left - x) + (top - y) * (top - y);
                    if (shift >= distance) { return; }
                    var clear = obstacles.every(function (bounds) {
                        return left + width + gap <= bounds.left || left >= bounds.right + gap ||
                            top + height + gap <= bounds.top || top >= bounds.bottom + gap;
                    });
                    if (clear) { best = { x: left, y: top }; distance = shift; }
                });
            });
            return best;
        }
        return place(8) || place(0);
    };

    // UI gestures must never also become a game tap or held swipe.
    [toolbar, deck, menu].forEach(function (surface) {
        surface.addEventListener('touchstart', function (event) {
            if (Mobile._instance) { Mobile._instance.cancelTouch(); }
            event.stopPropagation();
        });
        surface.addEventListener('mousedown', function (event) { event.stopPropagation(); });
        function buttonKey(event) {
            if (event.keyCode === 13 || event.keyCode === 32) { event.stopPropagation(); }
        }
        surface.addEventListener('keydown', buttonKey);
        surface.addEventListener('keyup', buttonKey);
    });
    function releaseInput() {
        Mobile.releasePlayerButtons();
        if (Mobile._instance) { Mobile._instance.cancelTouch(); }
    }
    window.addEventListener('blur', releaseInput);
    window.addEventListener('resize', function () {
        releaseInput();
        Mobile.layoutPlayerControls();
        canvasResize();
    });
    document.addEventListener('visibilitychange', function () { if (document.hidden) { releaseInput(); } });
    if (IDE && window.ResizeObserver) {
        new ResizeObserver(function () {
            Mobile.layoutPlayerControls();
            canvasResize();
        }).observe(root);
    }
    Mobile.syncPlayerControls();
};

Mobile.syncPlayerControls = function () {
    var toolbar = Mobile.playerToolbar;
    if (!toolbar) { return; }
    var root = Mobile.playerRoot;
    var unavailable = IDE && (!hasLoadedGame() || levelEditorOpened);
    var noUndo = 'noundo' in state.metadata;
    var noRestart = 'norestart' in state.metadata;
    var noAction = 'noaction' in state.metadata && !textMode;
    var noHistory = backups.length === 0;
    var title = state.metadata.title || (IDE ? 'Game preview' : document.getElementById('gametitle').textContent);
    var screen = [titleScreen, textMode, curlevel].join('|');
    var status = [screen, titleMode, noUndo, noRestart, noAction, noHistory, state.fgcolor, state.bgcolor, title, unavailable].join('|');
    if (Mobile.playerStatus === status) { return; }
    if (Mobile.playerScreen !== screen) { Mobile.releasePlayerButtons(); }
    Mobile.playerScreen = screen;
    Mobile.playerStatus = status;
    root.style.setProperty('--player-fg', state.fgcolor);
    root.style.setProperty('--player-bg', state.bgcolor);
    document.getElementById('gametitle').title = title;
    if (IDE) { document.getElementById('gametitle').textContent = title; }
    var availabilityChanged = root.classList.contains('controls-unavailable') !== unavailable;
    root.classList.toggle('controls-unavailable', unavailable);
    if (availabilityChanged) { requestAnimationFrame(function () { canvasResize(); }); }
    document.getElementById('backToTitleButton').disabled = titleScreen;
    var undo = document.getElementById('undoButton');
    var restart = document.getElementById('restartButton');
    var action = document.getElementById('actionButton');
    undo.hidden = noUndo;
    restart.hidden = noRestart;
    action.hidden = noAction;
    undo.disabled = unavailable || textMode || noHistory;
    restart.disabled = unavailable || textMode;
    action.disabled = unavailable;
    var actionLabel = textMode ? (titleScreen ? 'Start' : 'Continue') : 'Action';
    action.querySelector('span').textContent = actionLabel;
    action.setAttribute('aria-label', actionLabel + ' (X)');
    action.title = actionLabel + ' (X)';
    var actionLayoutChanged = action.parentNode.classList.contains('no-action') !== noAction;
    action.parentNode.classList.toggle('no-action', noAction);
    if (actionLayoutChanged && root.classList.contains('controls-stacked')) {
        requestAnimationFrame(function () { canvasResize(); });
    }
    ['up','left','down','right'].forEach(function (direction) {
        document.getElementById('move' + direction).disabled = unavailable || textMode &&
            !(titleScreen && titleMode === 1 && (direction === 'up' || direction === 'down'));
    });
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
        action: 88, // x
        left: 37, // left arrow
        right: 39, // right arrow
        up: 38, // up arrow
        down: 40, // down arrow
        undo: 85, // u
        restart: 82, // r
        quit: 27 // escape
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
        window.addEventListener('touchcancel', this.cancelTouch.bind(this));
    };

    proto.bootstrap = function () {
        if (Mobile.playerToolbar) {
            Mobile.playerToolbar.classList.add('touch-enabled');
            Mobile.playerRoot.classList.add('touch-enabled');
            var links = document.getElementById('playerLinks');
            if (links) { document.querySelector('.player-menu-footer').appendChild(links); }
            Mobile.layoutPlayerControls();
            canvasResize();
        } else {
            this.showTab();
            this.disableScrolling();
        }
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
        if (Mobile.playerToolbar && !document.querySelector('.gameContainer').contains(event.target)) {
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
            if (event.touches.length === 0 && event.target.id !== "unMuteButton" && event.target.id !== "muteButton") {
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
        if (!this.isFocused || !this.isTouching) {
            return;
        }
        if (levelEditorOpened) {
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

    proto.cancelTouch = function () {
        this.isTouching = false;
        this.mayBeSwiping = false;
        this.endRepeatWatcher();
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
        if (Mobile.playerToolbar) {
            Mobile.setPlayerMenu(!Mobile.playerMenuOpen);
            return;
        }
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

        //the reason I'm adding all these empty click events is on safari to disable double-tap to zoom (also needs some other css settings. cf https://github.com/increpare/PuzzleScript/issues/599 SMGDH)
        this.tabAffordance.addEventListener('touchstart', openCallback);
        this.tabAffordance.addEventListener("click", event => { });
        this.tabElem.addEventListener('touchstart', openCallback);
        this.tabElem.addEventListener("click", event => { });

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
        this.closeAffordance.addEventListener("click", event => { });
        closeTab.addEventListener('touchstart', closeCallback);
        closeTab.addEventListener("click", event => { });

        undo = this.menuElem.getElementsByClassName('undo')[0];
        if (undo) {
            undo.addEventListener('touchstart', function (event) {
                event.stopPropagation();
                self.emitKeydown('undo');
            });
            undo.addEventListener("click", event => { });
        }
        restart = this.menuElem.getElementsByClassName('restart')[0];
        if (restart) {
            restart.addEventListener('touchstart', function (event) {
                event.stopPropagation();
                self.emitKeydown('restart');
            });
            restart.addEventListener("click", event => { });
        }

        quit = this.menuElem.getElementsByClassName('quit')[0];
        quit.addEventListener('touchstart', function (event) {
            event.stopPropagation();
            self.emitKeydown('quit');
        });
        quit.addEventListener("click", event => { });

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
            opacityString = "display:none;"
        } else {
            this.closeAffordance.setAttribute('style', 'display: block;');
        }

        this.closeElem.setAttribute('style', style);

        this.menuElem.setAttribute('style', opacityString);
    };

    proto.disableScrolling = function () {
        var style = {
            height: "100%",
            overflow: "hidden",
            position: "fixed",
            width: "100%",
            margin: 0
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
        window.playSeed = function () { };
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
        body = Mobile.playerRoot || document.body;
        body.classList.add('disable-select');
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

    function tickUp() {
        var isFinished;
        ratio += increment;
        if (ratio >= 1.0) {
            isFinished = true;
            ratio = 1;
        }
        update(ratio);
        return isFinished;
    };

    function tickDown() {
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

(function () {
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
        window.requestAnimationFrame = function (callback, element) {
            var currTime, timeToCall, id;

            currTime = new Date().getTime();
            timeToCall = Math.max(0, 16 - (currTime - lastTime));
            id = window.setTimeout(function () {
                callback(currTime + timeToCall);
            }, timeToCall);
            lastTime = currTime + timeToCall;

            return id;
        };
    }

    if (!window.cancelAnimationFrame) {
        window.cancelAnimationFrame = function (id) {
            clearTimeout(id);
        };
    }

    // The editor initializes after its engine, code editor and layout are ready.
    if (!IDE) {
        Mobile.setupPlayerControls();
        Mobile.enable();
    }
}());
