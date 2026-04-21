'use strict';

function normalizeInputToken(input) {
    return input;
}

function captureTraceSnapshot(phase, inputIndex, inputToken, substepIndex, previousSoundCount) {
    const currentLevelTarget =
        typeof curlevelTarget === 'number'
            ? curlevelTarget
            : (curlevelTarget === null ? null : curlevelTarget);

    return {
        phase,
        input_index: inputIndex,
        input: normalizeInputToken(inputToken),
        substep_index: substepIndex,
        current_level_index: typeof curlevel === 'number' ? curlevel : 0,
        current_level_target: currentLevelTarget,
        title_screen: Boolean(titleScreen),
        text_mode: Boolean(textMode),
        title_mode: typeof titleMode === 'number' ? titleMode : 0,
        title_selection: typeof titleSelection === 'number' ? titleSelection : 0,
        title_selected: Boolean(titleSelected),
        message_selected: Boolean(messageselected),
        winning: Boolean(winning),
        againing: Boolean(againing),
        loaded_level_seed: typeof loadedLevelSeed === 'string' ? loadedLevelSeed : null,
        serialized_level: typeof convertLevelToString === 'function' ? convertLevelToString() : '',
        command_queue: level && Array.isArray(level.commandQueue) ? level.commandQueue.slice() : [],
        command_queue_source_rules: level && Array.isArray(level.commandQueueSourceRules) ? level.commandQueueSourceRules.slice() : [],
        sound_history_length: Array.isArray(soundHistory) ? soundHistory.length : 0,
        new_sounds: Array.isArray(soundHistory) ? soundHistory.slice(previousSoundCount) : [],
    };
}

function executeInputToken(inputToken) {
    if (inputToken === 'undo') {
        DoUndo(false, true);
    } else if (inputToken === 'restart') {
        DoRestart();
    } else if (inputToken === 'tick') {
        processInput(-1);
    } else {
        processInput(inputToken);
    }
}

function runInputTrace(inputTokens) {
    const snapshots = [];
    let previousSoundCount = 0;

    const pushSnapshot = (phase, inputIndex, inputToken, substepIndex) => {
        const snapshot = captureTraceSnapshot(phase, inputIndex, inputToken, substepIndex, previousSoundCount);
        previousSoundCount = snapshot.sound_history_length;
        snapshots.push(snapshot);
    };

    pushSnapshot('initial', null, null, 0);

    for (let inputIndex = 0; inputIndex < inputTokens.length; inputIndex++) {
        const inputToken = inputTokens[inputIndex];
        executeInputToken(inputToken);
        pushSnapshot('input', inputIndex, inputToken, 0);

        let againSubstep = 0;
        while (againing) {
            againing = false;
            processInput(-1);
            againSubstep += 1;
            pushSnapshot('again', inputIndex, inputToken, againSubstep);
        }
    }

    return {
        input_count: inputTokens.length,
        snapshot_count: snapshots.length,
        snapshots,
    };
}

module.exports = {
    captureTraceSnapshot,
    runInputTrace,
};
