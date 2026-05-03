'use strict';

let pluginOptimizationImpl = null;

function setPluginOptimizationHook(hook) {
    if (hook !== null && hook !== undefined && typeof hook !== 'function') {
        throw new Error('setPluginOptimizationHook expects a function or null.');
    }
    pluginOptimizationImpl = hook || null;
}

function pluginOptimizationHook(state) {
    if (typeof pluginOptimizationImpl === 'function') {
        pluginOptimizationImpl(state);
    }
}
