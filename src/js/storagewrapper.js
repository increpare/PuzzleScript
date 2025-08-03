'use strict';

function storage_has(key) {
    return localStorage.getItem(key) !== null;
}

function storage_get(key) {
    return localStorage.getItem(key);
}

function storage_get_int(key, defaultValue) {
    const storage_value = parseInt(localStorage.getItem(key), 10);
    // isNaN(parseInt(null)) is true, so
    return isNaN(storage_value) ? defaultValue : storage_value;
}

function storage_set(key, value) {
    return localStorage.setItem(key, value);
}

function storage_remove(key) {
    localStorage.removeItem(key);
}