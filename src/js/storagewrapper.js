function storage_has(key){
    return localStorage.getItem(key)!==null;
}

function storage_get(key){
    return localStorage.getItem(key);
}

function storage_set(key,value){
    return localStorage.setItem(key,value);
}

function storage_remove(key){
    localStorage.removeItem(key);
}