function storage_has(key){
    return localStorage.getItem(document.URL+key)!==null;
}

function storage_get(key){
    return localStorage.getItem(document.URL+key);
}

function storage_set(key,value){
    return localStorage.setItem(document.URL+key,value);
}

function storage_remove(key){
    localStorage.removeItem(document.URL+key);
}
