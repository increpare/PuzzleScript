'use strict';

function saveAs(text, type, filename) {
    let element = document.createElement('a');
    //encode text as blob
    let blob = new Blob([text], {type: type});
    let url = URL.createObjectURL(blob);
    element.setAttribute('href', url);
    element.setAttribute('download', filename);

    element.style.display = 'none';
    document.body.appendChild(element);

    element.click();

    document.body.removeChild(element);
}