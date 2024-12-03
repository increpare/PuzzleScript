function saveAs(text, type, filename) {
    var element = document.createElement('a');
    //encode text as blob
    var blob = new Blob([text], {type: type});
    var url = URL.createObjectURL(blob);
    element.setAttribute('href', url);
    element.setAttribute('download', filename);

    element.style.display = 'none';
    document.body.appendChild(element);

    element.click();

    document.body.removeChild(element);
}