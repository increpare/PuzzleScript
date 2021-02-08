(async function a() {
    const { Compress } = require('gzipper');

    var glob = require("glob")

    files = glob.sync("./bin/**/*.js");
    files = files.concat(glob.sync("./bin/**/*.html"));
    files = files.concat(glob.sync("./bin/**/*.css"));
    files = files.concat(glob.sync("./bin/**/*.txt"));

    var compressionTasks = files.map( fn=>new Compress(fn));
    var compressed = await Promise.all(compressionTasks.map(gzip=>gzip.run()));

    console.info('Compressed gzip files: ', gzipFiles);

})()
    // files is an array of filenames.
    // If the `nonull` option is set, and nothing
    // was found, then files is ["**/*.js"]
    // er is an error object or null.

// staticGzip =  require('http-static-gzip-regexp')
// app.use(staticGzip(/(\.html|\.js|\.css)$/));
