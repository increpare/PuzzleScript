require('dotenv').config()
const Koa = require('koa')
const _ = require('koa-route')
const static = require('koa-static')
const bodyParser = require('koa-bodyparser')
const octokit = require('@octokit/rest')()

// Load the GitHub token from the environment
const GITHUB_TOKEN = process.env['GITHUB_TOKEN']

const app = new Koa()
octokit.authenticate({type: 'token', token: GITHUB_TOKEN})

// Parse the body when a game is posted to "/save" (the "Share" link is clicked)
app.use(bodyParser())
// Serve static files from the main directory
app.use(static('./'))

// Create a new Gist
app.use(_.post('/save', async (ctx) => {
  // Create the gist on gist.github.com
  const {data: gist} = await octokit.gists.create(ctx.request.body)
  ctx.body = gist
}))

// Start up the webserver
app.listen(process.env['PORT'] || 3000)
