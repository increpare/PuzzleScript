/* eslint-env jasmine */
import fs from 'fs'
import path from 'path'
import Parser from './parser/parser'

const GISTS_ROOT = path.join(__dirname, '../games/')
const GIST_SOLUTIONS_ROOT = path.join(__dirname, '../game-solutions/')

describe('parsing files unambiguously', () => {

    const gistDirs = fs.readdirSync(GISTS_ROOT)
    // it('checks all files that they parse uniquely', () => {
    gistDirs.forEach((gistDirName) => {
        // Only parse files that do not have solutions
        // because solutions will be tested by running the games
        if (!fs.existsSync(path.join(GIST_SOLUTIONS_ROOT, `${gistDirName}.json`))) {
            const codeFile = path.join(GISTS_ROOT, gistDirName, 'script.txt')
            if (fs.existsSync(codeFile)) {
                it(`parses ${gistDirName} uniquely`, () => {
                    const code = fs.readFileSync(codeFile, 'utf-8')
                    Parser.parse(code)
                })
            }
        }
    })
    // })

})
