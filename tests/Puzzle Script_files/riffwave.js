/*
 * RIFFWAVE.js v0.02 - Audio encoder for HTML5 <audio> elements.
 * Copyright (C) 2011 Pedro Ladaria <pedro.ladaria at Gmail dot com>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * version 2 as published by the Free Software Foundation.
 * The full license is available at http://www.gnu.org/licenses/gpl.html
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 *
 * Changelog:
 *
 * 0.01 - First release
 * 0.02 - New faster base64 encoding
 *
 */

var FastBase64_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=';
var FastBase64_encLookup = [];

function FastBase64_Init() {
  for (var i = 0; i < 4096; i++) {
    FastBase64_encLookup[i] = FastBase64_chars[i >> 6] + FastBase64_chars[i & 0x3F];
  }
}

function FastBase64_Encode(src) {
  var len = src.length;
  var dst = '';
  var i = 0;
  while (len > 2) {
    n = (src[i] << 16) | (src[i + 1] << 8) | src[i + 2];
    dst += FastBase64_encLookup[n >> 12] + FastBase64_encLookup[n & 0xFFF];
    len -= 3;
    i += 3;
  }
  if (len > 0) {
    var n1 = (src[i] & 0xFC) >> 2;
    var n2 = (src[i] & 0x03) << 4;
    if (len > 1) n2 |= (src[++i] & 0xF0) >> 4;
    dst += FastBase64_chars[n1];
    dst += FastBase64_chars[n2];
    if (len == 2) {
      var n3 = (src[i++] & 0x0F) << 2;
      n3 |= (src[i] & 0xC0) >> 6;
      dst += FastBase64_chars[n3];
    }
    if (len == 1) dst += '=';
    dst += '=';
  }
  return dst;
} // end Encode

FastBase64_Init();

function u32ToArray(i) { return [i & 0xFF, (i >> 8) & 0xFF, (i >> 16) & 0xFF, (i >> 24) & 0xFF]; }

function u16ToArray(i) { return [i & 0xFF, (i >> 8) & 0xFF]; }

function MakeRiff ( sampleRate, bitsPerSample,data) {
  var dat = [];
  var wav=[];
  var dataURI=[];

  var header = {                         // OFFS SIZE NOTES
    chunkId: [0x52, 0x49, 0x46, 0x46], // 0    4    "RIFF" = 0x52494646
    chunkSize: 0,                     // 4    4    36+SubChunk2Size = 4+(8+SubChunk1Size)+(8+SubChunk2Size)
    format: [0x57, 0x41, 0x56, 0x45], // 8    4    "WAVE" = 0x57415645
    subChunk1Id: [0x66, 0x6d, 0x74, 0x20], // 12   4    "fmt " = 0x666d7420
    subChunk1Size: 16,                    // 16   4    16 for PCM
    audioFormat: 1,                     // 20   2    PCM = 1
    numChannels: 1,                     // 22   2    Mono = 1, Stereo = 2, etc.
    sampleRate: sampleRate,                  // 24   4    8000, 44100, etc
    byteRate: 0,                     // 28   4    SampleRate*NumChannels*BitsPerSample/8
    blockAlign: 0,                     // 32   2    NumChannels*BitsPerSample/8
    bitsPerSample: bitsPerSample,                     // 34   2    8 bits = 8, 16 bits = 16, etc...
    subChunk2Id: [0x64, 0x61, 0x74, 0x61], // 36   4    "data" = 0x64617461
    subChunk2Size: 0                      // 40   4    data size = NumSamples*NumChannels*BitsPerSample/8
  };

  header.byteRate = (header.sampleRate * header.numChannels * header.bitsPerSample) >> 3;
  header.blockAlign = (header.numChannels * header.bitsPerSample) >> 3;
  header.subChunk2Size = data.length;
  header.chunkSize = 36 + header.subChunk2Size;

  wav = header.chunkId.concat(
      u32ToArray(header.chunkSize),
      header.format,
      header.subChunk1Id,
      u32ToArray(header.subChunk1Size),
      u16ToArray(header.audioFormat),
      u16ToArray(header.numChannels),
      u32ToArray(header.sampleRate),
      u32ToArray(header.byteRate),
      u16ToArray(header.blockAlign),
      u16ToArray(header.bitsPerSample),
      header.subChunk2Id,
      u32ToArray(header.subChunk2Size),
      data
    );
    
    dataURI = 'data:audio/wav;base64,' + FastBase64_Encode(wav);

    var result = {
      dat:dat,
      wav:wav,
      header:header,
      dataURI:dataURI
    };

    return result;
}


if (typeof exports != 'undefined')  // For node.js
  exports.RIFFWAVE = RIFFWAVE;
