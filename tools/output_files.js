// Generate beside the destination, then replace it without truncating the old file.
"use strict";
const fs = require("node:fs/promises");
const path = require("node:path");
const { randomUUID } = require("node:crypto");

const SHARING_ERRORS = new Set(["UNKNOWN", "EBUSY", "EACCES", "EPERM"]);
const RETRY_DELAYS = [100, 200, 400, 800];
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

async function atomicOutput(outPath, writeTemp, { rename = fs.rename, wait = delay } = {}) {
  const destination = path.resolve(outPath);
  const extension = path.extname(destination);
  // PptxGenJS recognises the suffix, so .pptx must remain the actual extension.
  const temporary = path.join(path.dirname(destination),
    `.${path.basename(destination, extension)}.${process.pid}.${randomUUID()}.tmp${extension}`);
  try {
    await writeTemp(temporary);
    for (let attempt = 0; ; attempt++) {
      try {
        await rename(temporary, destination);
        return;
      } catch (error) {
        if (!SHARING_ERRORS.has(error.code) || attempt >= RETRY_DELAYS.length) throw error;
        await wait(RETRY_DELAYS[attempt]);
      }
    }
  } catch (error) {
    // Only our unique temporary file is removed; never delete the destination.
    try {
      await fs.unlink(temporary);
    } catch (cleanupError) {
      if (cleanupError.code !== "ENOENT") {
        throw new AggregateError([error, cleanupError],
          `Output failed and temporary cleanup failed: ${temporary}`, { cause: error });
      }
    }
    throw error;
  }
}

module.exports = { atomicOutput };
