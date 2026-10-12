// Offline filesystem checks: node tools/test_output_files.js
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const os = require("node:os");
const path = require("node:path");
const { atomicOutput } = require("./output_files");

async function inTemporaryDirectory(run) {
  const parent = path.resolve(os.tmpdir());
  const directory = await fs.mkdtemp(path.join(parent, "genai-node-output-"));
  try {
    await run(directory);
  } finally {
    assert.equal(path.dirname(directory), parent);
    assert.ok(path.basename(directory).startsWith("genai-node-output-"));
    await fs.rm(directory, { recursive: true, force: true });
  }
}

const failure = code => Object.assign(new Error(`simulated ${code}`), { code });

test("complete writes replace existing targets with unique adjacent paths and unchanged suffixes", async () => {
  await inTemporaryDirectory(async directory => {
    for (const extension of [".pptx", ".docx"]) {
      const destination = path.join(directory, `lecture.notes${extension}`);
      await fs.writeFile(destination, "old complete output");
      const temporaries = [];
      for (const content of ["first complete output", "second complete output"]) {
        await atomicOutput(destination, async temporary => {
          temporaries.push(temporary);
          assert.equal(path.dirname(temporary), directory);
          assert.equal(path.extname(temporary), extension);
          assert.notEqual(temporary, destination);
          await fs.writeFile(temporary, content);
        });
        assert.equal(await fs.readFile(destination, "utf8"), content);
      }
      assert.notEqual(temporaries[0], temporaries[1]);
      assert.deepEqual((await fs.readdir(directory)).filter(name => name.endsWith(extension)),
        [path.basename(destination)]);
    }
  });
});

test("failed generation leaves the old target intact and removes partial temporary output", async () => {
  await inTemporaryDirectory(async directory => {
    const destination = path.join(directory, "lecture.pptx");
    await fs.writeFile(destination, "old complete output");
    const error = failure("WRITE_FAILED");
    await assert.rejects(atomicOutput(destination, async temporary => {
      await fs.writeFile(temporary, "partial output");
      throw error;
    }), actual => actual === error);
    assert.equal(await fs.readFile(destination, "utf8"), "old complete output");
    assert.deepEqual(await fs.readdir(directory), ["lecture.pptx"]);
  });
});

test("transient sharing failures retry and preserve the target until replacement succeeds", async () => {
  await inTemporaryDirectory(async directory => {
    for (const code of ["UNKNOWN", "EBUSY", "EACCES", "EPERM"]) {
      const destination = path.join(directory, `${code}.docx`);
      await fs.writeFile(destination, "old complete output");
      const waits = [];
      let attempts = 0;
      await atomicOutput(destination, temporary => fs.writeFile(temporary, "new complete output"), {
        rename: async (temporary, target) => {
          attempts++;
          assert.equal(await fs.readFile(target, "utf8"), "old complete output");
          if (attempts === 1) throw failure(code);
          await fs.rename(temporary, target);
        },
        wait: async ms => { waits.push(ms); },
      });
      assert.equal(attempts, 2);
      assert.deepEqual(waits, [100]);
      assert.equal(await fs.readFile(destination, "utf8"), "new complete output");
    }
    assert.equal((await fs.readdir(directory)).length, 4);
  });
});

test("exhausted sharing retries keep the old output and clean the temporary file", async () => {
  await inTemporaryDirectory(async directory => {
    const destination = path.join(directory, "lecture.docx");
    await fs.writeFile(destination, "old complete output");
    const error = failure("UNKNOWN"), waits = [];
    let attempts = 0;
    await assert.rejects(atomicOutput(destination, temporary => fs.writeFile(temporary, "new output"), {
      rename: async () => { attempts++; throw error; },
      wait: async ms => { waits.push(ms); },
    }), actual => actual === error);
    assert.equal(attempts, 5);
    assert.deepEqual(waits, [100, 200, 400, 800]);
    assert.equal(await fs.readFile(destination, "utf8"), "old complete output");
    assert.deepEqual(await fs.readdir(directory), ["lecture.docx"]);
  });
});

test("permanent rename errors fail without retry or target deletion", async () => {
  await inTemporaryDirectory(async directory => {
    const destination = path.join(directory, "lecture.pptx");
    await fs.writeFile(destination, "old complete output");
    const error = failure("EINVAL");
    let attempts = 0, waits = 0;
    await assert.rejects(atomicOutput(destination, temporary => fs.writeFile(temporary, "new output"), {
      rename: async () => { attempts++; throw error; },
      wait: async () => { waits++; },
    }), actual => actual === error);
    assert.equal(attempts, 1);
    assert.equal(waits, 0);
    assert.equal(await fs.readFile(destination, "utf8"), "old complete output");
    assert.deepEqual(await fs.readdir(directory), ["lecture.pptx"]);
  });
});
