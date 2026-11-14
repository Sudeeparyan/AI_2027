# R0: platform facts for the teaching-pack upgrade

Checked: **2026-10-10**. Scope: official documentation and published runtime inventories.
This is research for E2; no curriculum or generated teaching files have been changed.
Published availability is separate from access measured in a student's account.

## Gemini Developer API

### Model IDs and free usage

These are teaching-relevant **stable text-output models**, not the entire media-model catalogue.
Their standard input/output tokens are listed as free, subject to eligibility and quota.

| Model ID | Catalogue status | Standard free input/output |
|---|---|---|
| `gemini-3.5-flash-lite` | Stable | Yes |
| `gemini-3.8-flash` | Stable | Yes |
| `gemini-3.6-flash` | Stable | Yes |
| `gemini-3.1-flash-lite` | Stable | Yes |

Sources: [model catalogue](https://ai.google.dev/gemini-api/docs/models) and
[pricing](https://ai.google.dev/gemini-api/docs/pricing), checked 2026-10-10.
The catalogue restricts 2.5-family access to previous users and directs new projects to newer models.
For E2, propose `GEMINI_MODEL = "gemini-3.5-flash-lite"` as one explicit constant, rather than a moving
`latest` alias. This is a design recommendation, not a live availability or performance result.
Its [model page](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite) lists text output
and text/image/audio/video/PDF inputs. It does not generate images or audio.
The pricing page lists `gemini-embedding-2` with free standard inputs, while the model catalogue
lists `gemini-embedding-2-preview`. Record this source discrepancy; confirm the endpoint before any
optional API embedding exercise. The RAG core should use local embeddings. Free text generation
does not imply free access to every tool or media model.

### Quotas and fallback

Google's [rate-limit page](https://ai.google.dev/gemini-api/docs/rate-limits), updated 2026-10-09,
directs users to their active limits in [AI Studio](https://aistudio.google.com/).
It supplies no universal numeric free-tier RPM/TPM/RPD table. Limits depend on model, project tier
and account status; capacity is not guaranteed. Limits apply per project, not per key. Daily limits
reset at midnight Pacific time. RPM means requests per minute; TPM means input tokens per minute;
RPD means requests per day.

E2 recommendation: use a small planned call count, never promise that the course's approximately
40-call ceiling is a student's entitlement, and count retry attempts against the budget. Retry 429
with bounded exponential backoff; a daily exhausted quota will not be fixed by rapid retries.
Return `None` after exhaustion so the local exercise completes. Cache ordinary synthetic prompt
responses on disk; avoid grounded-search calls, whose storage rules differ. No numeric account
quota was inspected in this session.

### Regions, account requirements and data use

[Supported regions](https://ai.google.dev/gemini-api/docs/available-regions) include Ireland, the
United Kingdom, Switzerland and EU countries. Colab requests are also affected by the runtime's
region. Users must be 18 or older; age verification and
[Workspace administrator settings](https://ai.google.dev/gemini-api/docs/workspace) can affect access.
[Billing guidance](https://ai.google.dev/gemini-api/docs/billing) describes new accounts starting on
the free tier, but does not establish free eligibility for every regional/student account.

The [additional terms](https://ai.google.dev/gemini-api/terms), effective 2026-03-23, distinguish:

- Generally, unpaid inputs/outputs may support product improvement and human review.
- For EEA/Swiss/UK users, paid-service data-use provisions also cover unpaid quota: no product
  improvement from prompts/responses; limited safety/legal logging remains.
- Separately, providing API Clients to EEA/Swiss/UK users requires Paid Services. API access is
  classified as paid only through a project with active billing.

**Course implication (inference):** these pages do not expressly resolve the status of distributed
notebooks using students' own keys. Do not promise a free Gemini course path in Ireland/UK or enable
billing to work around this. Retain the complete local route; flag the regional issue before E2
publishes opt-in instructions. Keep the project's synthetic-data-only rule in every region.

### SDK and key handling

Use the `google-genai` package (`from google import genai`). Google's
[current getting-started guide](https://ai.google.dev/gemini-api/docs/get-started) now demonstrates
`client.interactions.create(...)`. The
[generate-content API reference](https://ai.google.dev/api/generate-content) and
[official Python SDK](https://github.com/googleapis/python-genai) still document
`client.models.generate_content(...)`, which is sufficient for the planned stateless lab helper.
Select and test the SDK version in E2; these examples are documentation patterns, not executed calls.

The [official authentication notebook](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb)
uses Colab's Secrets sidebar: create `GEMINI_API_KEY`, enter its value, enable notebook access, then
read `userdata.get("GEMINI_API_KEY")`. Locally read `os.environ.get("GEMINI_API_KEY")`. Pass the value
explicitly to `genai.Client(api_key=api_key)`; the SDK otherwise gives `GOOGLE_API_KEY` precedence
when both environment names are set. Never print a key or put it in a saved code cell.

```python
# Documentation pattern for E2, after opt-in and a successful secret lookup.
from google import genai
from google.genai import types

GEMINI_MODEL = "gemini-3.5-flash-lite"
with genai.Client(api_key=api_key) as client:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents="Explain a probability using a fair six-sided die.",
        config=types.GenerateContentConfig(max_output_tokens=256),
    )
    answer_text = response.text
```

For E2, retrieve keys only when `USE_GEMINI` is true; missing, denied or unavailable Secrets must
produce a helpful message and `None`, without a call. Catch `google.genai.errors.APIError` and inspect
its numeric `code` for 429, as documented in the SDK's error-handling section. Limit retries and
request size. No live request, billing change or key access was made in R0.

## Widget support

Use standard `ipywidgets.FloatSlider`/`IntSlider`, with a numeric-variable fallback so the exercise
also runs when the UI is unavailable. Jupyter's
[installation guide](https://ipywidgets.readthedocs.io/en/stable/user_install.html) describes installing
`ipywidgets` in the kernel environment. The
[interaction guide](https://ipywidgets.readthedocs.io/en/stable/examples/Using%20Interact.html) supports
`continuous_update=False` and `interact_manual` for expensive computations.

Google's [widget-manager source](https://github.com/googlecolab/colabtools/blob/main/google/colab/output/_widgets.py)
provides `output.enable_custom_widget_manager()` for custom/third-party widgets. Use ordinary sliders
first; enabling a custom manager is not a universal fix for a missing kernel package.

In VS Code, install Microsoft's Python and Jupyter extensions, select the environment containing
`ipywidgets`, and run the notebook there; see
[notebook setup](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).
Microsoft's [widget support guidance](https://github.com/microsoft/vscode-jupyter/wiki/IPyWidget-Support-in-VS-Code-Python)
notes that supporting scripts can require CDN access via `jupyter.widgetScriptSources`; campus
network restrictions can prevent rendering. That wiki page is dated 2022, checked here in 2026;
it is troubleshooting guidance, not a live rendering test.

E2/week-lab recommendations: tag widget cells `skip-test`, keep training and API calls out of slider
callbacks, and let sliders recompute small local examples or display stored results. Verify actual
rendering later in both Colab and VS Code.

## Colab runtime facts

### Free-tier limits

The [official FAQ](https://research.google.com/colaboratory/faq.html) publishes no fixed free CPU/GPU
quota, GPU model, memory size or idle timeout. GPU access is restricted and varies with availability
and usage. Free notebooks may run for **at most 12 hours**, subject to those conditions; idle runtimes
terminate. Free VMs use the standard system-memory profile. Twelve hours is a ceiling, not a reserved
session length or a CPU performance promise.

Course decision: the required path must work on CPU; an optional GPU cannot be a prerequisite.
Save work and keep expensive training as a stretch. The 15-minute core waiting budget is the
project's requirement, not a Google service guarantee; it still needs a measured run.

### Published preinstalled package snapshots

Google's [backend inventory](https://github.com/googlecolab/backend-info), linked from the FAQ,
publishes CPU and GPU package lists. These are **published snapshots**, not measurements of a fresh
student runtime. The latest CPU-inventory update found on or before the check date is commit
`92a2364a31cd9b686f88075e52d0ff5e5c098a40`, dated 2026-10-02.

| Package | Published CPU, 2026-10-02 | Published GPU, 2026-10-02 | Pinned `2026.07` CPU / GPU |
|---|---|---|---|
| torch | `2.11.0+cpu` | `2.11.0+cu130` | `2.11.0+cpu` / `2.11.0+cu128` |
| transformers | `5.18.0` | `5.18.0` | `5.13.1` |
| diffusers | `0.40.0` | `0.40.0` | `0.39.0` |
| peft | `0.21.1` | `0.21.1` | `0.19.1` |
| trl | Not listed | Not listed | Not listed |
| sentence-transformers | `5.7.0` | `5.7.0` | `5.6.0` |
| ipywidgets | `7.7.1` | `7.7.1` | `7.7.1` |

Latest sources: [CPU freeze](https://raw.githubusercontent.com/googlecolab/backend-info/92a2364a31cd9b686f88075e52d0ff5e5c098a40/pip-freeze.txt),
[GPU freeze](https://raw.githubusercontent.com/googlecolab/backend-info/92a2364a31cd9b686f88075e52d0ff5e5c098a40/pip-freeze.gpu.txt),
[dated commit](https://github.com/googlecolab/backend-info/commit/92a2364a31cd9b686f88075e52d0ff5e5c098a40).
CPU torch values are read from the wheel filename in each freeze's `torch @ ...` entry.

Pinned-runtime sources: [CPU freeze](https://raw.githubusercontent.com/googlecolab/backend-info/ec53f93fe44a9cab118dc5451527b18e36e1a643/pip-freeze.txt),
[GPU freeze](https://raw.githubusercontent.com/googlecolab/backend-info/ec53f93fe44a9cab118dc5451527b18e36e1a643/pip-freeze.gpu.txt),
[dated commit](https://github.com/googlecolab/backend-info/commit/ec53f93fe44a9cab118dc5451527b18e36e1a643)
(2026-07-14), and the [runtime-version FAQ](https://research.google.com/colaboratory/runtime-version-faq.html).

The inventory README warns that production can be one or two days ahead; default environments change.
`trl` being absent means it is not listed in these freezes; no preinstalled version can be claimed.
E2 must install and validate a compatible version where needed. Published versions are evidence for
setup planning, not a tested dependency lock for the existing labs.

Run this **before installing packages** in a fresh Colab CPU runtime; save its output with the future
run record. Local execution does not establish Colab's package versions.

```python
# Record the environment actually allocated to the student.
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import platform

print("Captured:", datetime.now(timezone.utc).isoformat())
print("Python:", platform.python_version())
for package in (
    "torch", "transformers", "diffusers", "peft", "trl",
    "sentence-transformers", "ipywidgets",
):
    try:
        print(package, version(package))
    except PackageNotFoundError:
        print(package, "not installed")
```

## Verification and handoff

### E2 package supplement (checked 2026-10-10)

- The publisher's PyPI releases identify [google-genai 2.29.0](https://pypi.org/project/google-genai/2.29.0/)
  (Python ≥3.10) and [TRL 1.15.0](https://pypi.org/project/trl/1.15.0/) (Python ≥3.11).
  These are explicit installation pins, not claims that Colab preinstalls either package.
- E2 installed google-genai 2.29.0 and ipywidgets 7.7.1 locally. SDK configuration construction
  and a standard slider object succeeded; this does not verify live API access or browser widget rendering.
- The same pinned official CPU freeze lists `torchvision` 0.26.0+cpu alongside torch 2.11.0+cpu.
  Install cells pin torchvision 0.26.0 too, avoiding an unbounded upgrade of the paired torch package.
- Regional API-client distribution is unresolved by the examples alone. Notebook guidance links
  the terms, keeps the opt-in off, and provides complete local tasks without billing.

- Official pages checked on 2026-10-10; sources linked beside each finding.
- PASS in this session: 28 package entries cross-checked against four pinned official inventories;
  both Python examples parsed successfully; all R0 topics and snapshot/live distinctions present.
- Independent source review found no blocking issue. Its embedding-ID discrepancy was checked
  against both Google pages and recorded above.
- Unverified: live Colab package inventory/time, live Gemini quota/access, and widget rendering.
- Before E2: validate package compatibility, preserve CPU/local completion, and resolve the regional
  free-API distribution question without silently enabling billing.

### Teacher-authorised OpenAI alternative (checked 2026-10-11)

- The teacher permits an optional OpenAI API route alongside Gemini. Both switches default to false;
  exactly one can be enabled through `ask_model()`. Complete local tasks require no key or API spending.
- The official [GPT-4.1 mini page](https://developers.openai.com/api/docs/models/gpt-4.1-mini)
  lists the Responses endpoint. The [Python quickstart](https://developers.openai.com/api/docs/quickstart)
  uses the official SDK and `response.output_text`; shared setup uses this interface.
- API-key usage has separate [API pricing](https://developers.openai.com/api/docs/pricing).
  [ChatGPT pricing](https://learn.chatgpt.com/docs/pricing) distinguishes subscription usage from API
  token pricing. The notebooks do not implement Sign in with ChatGPT plan access or promise free OpenAI calls.
- OpenAI requests set `store=False`. [Data controls](https://developers.openai.com/api/docs/guides/your-data)
  describe endpoint state and abuse-monitoring retention; this setting does not guarantee zero retention.
- Publisher release metadata checked: [openai 3.28.0](https://pypi.org/project/openai/3.28.0/)
  and [python-dotenv 1.2.4](https://pypi.org/project/python-dotenv/1.2.4/). These are optional-install pins,
  not Colab preinstalled-version claims. The local lab environment installed both successfully.
- Keys are read only after opt-in: Colab Secrets, environment, then a selected local `.env` file.
  The private project `.env` is ignored and untracked; its contents were not inspected or copied.
- PASS observed: 22 network-free setup regressions cover selection, conflicting switches, Secrets,
  private `.env` access, shared attempt ceiling, cache, retries and local fallback. Live API access,
  account quotas and a timed Colab run remain unverified. No private-key API call was needed.


### Vendor and clean-install verification addendum (checked 2026-10-12)

Publisher release pages and their version-specific PyPI JSON metadata were rechecked on
**2026-10-12**. All 14 exact direct pins in `tools/build_week.py:LAB_PINS` have published releases;
this is a version-existence check, not a claim that each is the newest release.

| Notebook package | Verified exact pin and publisher release |
|---|---|
| torch | [2.11.0](https://pypi.org/project/torch/2.11.0/) |
| torchvision | [0.26.0](https://pypi.org/project/torchvision/0.26.0/) |
| transformers | [5.18.0](https://pypi.org/project/transformers/5.18.0/) |
| diffusers | [0.40.0](https://pypi.org/project/diffusers/0.40.0/) |
| peft | [0.21.1](https://pypi.org/project/peft/0.21.1/) |
| sentence-transformers | [5.7.0](https://pypi.org/project/sentence-transformers/5.7.0/) |
| ipywidgets | [7.7.1](https://pypi.org/project/ipywidgets/7.7.1/) |
| google-genai | [2.29.0](https://pypi.org/project/google-genai/2.29.0/) |
| trl | [1.15.0](https://pypi.org/project/trl/1.15.0/) |
| openai | [3.28.0](https://pypi.org/project/openai/3.28.0/) |
| python-dotenv | [1.2.4](https://pypi.org/project/python-dotenv/1.2.4/) |

The official CPU wheel indexes publish the selected Windows CPython 3.12 builds:
[torch 2.11.0+cpu](https://download.pytorch.org/whl/cpu/torch/) and
[torchvision 0.26.0+cpu](https://download.pytorch.org/whl/cpu/torchvision/).
Those exact build tags appear in the successful local torch installation log.

The deployment dependencies below are also exact direct pins, matching the versions
resolved in the clean installation. The coordinator added them to `LAB_PINS`; the final
rebuild applies these deployment pins to generated installation cells.

| Deployment package | Exact pin, observed clean-install version and publisher release |
|---|---|
| FastAPI | [0.143.0](https://pypi.org/project/fastapi/0.143.0/) |
| httpx | [0.28.1](https://pypi.org/project/httpx/0.28.1/) |
| uvicorn | [0.54.0](https://pypi.org/project/uvicorn/0.54.0/) |

**Local evidence:** the coordinator observed clean pip installation **PASS** and
`pip check` **PASS** in `build/qa_pinned_env` (Windows, CPython 3.12.10).
The [pip installation report](../../build/qa_pinned_install_core_report.json) records
resolved versions, download origins and hashes; installation logs are
`build/qa_pinned_install_torch.log` and `build/qa_pinned_install_core.log`.
Full CPU core reruns were still in progress when this addendum was written; installation
and dependency checks alone do not establish that every notebook executes successfully.

The dated [Gemini model catalogue](https://ai.google.dev/gemini-api/docs/models) and
[standard pricing](https://ai.google.dev/gemini-api/docs/pricing) still support the four
stable IDs recorded above. The [GPT-4.1 mini page](https://developers.openai.com/api/docs/models/gpt-4.1-mini)
lists Responses support; the [Responses creation reference](https://developers.openai.com/api/reference/python/resources/responses/methods/create)
specifies a minimum `max_output_tokens` of 16. These are documentation checks, not live calls.

**Still unverified:** live Colab installation, allocated package inventory and timing;
live provider access, quotas and optional routes; browser widget rendering; and the
EEA/Swiss/UK Gemini client-distribution condition described above. No API key was read,
no hosted request was made, and no billing was enabled for this verification.

### Student installation retry — checked 13 October 2026

Ten existing direct dependencies were still floating in student installation cells. Their exact versions already exist in the tested clean CPU environment. Each version-specific publisher PyPI release page was checked on 13 October; the builder now freezes these versions alongside the previous fourteen pins. Week 4 also installs torchvision for its optional MNIST section.

| Package | Tested exact version and checked release page |
|---|---|
| accelerate | [1.15.0](https://pypi.org/project/accelerate/1.15.0/) |
| datasets | [5.1.0](https://pypi.org/project/datasets/5.1.0/) |
| matplotlib | [3.11.2](https://pypi.org/project/matplotlib/3.11.2/) |
| numpy | [2.5.2](https://pypi.org/project/numpy/2.5.2/) |
| pandas | [3.0.6](https://pypi.org/project/pandas/3.0.6/) |
| pydantic | [2.14.1](https://pypi.org/project/pydantic/2.14.1/) |
| rank-bm25 | [0.2.2](https://pypi.org/project/rank-bm25/0.2.2/) |
| scikit-learn | [1.9.1](https://pypi.org/project/scikit-learn/1.9.1/) |
| scipy | [1.18.1](https://pypi.org/project/scipy/1.18.1/) |
| soundfile | [0.14.0](https://pypi.org/project/soundfile/0.14.0/) |

This freezes the tested direct packages; it does not freeze every transitive dependency or establish a timed Colab installation. Local dependency and notebook retry evidence is recorded separately in build/retry_20261013/.


### Runtime and widget retry observations — 13 October 2026

The pinned [NumPy 2.5.2 metadata](https://pypi.org/pypi/numpy/2.5.2/json) and [SciPy 1.18.1 metadata](https://pypi.org/pypi/scipy/1.18.1/json) require Python 3.12 or newer, also confirmed from the installed distributions. The notebook banner now specifies Python 3.12+. The local tested interpreter remains 3.12.10.

Native browser access opened the uploaded Week 1 solution notebook in Colab. Its visible release notes mention Python 3.13.5; this was not an allocated-runtime inventory. No Colab runtime was connected or timed because browser control was interrupted. Actual Week 1 widget code was executed locally in VS Code with the tested NumPy, matplotlib and ipywidgets versions, but its renderer requested third-party widget support files. No download-permission setting was changed. Students can use the numeric settings when a slider cannot display; all core paths remain available. UI observations are retained in build/retry_20261013/ui_observations.json.


### VS Code widget pilot follow-up — 13 October 2026

The current Week 1 temperature widget was rerun in VS Code with Python 3.12.10 and ipywidgets 7.7.1. Its visible control and chart updated at T=1.0, 1.7 and 0.4. Four exact distributed code cells used a five-name invented fixture. No permission setting was changed by the agent. This closes the local Week 1 interaction check; other frontend checks and timed Colab remain unverified. Evidence: build/retry_20261013/widget_ui_verified.json.
