# Continue this project in a new session

Goal: beginner-friendly MSc Generative AI course, 12 weeks, each with >=25 slides and lecturer notes, a study guide with references and exam practice, and a guided Python notebook. Current decks have 40 slides each. The current syllabus is in the lesson files and docs/SYLLABUS_ALIGNMENT.md.

Read README.md and docs/APPLICATION_GUIDE.md. Inspect the relevant content/week_XX files, previews and review notes before editing. Keep technical block diagrams exact and explanations concrete. Use supplied sources as references requiring verification. Retain advanced MCP/multi-agent content as optional unless the syllabus is deliberately revised.

Run `python studio.py status`. Generators are course.py, src/slides.py and src/handouts.py. The optional provider adapter is src/providers.py, configured through config/providers.json and environment variables. Credentials must not be embedded in content or committed. No live API integration has been verified using user credentials yet.

Make a specific improvement, rebuild only the relevant weeks, inspect the resulting visual previews, execute meaningful tests, and record evidence. The authoring CLI stages AI drafts; it does not automatically replace sources. Inspect any model-written code before executing it. Keep version history and export a new package using package_project.py.

The current scope is a single-operator CLI with resumable file jobs. There is no deployed web app, scheduled unattended worker, autonomous infinite research loop or distributed backend. Add those only when explicitly requested.
