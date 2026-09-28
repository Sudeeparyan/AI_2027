# Teaching redesign

Each week now follows a 40-slide lesson with a concrete opening problem, five concept cycles, a worked lab bridge, a transfer case and retrieval practice. Each concept cycle asks students to predict, connects an everyday analogy to the technical idea, explains the actual mechanism, works an example and resolves the opening question.

`weeks.json` is the pedagogical source. `build_lessons.py` compiles it into editable `content/week_XX/slides.json` and a teaching companion. Run it only when intentionally replacing compiled slide content. The production build reads slides.json as usual. Manual slide edits therefore survive ordinary course.py builds.

Examples are synthetic but represent recognisable campus and workplace situations. They are not live services, measured company results or forecasts about 2027. Every analogy includes a limitation.
