# Prompt for the next revision

Use this project as the source of truth for my Generative AI teaching materials.
Read README.md and reviews/REVISION_LOG.md first. Inspect the latest files under
outputs/current and the matching content/week_XX source files.

My requested improvement is: [describe the week, slide IDs and change].

Keep the material understandable for beginner MSc students. Explain new symbols
and vocabulary. Use a worked example, a useful picture or diagram, and a clear
speaker script. Preserve correct content and reuse existing assets when suitable.
Save new illustrations with versioned names and retain their prompts. Update the
weekly JSON and the corresponding handout/notebook when a teaching claim changes.

Run `python course.py improve --weeks [numbers] --max-passes 3`, inspect the new
gallery, and record the result in reviews/. Keep previous runs and editable
diagrams. Report what changed, what you checked and any remaining issues. Do not
claim that passing automatic layout checks establishes educational quality.
