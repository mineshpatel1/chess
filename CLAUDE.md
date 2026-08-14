# Comments

Comments are for code that cannot explain itself. Prefer making the code self-explanatory —
clearer names, smaller functions, a better structure — over explaining code that isn't.

- Keep them short. A line or two, not a paragraph.
- No war stories. No "this used to be broken", no "caught in flight", no history of a bugfix,
  no bug numbers or dates. Git has that.
- No measurement logs or tuning tables in prose. If a constant was measured, say what it is and
  move on.
- Say *why*, not *what*. A comment restating the code is noise.
- Docstrings say what a thing does and what its arguments mean. One or two sentences is usually
  enough.
