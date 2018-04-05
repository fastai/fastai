# fastai doc project

The fastai doc project is just getting underway! So now is a great time to get involved. Here are some thoughts and guidelines to help you get oriented...

## Project goals and approach

The idea of this project is to create documentation that makes readers say "wow that's the most fantastic documentation I've ever read". So... no pressure. :) How do we do this? By taking the philosophies demonstrated in fast.ai's courses and bringing them to the world of documentation. Here are a few guidelines to consider:

- Assume the reader is intelligent and interested
- Don't assume the reader has any specific knowledge about the field you're documenting
- If you need the reader to have some knowledge to understand your documentation, and there is some effective external resource they can learn from, point them there rather than trying to do it all yourself
- Use code to describe what's going on where possible, not math
- Create a notebook demonstrating the ideas you're documenting (include the notebook in this repo) and show examples from the notebook directly in your docs
- Use a top-down approach; that is, first explain what problem the code is meant to solve, and at a high level how it solves it, and then go deeper into the details once those concepts are well understood
- For common tasks, show full end-to-end examples of how to complete the task.

Use pictures, tables, analogies, and other explanatory devices (even embedded video!) wherever they can help the reader understand. Use hyperlinks liberally, both within these docs and to external resources. 

We don't want this detailed documentation to create clutter in the code, and we also don't want to overwhelm the user when they just want a quick summary of what a method does. Therefore, docstrings should generally be limited to a single line. The python standard library is documented this way--for instance, the docstring for `re.compile()` is the single line "*Compile a regular expression pattern, returning a pattern object.*" But the full documentation of the `re` library on the python web site goes into detail about this method, how it's used, and its relation to other parts of the library.

