# Long request cases

Build a request with 20,000–30,000 input tokens from invented or shareable meeting
turns. Do not estimate token count from file size: models tokenize the same text
differently. Use your server's `/apply-template` and `/tokenize` endpoints with the
exact messages and chat template settings, then inspect `usage.prompt_tokens`
after running it.

Start with a request.example.json from a job folder. Extend its transcript or add
several dated meetings to its user message. Include distinct decisions near the
beginning, middle and end, and refer to those facts in the task. Avoid merely
repeating a paragraph: repetition tests a different failure mode. For cleanup,
extend the turns array with unique ids; retain a copy as the fidelity reference.

Keep maxTokens at the job's required output budget, up to 8192 for long outputs.
Set --n-ctx to measured input tokens plus that budget and a margin for the chat
template. A 30,000-token input with an 8192-token output does not fit in 32768.
Run --key-only before starting an arm, and compare models on identical request
bytes and sampling settings. A truncation remains a structural failure even if
the visible prefix looks correct.

Store large requests outside this public repository. Record their SHA-256 and
case labels with the results. Request-file records contain the full messages;
review them before sharing records or sending them to the optional judge.
