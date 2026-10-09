package llm

import _ "embed" // prompt and schema are versioned files compiled into the binary

// PromptVersion is recorded on every extraction so a regression can be traced
// to the exact prompt and schema that produced it. Changing either file means
// adding a new version, never editing one in place.
const PromptVersion = "invoice-v3"

//go:embed prompts/invoice-v3.txt
var systemPrompt string

//go:embed prompts/invoice-v3.schema.json
var responseSchema []byte
