# AI Agent Constraints

## Semantic Understanding
- Do not implement receptionist understanding with regex or keyword-routing.
- Do not add `if utterance contains X` style branches for `name`, `affiliation`, or `purpose`.
- Use LLM prompts for semantic extraction and correction handling.
- Reducer/FSM logic may validate consistency, but it must not replace LLM understanding with handcrafted pattern matching.

## Allowed Deterministic Logic
- Canonical state updates
- Phase transitions
- Duplicate/stale result dropping
- Generic grounding checks such as normalized substring inclusion
- Structural validation such as rejecting duplicate slot values across different fields

## Disallowed Patterns
- Regex extraction for self-introduction names
- Regex extraction for purpose clauses
- Regex extraction for affiliation suffixes
- Scenario-specific lexical rescue logic

## Design Goal
- Keep the system AI-native:
  - `ASR -> LLM -> reducer/FSM -> TTS`
  - LLM performs meaning understanding
  - reducer/FSM enforces safety and state consistency
