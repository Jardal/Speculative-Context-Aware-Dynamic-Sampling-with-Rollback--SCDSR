Scriptwriting and Tool Calls with Dynamic Parametric Creativity inside Inference

# Speculative Context-Aware Dynamic Sampling & Rollback (SCDSR)

A design pattern and harness specification for dynamic LLM sampling parameter manipulation (`temperature`, `top_p`, `n_predict`) coupled with **state-triggered context rollback**.

---

## 1. Executive Summary

Standard Large Language Model (LLM) generation uses static sampling parameters throughout a sequence. However, complex generation tasks require conflicting characteristics:
- **Creative/Exploratory phases** demand higher entropy (`temperature` > 0.8) for novelty and variance.
- **Structured/Contractual phases** require low entropy (`temperature` < 0.2) or deterministic decoding for precision, syntactic adherence, and schema compliance.

**SCDSR (Speculative Context-Aware Dynamic Sampling & Rollback)** solves this conflict by treating text generation as a discrete state machine. The harness continuously monitors output tokens, triggers state transitions upon matching specific patterns, performs a **context rollback** to scrub speculative low-precision tokens, and resumes generation under tightened parameters.

---

## 2. Core Architecture & Workflow

The architecture relies on a **micro-batch generation loop** (`n_predict` chunking) combined with pattern matching and KV-cache rewinding.

```
                  +--------------------------------+
                  |    State 0: NORMAL/CREATIVE    |
                  |     (high temp, large chunk)   |
                  +---------------+----------------+
                                  |
                                  | Pattern Detected (e.g., `<tool_call>` or `CHARACTER:`)
                                  v
                  +--------------------------------+
                  |         TRIGGER MATCH          |
                  +---------------+----------------+
                                  |
                                  | 1. Truncate context to Trigger Boundary
                                  | 2. Re-align KV Cache (Rollback)
                                  v
                  +--------------------------------+
                  |  State 1: DETERMINISTIC / FOCUSED|
                  |     (low temp, target schema)  |
                  +---------------+----------------+
                                  |
                                  | Completion / Exit Signal Detected
                                  v
                  +--------------------------------+
                  |       State 2: COOLDOWN        |
                  |  (medium temp, anchoring state)|
                  +--------------------------------+
```

### Key Components

1. **State Engine**: Manages sampling profiles (`temp`, `top_p`, `top_k`, `min_p`) per state.
2. **Context Rewind / Rollback Buffer**: Tracks token indices to cleanly discard "overshot" tokens generated before a pattern trigger was fully recognized.
3. **Pattern Matcher**: Evaluates real-time token streams against Regex or AST rules.

---

## 3. Use Case Specifications

### Use Case A: Dynamic Screenplay & Narrative Dialogue Generation

#### Problem
Writing narrative fiction or screenplays with static sampling creates a compromise:
- High temperature makes wild characters engaging, but narrative background descriptions become incoherent.
- Low temperature makes background descriptions structured, but character dialogue becomes repetitive and cliché.

#### Mechanism
The harness adjusts generation profiles dynamically based on character speaker tags and automatically applies a rollback to prevent high-entropy bleeding into structural script headers.

#### State Machine Flow
1. **Narrative/Scene Description (Base State)**: `temp = 0.5`, `n_predict = 32`
2. **Character Trigger Detected** (e.g., `PETR (unpredictable):`):
   - **Rollback**: Trim any extra tokens generated after the character header.
   - **Transition**: `DICE_ROLL` State (`temp = 1.1`, `n_predict = 48`, higher `top_p`).
3. **Character Exit / Scene Shift**:
   - **Transition**: `COOLDOWN` State (`temp = 0.4`, `n_predict = 16`) for 1-2 chunks to restore narrative coherence before returning to Base State.

#### Pseudo-Code Implementation

```python
class ScreenplayStateEngine:
    def __init__(self):
        self.profiles = {
            "NARRATIVE": {"temp": 0.5, "top_p": 0.9, "chunk": 32},
            "CHAOTIC_CHAR": {"temp": 1.1, "top_p": 0.95, "chunk": 48},
            "GROUNDED_CHAR": {"temp": 0.3, "top_p": 0.8, "chunk": 24},
            "COOLDOWN": {"temp": 0.4, "top_p": 0.85, "chunk": 16}
        }
        self.current_state = "NARRATIVE"

    def evaluate_transition(self, generated_text: str):
        if "PETR:" in generated_text:
            return "CHAOTIC_CHAR", "PETR:"
        elif "JAN:" in generated_text:
            return "GROUNDED_CHAR", "JAN:"
        elif "\n\n" in generated_text and self.current_state != "NARRATIVE":
            return "COOLDOWN", "\n\n"
        return self.current_state, None

    def generate(self, prompt: str, llm_backend) -> str:
        context = prompt
        while not llm_backend.is_finished():
            cfg = self.profiles[self.current_state]
            
            # Micro-batch execution
            tokens = llm_backend.completion(
                prompt=context,
                n_predict=cfg["chunk"],
                temperature=cfg["temp"],
                top_p=cfg["top_p"]
            )
            
            next_state, match_pattern = self.evaluate_transition(tokens)
            
            if next_state != self.current_state:
                # Execute Rollback: slice text right after the match pattern
                clean_tokens = rollback_to_pattern(tokens, match_pattern)
                context += clean_tokens
                self.current_state = next_state
            else:
                context += tokens
                
        return context
```

---

### Use Case B: High-Creativity Agent with Guarded Tool Calls

#### Problem
Reasoning and agentic chain-of-thought benefit from high creativity (`temp >= 0.8`) to discover non-obvious problem-solving paths. However, tool invocation demands high structural integrity. High temperature during tool call generation frequently leads to:
- Malformed JSON syntax.
- Hallucinated tool names or parameter types.

#### Mechanism
The agent generates chain-of-thought at a high temperature. The moment the token stream enters a tool-call boundary (e.g., `<tool_call>`), the engine triggers a **context rollback** to the `<tool_call>` tag and switches to a deterministic profile (`temp = 0.0` or JSON grammar constrained) for the payload.

#### State Machine Flow

```
[Thought / CoT Phase]  --->  [Trigger: "<tool_call>"]  --->  [Rollback & Switch]  --->  [Payload Phase]
temp = 0.85, top_p = 0.9       Detected mid-chunk            Discard pre-tag noise       temp = 0.0 / Constrained Schema
```

1. **Reasoning State**: `temp = 0.85`, `top_p = 0.9`
2. **Trigger Detection**: Stream matcher identifies `<tool_call>` (or partial match like `{"name":`).
3. **Rollback Execution**:
   - Truncate output exactly to `<tool_call>`.
   - Invalidate KV-cache for discarded speculative tokens.
4. **Execution State**: `temp = 0.0`, strict structural sampling (or CFG grammar enforcement).
5. **Post-Tool State**: Return to `temp = 0.85` after closing tag `</tool_call>`.

#### Pseudo-Code Implementation

```python
import re

class GuardedToolCallEngine:
    def __init__(self, backend):
        self.backend = backend
        self.TOOL_START = "<tool_call>"
        self.TOOL_END = "</tool_call>"

    def generate_agent_step(self, prompt: str) -> str:
        context = prompt
        in_tool_call = False
        
        while True:
            temp = 0.0 if in_tool_call else 0.85
            chunk_size = 12 if in_tool_call else 32
            
            # Micro-batch call
            chunk = self.backend.generate(
                prompt=context,
                n_predict=chunk_size,
                temperature=temp
            )
            
            # Check for tool call trigger during creative mode
            if not in_tool_call and self.TOOL_START in chunk:
                # 1. Rollback: Locate start of tool call tag
                prefix, _, _ = chunk.partition(self.TOOL_START)
                
                # 2. Append valid creative prefix + exact start marker
                context += prefix + self.TOOL_START
                
                # 3. Switch State
                in_tool_call = True
                
                # Rewind KV Cache backend to exact match offset
                self.backend.rewind_kv_cache(to_text=context)
                continue

            # Check for tool call exit during deterministic mode
            if in_tool_call and self.TOOL_END in chunk:
                tool_payload, _, trailing = chunk.partition(self.TOOL_END)
                context += tool_payload + self.TOOL_END
                in_tool_call = False
                continue

            context += chunk
            if self.backend.is_eos():
                break

        return context
```

---

## 4. Trade-offs and Considerations

| Factor | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **Latency Re-computation** | Truncating generated tokens invalidates KV-cache tail. | Keep micro-batch sizes (`n_predict`) small (8–16 tokens) around trigger zones to minimize wasted generation. |
| **Boundary Artefacts** | Sudden temperature drops mid-phrase can feel unnatural in narrative text. | Implement a **`COOLDOWN`** intermediate state (gradual temperature step-down) rather than hard binary toggles. |
| **Backend Support** | Requires engine support for prompt/KV-cache truncation (e.g., `llama.cpp` sequence management, vLLM prompt caching). | Use stateful inference engines with explicit cache state manipulation APIs. |

---

## 5. Summary & Next Steps

Dynamic Sampling with Context Rollback bridges the gap between open-ended LLM intelligence and deterministic execution requirements. 

- **Next Implementations**:
  - Integration of **Grammar-Based Constrained Decoding (GBNF / JSON Schema)** automatically attached upon `TOOL_START` triggers.
  - Latency profiling for KV-cache rewinds across local engine backends (`llama.cpp` C++ API vs. HTTP server endpoints).
  - 
