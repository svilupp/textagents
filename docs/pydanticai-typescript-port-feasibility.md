# PydanticAI to TypeScript Port Feasibility Analysis

**Date**: January 2026
**Target Stack**: AI SDK v6 + textprompts-ts + Zod
**Source**: PydanticAI v1.41.0

---

## Executive Summary

A TypeScript port of PydanticAI concepts is **highly feasible** with the AI SDK v6 as the foundation. The AI SDK has evolved significantly and now provides most of the abstractions that PydanticAI offers. However, some features require custom implementation or will need unsupported warnings.

| Category | Feasibility | Notes |
|----------|-------------|-------|
| Core Agent abstraction | ✅ Excellent | AI SDK has `Agent` and `ToolLoopAgent` classes |
| Tool system | ✅ Excellent | Near 1:1 mapping with Zod schemas |
| Structured output | ✅ Excellent | `Output.object()` with Zod validation |
| Streaming | ✅ Excellent | `streamText` with async iterators |
| Dependencies/Context | ⚠️ Partial | Must be implemented as custom pattern |
| Message history | ✅ Excellent | `UIMessage` and `ModelMessage` types |
| Retries | ⚠️ Partial | Built-in for tools, manual for output validation |
| Model providers | ✅ Excellent | AI Gateway supports all major providers |
| Multi-agent | ✅ Good | Programmatic hand-off supported |
| MCP | ✅ Good | Native MCP support in AI SDK 6 |
| Evals | ⚠️ Partial | No built-in eval framework |
| Logfire/Observability | ⚠️ Partial | OpenTelemetry manual integration |

---

## Detailed Mapping: PydanticAI → AI SDK

### 1. Agent Class

#### PydanticAI
```python
from pydantic_ai import Agent

agent = Agent(
    'openai:gpt-5',
    deps_type=MyDeps,
    output_type=MyOutput,
    instructions='You are helpful.',
    retries=3,
    model_settings=ModelSettings(temperature=0),
    tools=[my_tool],
    end_strategy='exhaustive',
)
```

#### AI SDK Equivalent
```typescript
import { Agent, Output, tool } from 'ai';
import { z } from 'zod';

const agent = new Agent({
  model: 'openai/gpt-5',
  system: 'You are helpful.',
  tools: { myTool },
  // No direct deps_type - use closure or context pattern
  // No direct output_type - use Output.object() in generate()
});

// Or use ToolLoopAgent for automatic tool execution
import { ToolLoopAgent } from 'ai';

const loopAgent = new ToolLoopAgent({
  model: 'openai/gpt-5',
  system: 'You are helpful.',
  tools: { myTool },
  stopWhen: stepCountIs(10), // Similar to end_strategy
});
```

### 2. Agent Constructor Parameters Mapping

| PydanticAI Parameter | AI SDK Equivalent | Status |
|---------------------|-------------------|--------|
| `model` | `model` | ✅ Direct |
| `output_type` | `Output.object({ schema })` | ✅ At generate() |
| `instructions` | `system` | ✅ Direct |
| `system_prompt` | `system` | ✅ Direct |
| `deps_type` | N/A - use closure pattern | ⚠️ Custom |
| `name` | N/A | ❌ Not supported |
| `model_settings` | `providerOptions` | ✅ Partial |
| `retries` | Tool-level `maxRetries` | ⚠️ Partial |
| `output_retries` | N/A | ❌ Custom needed |
| `tools` | `tools` | ✅ Direct |
| `builtin_tools` | Provider-specific tools | ✅ Native |
| `toolsets` | N/A | ❌ Custom needed |
| `mcp_servers` | Native MCP support | ✅ Direct |
| `end_strategy` | `stopWhen` | ✅ Equivalent |
| `instrument` | Manual OTel | ⚠️ Custom |
| `validation_context` | N/A | ❌ Custom needed |
| `prepare_tools` | `prepareStep` | ✅ Equivalent |

### 3. Tool System

#### PydanticAI
```python
@agent.tool
async def get_weather(ctx: RunContext[MyDeps], city: str) -> str:
    """Get weather for a city."""
    return await ctx.deps.weather_api.get(city)

@agent.tool_plain
def calculate(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
```

#### AI SDK Equivalent
```typescript
import { tool } from 'ai';
import { z } from 'zod';

const getWeather = tool({
  description: 'Get weather for a city',
  inputSchema: z.object({
    city: z.string().describe('City name'),
  }),
  outputSchema: z.object({
    temperature: z.number(),
    conditions: z.string(),
  }),
  execute: async ({ city }) => {
    // Access deps via closure
    return await weatherApi.get(city);
  },
});

const calculate = tool({
  description: 'Add two numbers',
  inputSchema: z.object({
    a: z.number().describe('First number'),
    b: z.number().describe('Second number'),
  }),
  execute: async ({ a, b }) => ({ result: a + b }),
});
```

#### Tool Parameter Mapping

| PydanticAI Tool Param | AI SDK Equivalent | Status |
|----------------------|-------------------|--------|
| `function` | `execute` | ✅ Direct |
| `takes_ctx` | N/A (closure pattern) | ⚠️ Custom |
| `max_retries` | N/A | ⚠️ Custom |
| `name` | Object key in `tools` | ✅ Direct |
| `description` | `description` | ✅ Direct |
| `prepare` | `prepareStep` (agent-level) | ⚠️ Partial |
| `docstring_format` | N/A (explicit descriptions) | ❌ Not needed |
| `require_parameter_descriptions` | N/A | ❌ Not applicable |
| `strict` | N/A | ❌ Provider-specific |
| `sequential` | N/A | ❌ Custom needed |
| `requires_approval` | `needsApproval` | ✅ Direct |
| `timeout` | N/A | ❌ Custom needed |

### 4. RunContext / Dependencies

#### PydanticAI
```python
from dataclasses import dataclass

@dataclass
class MyDeps:
    api_key: str
    db: Database

@agent.tool
async def query(ctx: RunContext[MyDeps], sql: str) -> list:
    return await ctx.deps.db.query(sql)

result = await agent.run('prompt', deps=MyDeps(...))
```

#### AI SDK Equivalent (Closure Pattern)
```typescript
// Create agent factory with dependencies
function createAgent(deps: MyDeps) {
  const queryTool = tool({
    description: 'Query the database',
    inputSchema: z.object({ sql: z.string() }),
    execute: async ({ sql }) => {
      // Access deps via closure
      return await deps.db.query(sql);
    },
  });

  return new ToolLoopAgent({
    model: 'openai/gpt-5',
    tools: { query: queryTool },
  });
}

// Usage
const agent = createAgent({ apiKey: 'xxx', db: myDb });
const result = await agent.generate({ prompt: 'Query users' });
```

**⚠️ Warning**: AI SDK has no built-in dependency injection. Use factory functions or closures.

### 5. Structured Output

#### PydanticAI
```python
from pydantic import BaseModel

class WeatherReport(BaseModel):
    city: str
    temperature: float
    conditions: str

agent = Agent('openai:gpt-5', output_type=WeatherReport)
result = await agent.run('Weather in Paris?')
print(result.output.temperature)  # Typed!
```

#### AI SDK Equivalent
```typescript
import { generateText, Output } from 'ai';
import { z } from 'zod';

const WeatherReport = z.object({
  city: z.string(),
  temperature: z.number(),
  conditions: z.string(),
});

const { output } = await generateText({
  model: 'openai/gpt-5',
  prompt: 'Weather in Paris?',
  output: Output.object({ schema: WeatherReport }),
});

console.log(output.temperature); // Typed!
```

### 6. Output Validators

#### PydanticAI
```python
@agent.output_validator
async def validate(ctx: RunContext, output: str) -> str:
    if 'forbidden' in output:
        raise ModelRetry('Cannot use that word')
    return output
```

#### AI SDK Equivalent
```typescript
import { generateText, Output } from 'ai';
import { z } from 'zod';

// Use Zod refinements for validation
const OutputSchema = z.object({
  text: z.string().refine(
    (val) => !val.includes('forbidden'),
    { message: 'Cannot use that word' }
  ),
});

// For retry on validation failure - custom loop needed
async function generateWithRetry(prompt: string, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const { output } = await generateText({
        model: 'openai/gpt-5',
        prompt,
        output: Output.object({ schema: OutputSchema }),
      });
      return output;
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      // Add error context to prompt for retry
    }
  }
}
```

**⚠️ Warning**: AI SDK doesn't have automatic output validation retries. Custom implementation needed.

### 7. Streaming

#### PydanticAI
```python
async with agent.run_stream('Tell me a story') as stream:
    async for chunk in stream.stream_text():
        print(chunk, end='')

    result = await stream.get_output()
```

#### AI SDK Equivalent
```typescript
import { streamText } from 'ai';

const result = streamText({
  model: 'openai/gpt-5',
  prompt: 'Tell me a story',
});

for await (const chunk of result.textStream) {
  process.stdout.write(chunk);
}

const finalText = await result.text;
```

### 8. Streaming with Structured Output

#### PydanticAI
```python
async with agent.run_stream('Describe a whale') as stream:
    async for partial in stream.stream_structured():
        print(partial)  # Partially validated
```

#### AI SDK Equivalent
```typescript
import { streamText, Output } from 'ai';

const result = streamText({
  model: 'openai/gpt-5',
  prompt: 'Describe a whale',
  output: Output.object({ schema: WhaleSchema }),
});

for await (const partial of result.partialOutputStream) {
  console.log(partial); // Partial object
}

const whale = await result.output; // Final validated
```

### 9. Message History / Multi-turn

#### PydanticAI
```python
result1 = await agent.run('What is Python?')
result2 = await agent.run(
    'Tell me more',
    message_history=result1.new_messages()
)
```

#### AI SDK Equivalent
```typescript
import { generateText, convertToModelMessages } from 'ai';

const result1 = await generateText({
  model: 'openai/gpt-5',
  prompt: 'What is Python?',
});

const result2 = await generateText({
  model: 'openai/gpt-5',
  messages: [
    ...convertToModelMessages(result1.messages),
    { role: 'user', content: 'Tell me more' },
  ],
});
```

### 10. Dynamic Instructions/System Prompts

#### PydanticAI
```python
@agent.instructions
async def dynamic_instructions(ctx: RunContext[MyDeps]) -> str:
    name = await ctx.deps.get_user_name()
    return f"The user's name is {name}"
```

#### AI SDK Equivalent
```typescript
// Use function to generate system prompt
async function generateWithDynamicSystem(deps: MyDeps, prompt: string) {
  const userName = await deps.getUserName();

  return generateText({
    model: 'openai/gpt-5',
    system: `The user's name is ${userName}`,
    prompt,
  });
}
```

### 11. Model Settings

#### PydanticAI
```python
from pydantic_ai import ModelSettings

agent = Agent(
    'openai:gpt-5',
    model_settings=ModelSettings(
        temperature=0.7,
        max_tokens=1000,
        timeout=30,
    )
)
```

#### AI SDK Equivalent
```typescript
const result = await generateText({
  model: 'openai/gpt-5',
  prompt: 'Hello',
  maxTokens: 1000,
  temperature: 0.7,
  // timeout via AbortController
  abortSignal: AbortSignal.timeout(30000),
  providerOptions: {
    openai: {
      // Provider-specific options
    },
  },
});
```

| ModelSettings Param | AI SDK | Status |
|---------------------|--------|--------|
| `temperature` | `temperature` | ✅ Direct |
| `max_tokens` | `maxTokens` | ✅ Direct |
| `timeout` | `AbortSignal.timeout()` | ⚠️ Different API |
| `parallel_tool_calls` | N/A | ❌ Provider-specific |
| `seed` | `seed` | ✅ Direct |
| `presence_penalty` | `presencePenalty` | ✅ Direct |
| `frequency_penalty` | `frequencyPenalty` | ✅ Direct |
| `top_p` | `topP` | ✅ Direct |

### 12. Retry Logic (ModelRetry)

#### PydanticAI
```python
from pydantic_ai import ModelRetry

@agent.tool
async def get_user(ctx: RunContext, name: str) -> int:
    user = ctx.deps.db.find(name)
    if not user:
        raise ModelRetry(f'No user named {name}')
    return user.id
```

#### AI SDK Equivalent
```typescript
// AI SDK tools can throw errors that become tool results
const getUserTool = tool({
  description: 'Get user by name',
  inputSchema: z.object({ name: z.string() }),
  execute: async ({ name }) => {
    const user = await db.find(name);
    if (!user) {
      // Return error message as result - model sees this
      throw new Error(`No user named ${name}, try another`);
    }
    return { id: user.id };
  },
});
```

**⚠️ Warning**: No explicit `ModelRetry` exception. Tool errors become results for the model to see.

### 13. Usage Limits

#### PydanticAI
```python
from pydantic_ai import UsageLimits

result = await agent.run(
    'prompt',
    usage_limits=UsageLimits(
        request_limit=10,
        total_tokens_limit=7000,
    )
)
```

#### AI SDK Equivalent
```typescript
const result = await generateText({
  model: 'openai/gpt-5',
  prompt: 'prompt',
  stopWhen: stepCountIs(10), // Request limit equivalent
  maxTokens: 7000, // Only controls output tokens
  // For total token limits - custom tracking needed
});
```

**⚠️ Warning**: AI SDK has limited usage limits. `stopWhen` for steps, but no built-in token tracking.

### 14. Multi-Agent Patterns

#### PydanticAI (Delegation)
```python
@main_agent.tool
async def delegate(ctx: RunContext, query: str) -> str:
    result = await research_agent.run(query, usage=ctx.usage)
    return result.output
```

#### AI SDK Equivalent
```typescript
const mainAgent = new ToolLoopAgent({
  model: 'openai/gpt-5',
  tools: {
    delegate: tool({
      description: 'Delegate research task',
      inputSchema: z.object({ query: z.string() }),
      execute: async ({ query }) => {
        const result = await researchAgent.generate({ prompt: query });
        return result.text;
      },
    }),
  },
});
```

### 15. Human-in-the-Loop / Tool Approval

#### PydanticAI
```python
@agent.tool(requires_approval=True)
async def dangerous_action(ctx: RunContext) -> str:
    ...

# Returns DeferredToolRequests
# Resume with DeferredToolResults
```

#### AI SDK Equivalent
```typescript
const dangerousTool = tool({
  description: 'Dangerous action',
  inputSchema: z.object({ action: z.string() }),
  needsApproval: true,
  execute: async ({ action }) => {
    // Execution happens only after approval
  },
});

// AI SDK handles approval flow automatically in streaming
```

### 16. MCP Integration

#### PydanticAI
```python
from pydantic_ai.mcp import MCPServerStreamableHTTP

server = MCPServerStreamableHTTP('http://localhost:8000')
async with Agent('openai:gpt-5', mcp_servers=[server]) as agent:
    result = await agent.run('Use MCP')
```

#### AI SDK Equivalent
```typescript
// AI SDK 6 has native MCP support
import { openai } from '@ai-sdk/openai';

// MCP tools via provider-specific integration
const result = await generateText({
  model: 'openai/gpt-5',
  tools: {
    // MCP tools can be imported/configured
  },
});
```

---

## Features Requiring Custom Implementation

### 1. Dependency Injection System

**Implementation Strategy**:
```typescript
// deps.ts
export interface AgentDeps<T = unknown> {
  deps: T;
}

export function createAgentWithDeps<TDeps, TOutput>(
  config: AgentConfig<TDeps, TOutput>
) {
  return (deps: TDeps) => {
    const tools = Object.fromEntries(
      Object.entries(config.tools).map(([name, toolFn]) => [
        name,
        toolFn(deps),
      ])
    );

    return new ToolLoopAgent({
      model: config.model,
      system: typeof config.system === 'function'
        ? config.system(deps)
        : config.system,
      tools,
    });
  };
}
```

### 2. Output Validation with Retry

**Implementation Strategy**:
```typescript
// output-validator.ts
export async function generateWithValidation<T>(
  config: GenerateConfig,
  schema: z.ZodSchema<T>,
  validators: ((output: T) => T | Promise<T>)[],
  maxRetries = 3
): Promise<T> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const { output } = await generateText({
        ...config,
        output: Output.object({ schema }),
        prompt: lastError
          ? `${config.prompt}\n\nPrevious attempt failed: ${lastError.message}`
          : config.prompt,
      });

      // Run custom validators
      let validated = output;
      for (const validator of validators) {
        validated = await validator(validated);
      }

      return validated;
    } catch (error) {
      lastError = error as Error;
    }
  }

  throw lastError;
}
```

### 3. Toolsets

**Implementation Strategy**:
```typescript
// toolset.ts
export interface Toolset {
  getTools(): Record<string, ReturnType<typeof tool>>;
}

export function combineToolsets(...toolsets: Toolset[]) {
  return toolsets.reduce(
    (acc, ts) => ({ ...acc, ...ts.getTools() }),
    {}
  );
}
```

---

## Features NOT Portable (Emit Warnings)

These features should emit clear warnings when used:

### 1. `docstring_format`
```typescript
console.warn(
  '[textagents] docstring_format is not supported in TypeScript. ' +
  'Use explicit description fields in tool definitions.'
);
```

### 2. `validation_context`
```typescript
console.warn(
  '[textagents] validation_context is not supported. ' +
  'Use closure pattern to pass context to validators.'
);
```

### 3. `output_retries` (without custom implementation)
```typescript
console.warn(
  '[textagents] output_retries requires custom implementation. ' +
  'Use generateWithValidation() helper instead.'
);
```

### 4. `sequential` tools
```typescript
console.warn(
  '[textagents] sequential tool execution not natively supported. ' +
  'Implement custom tool orchestration if needed.'
);
```

### 5. `prepare` function (tool-level)
```typescript
console.warn(
  '[textagents] Tool-level prepare functions not supported. ' +
  'Use prepareStep at agent level instead.'
);
```

---

## Integration with textprompts-ts

The existing `textprompts-ts` package provides:
- Template loading from `.txt` files with TOML metadata
- Safe variable substitution (`PromptString.format()`)
- Cross-language compatibility

### Usage in TypeScript Port

```typescript
import { loadPrompt, PromptString } from 'textprompts';
import { generateText, Output } from 'ai';

// Load prompt template
const prompt = await loadPrompt('./prompts/safety-judge.txt');

// Format with variables
const formatted = prompt.format({
  user_input: userMessage,
  model_output: aiResponse,
});

// Use with AI SDK
const { output } = await generateText({
  model: 'openai/gpt-5',
  system: prompt.meta.instructions,
  prompt: formatted,
  output: Output.object({ schema: JudgeOutputSchema }),
});
```

---

## Proposed Architecture

```
textagents-ts/
├── src/
│   ├── agent.ts           # TextAgent class wrapping AI SDK
│   ├── loader.ts          # Load .txt agent definitions
│   ├── parser.ts          # TOML front-matter parsing
│   ├── schema-builder.ts  # Convert TOML output specs to Zod
│   ├── deps.ts            # Dependency injection patterns
│   ├── validators.ts      # Output validation with retry
│   ├── warnings.ts        # Unsupported feature warnings
│   └── index.ts           # Public API
├── package.json
└── tsconfig.json
```

### Core TextAgent Class

```typescript
import { z } from 'zod';
import { generateText, streamText, Output, tool } from 'ai';
import { loadPrompt } from 'textprompts';

export class TextAgent<TOutput> {
  private spec: AgentSpec;
  private outputSchema: z.ZodSchema<TOutput>;

  constructor(spec: AgentSpec) {
    this.spec = spec;
    this.outputSchema = buildZodSchema(spec.outputType);
  }

  static async load<T>(path: string): Promise<TextAgent<T>> {
    const spec = await parseAgentFile(path);
    return new TextAgent<T>(spec);
  }

  async run(inputs: Record<string, unknown>): Promise<TOutput> {
    const prompt = this.formatPrompt(inputs);

    const { output } = await generateText({
      model: this.spec.model,
      system: this.spec.instructions,
      prompt,
      output: Output.object({ schema: this.outputSchema }),
      temperature: this.spec.settings?.temperature ?? 0,
    });

    return output;
  }

  async *stream(inputs: Record<string, unknown>) {
    const prompt = this.formatPrompt(inputs);

    const result = streamText({
      model: this.spec.model,
      system: this.spec.instructions,
      prompt,
      output: Output.object({ schema: this.outputSchema }),
    });

    for await (const partial of result.partialOutputStream) {
      yield partial;
    }
  }
}
```

---

## Recommendations

### Phase 1: Core Port (MVP)
1. ✅ Agent loading from `.txt` files
2. ✅ Basic `run()` and `stream()` methods
3. ✅ Structured output with Zod schemas
4. ✅ Integration with textprompts-ts
5. ✅ Warning system for unsupported features

### Phase 2: Enhanced Features
1. ⚠️ Dependency injection pattern
2. ⚠️ Output validation with retry
3. ⚠️ Toolset abstractions
4. ⚠️ Usage limits tracking

### Phase 3: Advanced Features
1. ❌ Multi-agent orchestration
2. ❌ Eval framework
3. ❌ Observability integration

---

## Conclusion

**Feasibility: HIGH**

The TypeScript port is highly feasible because:

1. **AI SDK v6 alignment**: The AI SDK has evolved to provide most PydanticAI abstractions natively
2. **Zod ↔ Pydantic**: Near-perfect mapping for schema validation
3. **textprompts-ts exists**: Cross-language prompt loading is already solved
4. **Streaming parity**: Both frameworks have excellent streaming support
5. **Type safety**: TypeScript generics can express the same patterns as Python's type hints

**Key gaps requiring custom implementation**:
- Dependency injection (use factory/closure pattern)
- Output validation retries (custom wrapper)
- Some tool-level configurations

**Recommended approach**: Start with the core `TextAgent` class that wraps AI SDK, emit warnings for unsupported PydanticAI features, and iteratively add advanced features based on user needs.
