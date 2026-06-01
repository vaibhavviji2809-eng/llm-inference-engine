import type {
  GenerateRequest,
  GenerateResult,
  InferenceProvider,
  ModelDescriptor,
} from "../core/types.js";
import { countMessageTokens, countTextTokens } from "../core/tokenizer.js";
import { sleep } from "../utils/sleep.js";

const MODEL_CATALOG: ModelDescriptor[] = [
  {
    id: "mock-llama-8b",
    owner: "codex-labs",
    created: 1_717_171_717,
    contextWindow: 8192,
    maxOutputTokens: 1024,
  },
  {
    id: "mock-mixture-42b",
    owner: "codex-labs",
    created: 1_717_171_718,
    contextWindow: 32768,
    maxOutputTokens: 2048,
  },
];

function lastUserMessage(request: GenerateRequest): string {
  const userMessages = request.messages.filter((message) => message.role === "user");
  return userMessages.at(-1)?.content ?? "No user prompt was provided.";
}

function buildCompletion(request: GenerateRequest): string {
  const prompt = lastUserMessage(request);
  const normalizedPrompt = prompt.replace(/\s+/g, " ").trim();

  const sections = [
    `Model ${request.model} accepted the prompt and prepared a response path.`,
    `Prompt summary: ${normalizedPrompt || "empty input"}.`,
    `Sampling config: temperature=${request.sampling.temperature.toFixed(2)}, top_p=${request.sampling.topP.toFixed(2)}, max_tokens=${request.sampling.maxTokens}.`,
    "This mock backend is standing in for a real decoder loop, KV cache, and scheduler.",
    "Swap this provider with a GPU-backed runtime and the API surface can stay the same.",
  ];

  return sections.join(" ");
}

export class MockInferenceProvider implements InferenceProvider {
  readonly name = "mock";

  listModels(): ModelDescriptor[] {
    return MODEL_CATALOG;
  }

  supports(model: string): boolean {
    return MODEL_CATALOG.some((candidate) => candidate.id === model);
  }

  async generate(request: GenerateRequest): Promise<GenerateResult> {
    const outputText = buildCompletion(request);
    const promptTokens = countMessageTokens(request.messages);
    const completionTokens = countTextTokens(outputText);

    return {
      model: request.model,
      outputText,
      promptTokens,
      completionTokens,
      finishReason:
        completionTokens >= request.sampling.maxTokens ? "length" : "stop",
    };
  }

  async *stream(request: GenerateRequest): AsyncGenerator<string> {
    const result = buildCompletion(request);
    const chunks = result.split(/(\s+)/).filter(Boolean);

    for (const chunk of chunks) {
      await sleep(35);
      yield chunk;
    }
  }
}
