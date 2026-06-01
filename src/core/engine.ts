import type { ChatCompletionRequest } from "../types/openai.js";
import type {
  GenerateRequest,
  GenerateResult,
  InferenceProvider,
  ModelDescriptor,
} from "./types.js";

export class InferenceEngine {
  constructor(private readonly providers: InferenceProvider[]) {}

  listModels(): ModelDescriptor[] {
    return this.providers.flatMap((provider) => provider.listModels());
  }

  async generate(request: ChatCompletionRequest): Promise<GenerateResult> {
    const provider = this.resolveProvider(request.model);
    return provider.generate(this.toGenerateRequest(request));
  }

  stream(request: ChatCompletionRequest): AsyncGenerator<string> {
    const provider = this.resolveProvider(request.model);
    return provider.stream(this.toGenerateRequest(request));
  }

  private resolveProvider(model: string): InferenceProvider {
    const provider = this.providers.find((candidate) => candidate.supports(model));

    if (!provider) {
      throw new Error(`Unknown model "${model}".`);
    }

    return provider;
  }

  private toGenerateRequest(request: ChatCompletionRequest): GenerateRequest {
    return {
      model: request.model,
      messages: request.messages,
      sampling: {
        temperature: request.temperature ?? 0.7,
        topP: request.top_p ?? 1,
        maxTokens: request.max_tokens ?? 256,
      },
    };
  }
}
