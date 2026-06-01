import type { ChatMessage } from "../types/openai.js";

export interface SamplingConfig {
  temperature: number;
  topP: number;
  maxTokens: number;
}

export interface GenerateRequest {
  model: string;
  messages: ChatMessage[];
  sampling: SamplingConfig;
}

export interface GenerateResult {
  model: string;
  outputText: string;
  promptTokens: number;
  completionTokens: number;
  finishReason: "stop" | "length";
}

export interface ModelDescriptor {
  id: string;
  owner: string;
  created: number;
  contextWindow: number;
  maxOutputTokens: number;
}

export interface InferenceProvider {
  readonly name: string;
  listModels(): ModelDescriptor[];
  supports(model: string): boolean;
  generate(request: GenerateRequest): Promise<GenerateResult>;
  stream(request: GenerateRequest): AsyncGenerator<string>;
}
