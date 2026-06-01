import type { ChatMessage } from "../types/openai.js";

export function countTextTokens(text: string): number {
  const trimmed = text.trim();
  if (!trimmed) {
    return 0;
  }

  return trimmed.split(/\s+/).length;
}

export function countMessageTokens(messages: ChatMessage[]): number {
  return messages.reduce((total, message) => {
    return total + countTextTokens(message.content) + 4;
  }, 0);
}
