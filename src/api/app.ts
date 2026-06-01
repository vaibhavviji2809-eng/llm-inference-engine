import cors from "cors";
import express from "express";
import { v4 as uuidv4 } from "uuid";
import { chatCompletionRequestSchema } from "./schemas.js";
import type { InferenceEngine } from "../core/engine.js";
import type {
  ChatCompletionResponse,
  ModelCard,
} from "../types/openai.js";

function writeSseEvent(response: express.Response, data: unknown): void {
  response.write(`data: ${JSON.stringify(data)}\n\n`);
}

export function createApp(engine: InferenceEngine): express.Express {
  const app = express();

  app.use(cors());
  app.use(express.json({ limit: "1mb" }));

  app.get("/health", (_request, response) => {
    response.json({
      ok: true,
      models: engine.listModels().length,
    });
  });

  app.get("/v1/models", (_request, response) => {
    const models: ModelCard[] = engine.listModels().map((model) => ({
      id: model.id,
      object: "model",
      created: model.created,
      owned_by: model.owner,
    }));

    response.json({
      object: "list",
      data: models,
    });
  });

  app.post("/v1/chat/completions", async (request, response) => {
    const parsed = chatCompletionRequestSchema.safeParse(request.body);

    if (!parsed.success) {
      response.status(400).json({
        error: {
          message: "Invalid request payload.",
          details: parsed.error.flatten(),
        },
      });
      return;
    }

    try {
      const payload = parsed.data;

      if (payload.stream) {
        response.setHeader("Content-Type", "text/event-stream");
        response.setHeader("Cache-Control", "no-cache");
        response.setHeader("Connection", "keep-alive");
        response.flushHeaders();

        const id = `chatcmpl-${uuidv4()}`;
        const created = Math.floor(Date.now() / 1000);

        for await (const chunk of engine.stream(payload)) {
          writeSseEvent(response, {
            id,
            object: "chat.completion.chunk",
            created,
            model: payload.model,
            choices: [
              {
                index: 0,
                delta: {
                  content: chunk,
                },
                finish_reason: null,
              },
            ],
          });
        }

        writeSseEvent(response, {
          id,
          object: "chat.completion.chunk",
          created,
          model: payload.model,
          choices: [
            {
              index: 0,
              delta: {},
              finish_reason: "stop",
            },
          ],
        });
        response.write("data: [DONE]\n\n");
        response.end();
        return;
      }

      const result = await engine.generate(payload);
      const responseBody: ChatCompletionResponse = {
        id: `chatcmpl-${uuidv4()}`,
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: result.model,
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: result.outputText,
            },
            finish_reason: result.finishReason,
          },
        ],
        usage: {
          prompt_tokens: result.promptTokens,
          completion_tokens: result.completionTokens,
          total_tokens: result.promptTokens + result.completionTokens,
        },
      };

      response.json(responseBody);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unknown inference error.";
      response.status(404).json({
        error: {
          message,
        },
      });
    }
  });

  return app;
}
