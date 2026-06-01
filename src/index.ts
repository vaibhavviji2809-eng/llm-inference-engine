import { createApp } from "./api/app.js";
import { InferenceEngine } from "./core/engine.js";
import { MockInferenceProvider } from "./providers/mockProvider.js";

const port = Number(process.env.PORT ?? 4000);
const host = process.env.HOST ?? "0.0.0.0";

const engine = new InferenceEngine([new MockInferenceProvider()]);
const app = createApp(engine);

app.listen(port, host, () => {
  console.log(`LLM inference engine listening at http://${host}:${port}`);
});
