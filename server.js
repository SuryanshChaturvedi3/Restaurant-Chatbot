// server.js — Friendly Waiter Agent (INR), Gemini 2.0 Flash, Tool-Calling Agent

import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
dotenv.config();


/*------------------CAUTION---------------------
## Note
This project uses third-party APIs.
The live response may be unavailable if the API quota or free-tier limit is exceeded.
The codebase and logic remain fully functional.

--------------------------------------------*/

/* ----------------------------- LangChain imports ----------------------------- */
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { z } from "zod";

// Requires a langchain version that exports this subpath (e.g., 0.2.15+)
import { createToolCallingAgent, AgentExecutor } from "langchain/agents";

/* --------------------------- 1) Model: Gemini 2.0 Flash ---------------------- */
const model = new ChatGoogleGenerativeAI({
  model: "gemini-2.0-flash",
  temperature: 0.7,
  maxOutputTokens: 1024,
  apiKey: process.env.GOOGLE_API_KEY,
});

/* --------------------------- 2) Tool: getMenu (INR) -------------------------- */
const getMenuTool = new DynamicStructuredTool({
  name: "getMenu",
  description:
    "Return menu items and INR prices for a category: breakfast, lunch, or dinner.",
  schema: z.object({
    category: z
      .enum(["breakfast", "lunch", "dinner"])
      .describe("Menu category to retrieve (use lowercase)"),
  }),
  func: async ({ category }) => {
    const menus = {
      breakfast: [
        "Masala Omelette – ₹179",
        "Pancakes – ₹149",
        "Poha – ₹99",
        "Idli Sambar – ₹129",
      ],
      lunch: [
        "Veg Thali – ₹249",
        "Butter Chicken – ₹349",
        "Jeera Rice – ₹129",
        "Paneer Butter Masala – ₹299",
      ],
      dinner: [
        "Paneer Tikka – ₹299",
        "Dal Makhani – ₹219",
        "Tandoori Roti – ₹25",
        "Veg Biryani – ₹269",
      ],
    };

    const list = menus[category];
    if (!list) return "Category not found.";
    // Return a friendly, readable string (agent will wrap it in nice text)
    return list.map((x) => `• ${x}`).join("\n");
  },
});

/* --------------------------- 3) Prompt (must include agent_scratchpad) ------- */
const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `You are a warm, friendly restaurant assistant (like a waiter).
- When the user asks for a menu category, call the getMenu tool.
- Always answer in a polite, helpful tone.
- Prices are in INR (₹). Keep responses concise and easy to read.`,
  ],
  ["human", "{input}"],
  // Required for tool-calling agents so they can think/plan & log steps:
  ["ai", "{agent_scratchpad}"],
]);

/* --------------------------- 4) Agent + Executor ----------------------------- */
const tools = [getMenuTool];

const agent = await createToolCallingAgent({
  llm: model,
  tools,
  prompt,
 
});

const executor = await AgentExecutor.fromAgentAndTools({
  agent,
  tools,
  // Capture intermediate tool steps in case the model does not emit a final message
  returnIntermediateSteps: true,
  maxIterations: 3,
 
});

/* --------------------------- 5) Express Server ------------------------------- */
const app = express();
app.use(express.json());

// ESM-friendly __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Serve static assets (e.g., index.html)
app.use(express.static(path.join(__dirname, "public")));


app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
}
);

// Basic health route

app.post("/api/chat", async (req, res) => {
  const userInput = req.body.input;
  console.log("User input:", userInput);

  if (!process.env.GOOGLE_API_KEY) {
    return res.status(500).json({
      error: "Missing GOOGLE_API_KEY in environment. Please set it and restart the server.",
    });
  }

  try {
    const result = await executor.invoke({ input: userInput });

    const output = typeof result.output === "string" ? result.output.trim() : "";
    const steps = result.steps || result.intermediateSteps || [];

    // ✅ Use final output if available
    if (output && output !== "Agent stopped without final output.") {
      return res.json({ response: output });
    }

    // ✅ Fallback: use tool observation if output missing
    if (Array.isArray(steps) && steps.length > 0) {
      const lastStep = steps[steps.length - 1];
      if (lastStep?.observation) {
        return res.json({ response: String(lastStep.observation) });
      }
    }

    // ✅ Final fallback
    return res.json({
      response: "Sorry, I couldn’t find that. Could you try rephrasing?",
    });

  } catch (error) {
    console.error("/api/chat error", error);
    return res.status(500).json({ error: "Internal Server Error" });
  }
});


const PORT = 3000;
app.listen(PORT, () =>
  console.log(`🚀 Server running at http://localhost:${PORT}`)
);
