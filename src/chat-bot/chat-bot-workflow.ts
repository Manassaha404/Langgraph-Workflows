import "dotenv/config";
import { randomUUID } from "crypto";
import * as readline from "readline";
import { StateGraph, StateSchema, START, END, MessagesValue } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import type { GraphNode } from "@langchain/langgraph";
import { z } from "zod/v4";
import { ReducedValue } from "@langchain/langgraph";
import { BaseMessage, HumanMessage } from "@langchain/core/messages";
import { RedisSaver } from "@langchain/langgraph-checkpoint-redis";
// Model 
const model = new ChatOpenAI({
  model: "gpt-4o-mini",
});

// State Schema 
const MessagesState = new StateSchema({
  messages: MessagesValue
});

// Graph Node 
const llmCall: GraphNode<typeof MessagesState> = async (state) => {
  const response = await model.invoke(state.messages);
  return { messages: [response] };
};

const graph = new StateGraph(MessagesState)
  .addNode("llmCall", llmCall)
  .addEdge(START, "llmCall")
  .addEdge("llmCall", END);

const checkpointer = await RedisSaver.fromUrl("redis://localhost:6379", {
  defaultTTL: 60, // TTL in minutes
  refreshOnRead: true,
});
const agent = graph.compile({ checkpointer });

const startChatBot = async () => {
  const threadId = randomUUID(); // random thread Id 

  //readline for input output 
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  //config for 
  const config = {
    configurable: { thread_id: threadId },
  };
  const ask = (): void => {
    rl.question("You: ", async (input) => {
      const trimmed = input.trim();
      if (!trimmed || trimmed.toLowerCase() === "exit") {
        console.log("\nGoodbye!\n");
        rl.close();
        await checkpointer.end();
        return;
      }

      try {
        const result = await agent.invoke(
          { messages: [new HumanMessage(trimmed)] },
          config,
        );

        const last = result.messages.at(-1);
        console.log(`\nAssistant: ${last?.content}\n`);
      } catch (err) {
        console.error("Error:", err);
      }

      ask(); // next turn
    });
  };

  ask();
};

export default startChatBot;
