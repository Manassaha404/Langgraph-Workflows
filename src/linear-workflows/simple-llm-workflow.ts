// state -> llmCall -> end

import "dotenv/config";
import { StateGraph, StateSchema, START, END } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import type { GraphNode } from "@langchain/langgraph";
import { z } from "zod/v4";

// create a llm model
const model = new ChatOpenAI({
  model: "gpt-4o-mini",
});

// define state
const MessagesState = new StateSchema({
  messages: z.string(),
  llmMessage: z.string(),
});

// define a node(llm call) -> actually a function with state in args
const llmCall: GraphNode<typeof MessagesState> = async (state) => {
  // invoke the chat model
  const response = await model.invoke(state.messages);
  // get the responce
  const llmMessageContent = JSON.stringify(response.content);

  // return modified state
  return {
    messages: state.messages,
    llmMessage: llmMessageContent,
  };
};

// define graph
const agent = new StateGraph(MessagesState)
  .addNode("llmCall", llmCall)
  .addEdge(START, "llmCall")
  .addEdge("llmCall", END)
  .compile(); // ← required before invoking

const runSimplellmWorkflow = async () => {
  // invoke the graph or run the graph
  const result = await agent.invoke({
    messages: "what is the capital of India",
    llmMessage: "",
  });

  // print the result
  console.log(result);
};

export default runSimplellmWorkflow;
