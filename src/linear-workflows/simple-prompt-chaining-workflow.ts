// topic -> outline -> blog 


import * as z from "zod";
import { ChatOpenAI } from "@langchain/openai";
import { StateGraph, StateSchema, START, END } from "@langchain/langgraph";
import type { GraphNode } from "@langchain/langgraph";

const model = new ChatOpenAI({
  model: "gpt-4o-mini",
});

const state = new StateSchema({
  topic: z.string(),
  outline: z.string().nullable().default(null),
  blog: z.string().nullable().default(null),
});

const generateOutline: GraphNode<typeof state> = async (state) => {
  const topic = state.topic;
  const prompt = `create a detailed outline for blog of topic: ${topic}`;
  const responce = await model.invoke(prompt);
  const outline = JSON.stringify(responce.content);
  return {
    ...state,
    outline,
  };
};

const generateBlog: GraphNode<typeof state> = async (state) => {
  const outline = state.outline;
  const topic = state.topic;
  const prompt = `create a blog using given outline: ${outline} for the topic: ${topic}`;
  const responce = await model.invoke(prompt);
  const blog = JSON.stringify(responce.content);
  return {
    ...state,
    blog,
  };
};

const agent = new StateGraph(state)
  .addNode("generate-outline", generateOutline)
  .addNode("generate-blog", generateBlog)
  .addEdge(START, "generate-outline")
  .addEdge("generate-outline", "generate-blog")
  .addEdge("generate-blog", END)
  .compile();

const runPromptChainingWorkFlow = async () => {
  const result = await agent.invoke({
    topic: "FIFA World Cup 2022",
  });
  console.log(result);
};

export default runPromptChainingWorkFlow
