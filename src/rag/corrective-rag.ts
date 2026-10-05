import "dotenv/config";
import {
  END,
  START,
  StateGraph,
  StateSchema,
  type ConditionalEdgeRouter,
} from "@langchain/langgraph";
import { type GraphNode } from "@langchain/langgraph";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";
import { z } from "zod/v4";
import loadAnyDocument from "./doc-loader.js";
import { tavily } from "@tavily/core";
import { ReducedValue } from "@langchain/langgraph";

// model configuration
const model = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });

// embeddings configuration
const embeddings = new OpenAIEmbeddings({ model: "text-embedding-3-large" });

// tavily configuration
const tvly = tavily({ apiKey: process.env.TAVILY_API_KEY ?? "" });

// vector store configuration
const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
  url: process.env.QDRANT_URL ?? "http://localhost:6333",
  collectionName: "corrective-rag",
});

// text splitter configuration
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

// ingest function
export async function ingestDocument(source: string): Promise<void> {
  const docs = await loadAnyDocument(source);
  const chunks = await splitter.splitDocuments(docs);
  await vectorStore.addDocuments(chunks);
  console.log(`[ingest] Indexed ${chunks.length} chunks from "${source}"`);
}

// state configuration
const RagState = new StateSchema({
  query: z.string().describe("The user question"),
  refineQuery: z.string().describe("The refined user question").nullable().default(null),
  docs: new ReducedValue(z.array(z.instanceof(Document)).default([]), {
    reducer: (existing, update) => existing.concat(update),
  }),
  answer: z.string().default(""),
  evaluator_result: z
    .enum(["correct", "ambiguous", "incorrect"])
    .nullable()
    .default(null),
  good_docs: z.array(z.instanceof(Document)).default([]),
});

// nodes
// retrieve

const stepBackQuery: GraphNode<typeof RagState> = async (state) => {
    const prompt = `You are a query refinement engine. Rewrite the user's message into one improved search query for vector search and web search. Do NOT answer the question.

Current date: {{${new Date().toISOString().split("T")[0]}}}
User query: {{${state.query}}}

Rules:
- Fix typos and expand abbreviations (e.g. "k8s" -> "Kubernetes").
- Add relevant keywords, synonyms, and technical terms that improve retrieval.
- Include the year or version only if the query depends on recent information.
- Keep the user's original intent. Never invent facts, names, or versions.
- If the message is chitchat (e.g. "thanks"), return it unchanged.

Output ONLY the refined query as plain text. No JSON, no quotes, no explanation.

Examples:
Input: how do i make postgres graph faster
Output: how to improve performance of PostgreSQL graph queries Apache AGE indexing optimization large datasets

Input: latest react version features
Output: React latest version new features release notes 2026

Input: hey thanks!
Output: hey thanks!`
const response = await model.invoke(prompt);
  const content = response?.content;
  return { refineQuery: typeof content === "string" ? content : state.query };
}



const retrieve: GraphNode<typeof RagState> = async (state) => {
  const docs = await vectorStore.similaritySearch(state.refineQuery || state.query, 4);
  return { docs };
};

// evaluator
const UPPER_THRESHOLD = 0.7;
const LOWER_THRESHOLD = 0.3;
const evaluatorLLMSchema = z.object({
  score: z
    .number()
    .describe(
      "A score from 0 to 1 indicating how relevant the document is to the question.",
    ),
});
const evaluator: GraphNode<typeof RagState> = async (state) => {
  const good_docs = await Promise.all(
    state.docs.map(async (doc) => {
      const prompt = `You are a relevance scoring assistant. Your task is to evaluate how relevant a retrieved document is to answering a user's question.

Question: ${state.query}

Document:
${doc.pageContent}

Score the document's relevance to the question on a scale from 0.0 to 1.0 using the following criteria:
- 0.0 – 0.29: The document is off-topic or completely unrelated to the question.
- 0.3  – 0.69: The document is loosely related or only partially addresses the question.
- 0.7  – 1.0 : The document is directly relevant and contains information that helps answer the question.

Return only the numeric score.`;
      const score = await model
        .withStructuredOutput(evaluatorLLMSchema)
        .invoke(prompt);
      return { doc, score: score.score };
    }),
  );
  const correct_docs = good_docs
    .filter((d) => d.score >= UPPER_THRESHOLD)
    .map((d) => d.doc);
  const ambiguous_docs = good_docs
    .filter((d) => d.score < UPPER_THRESHOLD && d.score >= LOWER_THRESHOLD)
    .map((d) => d.doc);
  const incorrect_docs = good_docs
    .filter((d) => d.score < LOWER_THRESHOLD)
    .map((d) => d.doc);
  let result: "correct" | "ambiguous" | "incorrect" | null = null;
  if (correct_docs.length > 0) {
    result = "correct";
  } else if (ambiguous_docs.length > 0) {
    result = "ambiguous";
  } else {
    result = "incorrect";
  }
  return {
    evaluator_result: result,
    docs: [...correct_docs, ...ambiguous_docs],
  };
};

const afterEvaluator: ConditionalEdgeRouter<{
  InputSchema: typeof RagState;
  nodes: "refine" | "web-search";
}> = (state) => {
  if (state.evaluator_result === "correct") {
    return "refine";
  } else if (state.evaluator_result === "ambiguous") {
    return "web-search";
  } else {
    return "web-search";
  }
};

const webSearchNode: GraphNode<typeof RagState> = async (state) => {
  try {
    const res = await tvly.search(state.refineQuery || state.query, {
      maxResults: 3,
      searchDepth: "basic",
    });

    const docs = res.results
      .filter((r) => r.content?.trim())
      .map(
        (r) =>
          new Document({
            pageContent: r.content,
            metadata: { source: r.url, title: r.title, score: r.score },
          }),
      );
    return { docs };
  } catch (err) {
    console.error("Web search failed:", err);
    return { docs: [] };
  }
};
const refineLLMSchema = z.object({
  splits: z
    .array(z.string())
    .describe(
      "The splits of the document which are relevant to the question. If none of the splits are relevant, return an empty array.",
    ),
});
const refineDocs: GraphNode<typeof RagState> = async (state) => {
  const results = await Promise.all(
    state.docs.map(async (doc) => {
      const splits = doc.pageContent.split("\n").filter((s) => s.trim());
      const prompt = `You are a relevance filter. Given the user's question and a list of text splits from a document, return ONLY the splits that are directly relevant to answering the question. If none are relevant, return an empty array.

Question: ${state.query}

Document splits:
${splits.map((s) => `${s}`).join("\n")}

Return the relevant splits as-is.`;
      const good_splits = await model
        .withStructuredOutput(refineLLMSchema)
        .invoke(prompt);
      if (good_splits.splits.length !== 0) {
        return new Document({
          pageContent: good_splits.splits.join("\n"),
          metadata: doc.metadata,
        });
      }
      return undefined;
    }),
  );
  const good_docs = results.filter((d): d is Document => d !== undefined);
  return { good_docs };
};

// generate
const generate: GraphNode<typeof RagState> = async (state) => {
  const context = state.docs
    .map((d, i) => `[${i + 1}] ${d.pageContent}`)
    .join("\n\n");

  const response = await model.invoke(
    `Answer the question using ONLY the context below. If the context is insufficient, say so.
    Context:
    ${context}
    Question: ${state.query}
    Answer:`,
  );

  return {
    answer: String(response?.content)?.trim(),
  };
};

const graph = new StateGraph(RagState)
  .addNode("retrieve", retrieve)
  .addNode("generate", generate)
  .addNode("evaluator", evaluator)
  .addNode("web-search", webSearchNode)
  .addNode("refine", refineDocs)
  .addNode("step-back-query", stepBackQuery)
  .addEdge(START, "step-back-query")
  .addEdge("step-back-query", "retrieve")
  .addEdge("retrieve", "evaluator")
  .addConditionalEdges("evaluator", afterEvaluator)
  .addEdge("web-search", "refine")
  .addEdge("refine", "generate")
  .addEdge("generate", END);
export const ragAgent = graph.compile();

export async function ask(
  question: string,
): Promise<Awaited<ReturnType<typeof ragAgent.invoke>>> {
  const result = await ragAgent.invoke({ query: question });
  return result;
}
