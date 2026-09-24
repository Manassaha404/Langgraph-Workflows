// essay feedback workflow
import { z } from "zod";
import { ChatOpenAI } from "@langchain/openai";
import {
  StateGraph,
  StateSchema,
  START,
  END,
  type GraphNode,
} from "@langchain/langgraph";
import { ReducedValue } from "@langchain/langgraph";
const model = new ChatOpenAI({
  model: "gpt-4o-mini",
});

const schema = z.object({
  feedback: z
    .string()
    .describe("the feedback of the given essay according the given criteria"),
  score: z
    .number()
    .describe(
      "the score of the given essay out of 10, according the given criteria",
    ),
});

const structured_model = model.withStructuredOutput(schema);

const stateOfEssayFeedback = new StateSchema({
  essay: z.string(),
  analysis_feedback: z.string().nullable().default(null),
  depth_of_thought_feedback: z.string().nullable().default(null),
  language_feedback: z.string().nullable().default(null),
  score: new ReducedValue(z.array(z.number()).default([]), {
    reducer: (existing, update) => existing.concat(update),
  }),
  overall_feedback: z.string().nullable().default(null),
  average_score: z.number().nullable().default(null),
});



const depthOfAnalysisLLMCall: GraphNode<typeof stateOfEssayFeedback> = async (
  state,
) => {
  const prompt = `Evaluate the following essay purely on depth of analysis — not grammar, structure, or style. Depth of analysis means: does the writer go beyond surface-level observations to explore causes, implications, tensions, counterarguments, and nuanced connections between ideas? Score the essay from 1–10 based on this rubric:

1–3: Surface-level, mostly summary or restated facts, no real reasoning
4–6: Some analysis present, but shallow or one-sided; misses obvious counterpoints or deeper implications
7–8: Solid analysis with clear reasoning, some exploration of nuance or counterarguments
9–10: Exceptional depth — original insight, engages with complexity, anticipates objections, connects ideas across multiple angles

Give:

A score (X/10)
One specific suggestion and feedback 

Essay:
${state.essay}`;
  const response = await structured_model.invoke(prompt);

  return {
    analysis_feedback: response.feedback,
    score: [Number(response.score)],
  };
};

const depthOfThoughtLLMCall: GraphNode<typeof stateOfEssayFeedback> = async (
  state,
) => {
  const prompt = `Evaluate the following essay purely on depth of thought — not grammar, structure, or style. Depth of thought means: does the writer show genuine independent thinking — questioning assumptions, reasoning through implications, sitting with ambiguity, and arriving at ideas that feel earned rather than borrowed or obvious?

Score the essay from 1–10 based on this rubric:

1–3: No real thinking on display — clichés, received opinions, or ideas copied without reflection
4–6: Some genuine thought present, but predictable or safe; rarely questions its own assumptions
7–8: Clear evidence of independent reasoning; grapples with ambiguity or tension in the topic
9–10: Exceptional originality — surprising angles, self-aware reasoning, ideas that feel genuinely thought-through rather than assembled

Give:

A score (X/10)
One specific suggestion and feedback

Essay:
${state.essay}`;
  const response = await structured_model.invoke(prompt);
  return {
    depth_of_thought_feedback: response.feedback,
    score: [Number(response.score)],
  };
};

const languageLLMCall: GraphNode<typeof stateOfEssayFeedback> = async (
  state,
) => {
  const prompt = `Evaluate the following essay purely on language — not depth of analysis, structure, or argument quality. Language means: word choice, sentence variety, clarity, tone control, and the writer's command of expression (precision, fluency, and voice).

Score the essay from 1–10 based on this rubric:

1–3: Weak vocabulary, repetitive or awkward sentences, unclear phrasing throughout
4–6: Functional but plain language; limited variety, occasional awkward or imprecise wording
7–8: Clear, controlled language with good sentence variety and mostly precise word choice
9–10: Exceptional command of language — vivid, precise, varied, and stylistically confident throughout

Give:

A score (X/10)
One specific suggestion and feedback

Essay:
${state.essay}`;

  const response = await structured_model.invoke(prompt);

  return {
    language_feedback: response.feedback,
    score: [Number(response.score)],
  };
};

const finalEvaluationLLMCall: GraphNode<typeof stateOfEssayFeedback> = async (state) => {
  const prompt = `Generate a concise overall feedback summary for this essay, synthesizing the three criteria feedback below into a single, cohesive assessment. Highlight the essay's key strengths, its main areas for improvement, and one clear, actionable next step the writer should focus on.

Language feedback:
${state.language_feedback}

Depth of Analysis feedback:
${state.analysis_feedback}

Depth of Thought feedback:
${state.depth_of_thought_feedback} 

  and give average score: ${JSON.stringify(state.score)}
  `;
  const response = await structured_model.invoke(prompt);

  return {
    overall_feedback: response.feedback,
    average_score: Number(response.score),
  };
};


const agent = new StateGraph(stateOfEssayFeedback)
.addNode("analysis_feedback_llm", depthOfAnalysisLLMCall)
.addNode("depth_of_thought_llm", depthOfThoughtLLMCall)
.addNode("language_llm", languageLLMCall)
.addNode("final_evaluation_llm", finalEvaluationLLMCall)
.addEdge(START, "analysis_feedback_llm")
.addEdge(START, "depth_of_thought_llm")
.addEdge(START, "language_llm")
.addEdge("analysis_feedback_llm", "final_evaluation_llm")
.addEdge("depth_of_thought_llm", "final_evaluation_llm")
.addEdge("language_llm", "final_evaluation_llm")
.addEdge("final_evaluation_llm", END)
.compile() 

const runEssayFeedBackAgent = async(essay:string) => {
  const initialState = {
    essay
  }
  const result = await agent.invoke(initialState) 
  return result
}

export default runEssayFeedBackAgent;

