import { z } from "zod";
import { ChatOpenAI } from "@langchain/openai";
import {
  StateGraph,
  StateSchema,
  START,
  END,
  type GraphNode,
} from "@langchain/langgraph";
import { ChatPromptTemplate } from "@langchain/core/prompts";
// state
const xPostGeneratorState = new StateSchema({
  topic: z.string(),
  tweet: z.string().nullable().default(null),
  evaluateResult: z
    .enum(["approved", "need-optimization"])
    .nullable()
    .default(null),
  feedback: z.string().nullable().default(null),
  iteration: z.number().default(1),
  maxIteration: z.number().default(5),
});

// llm models
const generatorModel = new ChatOpenAI({
  model: "gpt-4o-mini",
}).withStructuredOutput(
  z.object({
    tweet: z.string().describe("generate tweet"),
  }),
);

const evaluationModelStructureOutputSchema = z.object({
  feedback: z.string().describe("the feedback of the tweet after evaluation"),
  evaluateResult: z.enum(["approved", "need-optimization"]),
});

const evaluationModel = new ChatOpenAI({
  model: "gpt-4o-mini",
}).withStructuredOutput(evaluationModelStructureOutputSchema);

// generation node
const generateTweet: GraphNode<typeof xPostGeneratorState> = async (state) => {
  const twitterPostPrompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `You are a developer-focused Twitter/X content writer.

Generate a Twitter/X post from the user's input.

Writing style:
- Authentic developer voice
- Concise and punchy
- Conversational, not corporate
- Slightly personal when the input describes an experience
- Technical terms are allowed
- Prefer simple words over complicated vocabulary
- Use line breaks to improve readability
- Make the first line strong enough to make someone stop scrolling
- Focus on one clear idea
- Avoid fake enthusiasm and motivational clichés
- Avoid phrases like "In today's fast-paced world", "I'm thrilled to announce", etc.
- Never fabricate technical details or personal experiences
- Don't overuse emojis or hashtags
- Use 0–2 relevant hashtags only when they genuinely add value

Structure:
Hook
↓
What happened / What I built
↓
What I learned or realized
↓
Strong closing line

Return ONLY the final Twitter/X post.`,
    ],
    ["human", `{input}`],
  ]);
  const response = await twitterPostPrompt.pipe(generatorModel).invoke({
    input: state.topic,
  });

  return { tweet: response.tweet };
};

// evaluation node
const evaluateTweet: GraphNode<typeof xPostGeneratorState> = async (state) => {
  const evaluationPrompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `You are an expert Twitter/X content editor and evaluator.

Your task is to evaluate a generated Twitter/X post based on the original topic and determine whether the post is ready to publish or needs optimization.

Evaluate the tweet using these criteria:

1. Topic Relevance:
   - Does the tweet accurately represent the given topic?
   - Does it stay focused on the topic?
   - Does it avoid adding unrelated information?

2. Hook:
   - Does the first line grab attention?
   - Is it interesting enough to make someone continue reading?

3. Clarity:
   - Is the main idea immediately understandable?
   - Is the writing easy to read?

4. Authenticity:
   - Does it sound like a real person wrote it?
   - Does it avoid generic, robotic, or AI-generated language?
   - Does it preserve a natural personal voice?

5. Value:
   - Does it provide an insight, experience, useful information, or interesting perspective?
   - Does every sentence contribute something meaningful?

6. Engagement:
   - Does it naturally encourage the reader to think, relate, or respond?
   - Do not require a question or engagement bait if it doesn't fit the topic.

7. Writing Quality:
   - Check grammar, spelling, sentence structure, and readability.
   - Avoid unnecessary repetition and filler.

8. Technical Accuracy:
   - If the tweet contains technical information, check whether the claims are logically and technically accurate.
   - Do not approve misleading or fabricated information.

9. Twitter/X Style:
   - Keep it concise and readable.
   - Use appropriate line breaks.
   - Avoid excessive emojis, hashtags, hype, and corporate language.
   - Avoid generic motivational clichés.

Decision rules:
- Return "approved" only when the tweet is strong enough to publish with little or no modification.
- Return "need-optimization" when there is a meaningful issue that should be fixed.
- Do not reject a tweet simply because it is short or does not contain hashtags.
- Do not rewrite the tweet.
- Feedback must be specific and actionable.
- If approved, briefly explain why the tweet works.
- If optimization is needed, clearly identify the problems and what should be improved.

Original topic:
{topic}

Now evaluate the generated tweet provided by the user message.`,
    ],
    ["ai", "{tweet}"],
  ]);
  const response = await evaluationPrompt.pipe(evaluationModel).invoke({
    topic: state.topic,
    tweet: state.tweet,
  });
  return {
    feedback: response.feedback,
    evaluateResult: response.evaluateResult,
  };
};

// optimization node
const optimizationTweet: GraphNode<typeof xPostGeneratorState> = async (
  state,
) => {
  const { topic, tweet, feedback, iteration } = state;
  const optimizationPrompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `You are an expert Twitter/X content editor.

Your task is to optimize a generated Twitter/X post using the original topic and evaluator feedback.

Your goals:
- Fix every meaningful issue mentioned in the feedback.
- Keep the original meaning and core idea.
- Stay faithful to the given topic.
- Preserve the author's personality and authentic voice.
- Make the post natural, concise, and engaging.
- Improve the hook when necessary.
- Improve clarity, readability, grammar, and structure.
- Remove unnecessary words, repetition, filler, and generic statements.
- Make technical claims accurate without inventing information.
- Use line breaks when they improve readability.
- Avoid excessive emojis and hashtags.
- Avoid corporate, robotic, or obviously AI-generated language.
- Do not add facts, experiences, achievements, or opinions that are not supported by the original tweet or topic.
- Do not change the topic just to make the tweet more engaging.

Important:
- Treat the evaluator feedback as instructions for improvement, not as content that must be copied into the tweet.
- If the feedback suggests a problem that does not actually exist, use your own judgment and do not make unnecessary changes.
- Do not explain your changes.
- Return ONLY the final optimized Twitter/X post.
- Do not wrap the tweet in quotation marks.
- Do not add labels such as "Optimized Tweet:".

Original topic:
{topic}

Evaluator feedback:
{feedback}

Now optimize the following tweet:
`,
    ],
    ["human", "{tweet}"],
  ]);
  const response = await optimizationPrompt.pipe(generatorModel).invoke({
    topic,
    tweet,
    feedback,
  });
  return {
    tweet: response.tweet,
    iteration: iteration + 1,
  };
};

//condition check func
const conditionCheck = (state:any) => {
  if (
    state.evaluateResult === "approved" ||
    state.iteration === state.maxIteration
  ) {
    return "approved";
  } else {
    return "need-optimization";
  }
};

const agent = new StateGraph(xPostGeneratorState)
  .addNode("generate-tweet", generateTweet)
  .addNode("evaluate-tweet", evaluateTweet)
  .addNode("optimize-tweet", optimizationTweet)
  .addEdge(START, "generate-tweet")
  .addEdge("generate-tweet", "evaluate-tweet")
  .addConditionalEdges("evaluate-tweet", conditionCheck, {
    approved: END,
    "need-optimization": "optimize-tweet",
  })
  .addEdge("optimize-tweet", "evaluate-tweet")
  .compile();

const generateXPost = async (topic: string, maxIteration: number = 5): Promise<z.infer<typeof xPostGeneratorState>> => {
  const initialState = {
    topic,
    maxIteration,
  };
  const result = await agent.invoke(initialState);
  return result;
};

export default generateXPost
