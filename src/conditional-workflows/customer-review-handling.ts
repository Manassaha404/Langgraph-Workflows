// customer review handling

import { z } from "zod";
import { ChatOpenAI } from "@langchain/openai";
import {
  StateGraph,
  StateSchema,
  START,
  END,
  type ConditionalEdgeRouter,
  type GraphNode,
} from "@langchain/langgraph";
import { ChatPromptTemplate } from "@langchain/core/prompts";
const model = new ChatOpenAI({
  model: "gpt-4o-mini",
});

const reviewHandlingState = new StateSchema({
  customer_review: z.string().trim(),
  review_sentiment: z.enum(["positive", "negative"]).nullable().default(null),
  tone: z.string().nullable().default(null),
  topic: z.string().nullable().default(null),
  urgency: z.string().nullable().default(null),
  response: z.string().trim().nullable().default(null),
});

// review sentiment model
const reviewSentimentModelSchema = z.object({
  review_sentiment: z
    .enum(["positive", "negative"])
    .describe("the sentiment of the review positive or negative"),
});
const reviewSentimentModelPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `You are a customer review sentiment classifier.

Analyze the customer's review and classify its overall sentiment as exactly one of:
- "positive" — satisfaction, praise, approval, or favorable opinion.
- "negative" — dissatisfaction, frustration, criticism, or unfavorable opinion.

Rules:
1. Focus on the overall sentiment.
2. Consider context, sarcasm, and negation.
3. For mixed sentiment, choose the overall dominant sentiment.
4. Return only the structured result matching the provided schema.
5. review_sentiment must be exactly "positive" or "negative".`,
  ],
  ["human", "{review}"],
]);
const reviewSentimentModel = model.withStructuredOutput(
  reviewSentimentModelSchema,
);

const extractReviewSentiment: GraphNode<typeof reviewHandlingState> = async (
  state,
) => {
  const response = await reviewSentimentModelPrompt
    .pipe(reviewSentimentModel)
    .invoke({
      review: state.customer_review,
    });
  return {
    review_sentiment: response.review_sentiment,
  };
};

// review diagnosis
const diagnosisModelSchema = z.object({
  tone: z.string(),
  topic: z.string(),
  urgency: z.string(),
});
const diagnosisModel = model.withStructuredOutput(diagnosisModelSchema);
const diagnosisModelPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `You are a customer complaint analysis assistant.

Analyze the customer's negative review and diagnose the complaint using these three fields:

tone: Describe the emotional tone of the customer, such as angry, frustrated, disappointed, annoyed, dissatisfied, concerned, or upset.
topic: Identify the main issue or subject of the complaint, such as product quality, delivery, customer service, pricing, billing, refund, usability, technical issue, or defective product.
urgency: Determine how urgently the issue should be addressed based on the customer's language and situation. Use values such as low, medium, or high.

Rules:

Focus on the customer's actual complaint.
Identify the primary topic rather than listing every minor issue.
Determine tone from the language and context, not from isolated words.
Determine urgency based on the severity, potential impact, and language used by the customer.
Do not invent information that is not present in the review.
Keep each field concise and descriptive.
Return only the structured output matching the provided schema.

Analyze this negative customer review.`,
  ],
  ["human", "{review}"],
]);
const reviewDiagnosis: GraphNode<typeof reviewHandlingState> = async (
  state,
) => {
  const response = await diagnosisModelPrompt.pipe(diagnosisModel).invoke({
    review: state.customer_review,
  });
  const { tone, topic, urgency } = response;
  return { tone, topic, urgency };
};

const PositiveReviewResponsePrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `You are a customer support assistant responsible for responding to positive customer reviews.

Write a warm, natural, and professional response to the customer's review.

Guidelines:

1. Thank the customer for taking the time to leave a review.
2. Acknowledge the specific positive aspects mentioned in the review.
3. Show genuine appreciation for the customer's support.
4. Keep the response concise and conversational.
5. Avoid generic or overly promotional language.
6. Do not repeat the customer's entire review.
7. Do not make claims or promises that are not supported by the review.
8. Do not use emojis unless they naturally fit the tone.
9. Respond as the company, using "we" or "our" where appropriate.
10. Return only the response text, without headings, explanations, or quotation marks.

Generate a response to this positive customer review:
`,
  ],
  ["human", "{review}"],
]);
const generateResponseForPositiveReview: GraphNode<
  typeof reviewHandlingState
> = async (state) => {
  const response = await PositiveReviewResponsePrompt.pipe(model).invoke({
    review: state.customer_review,
  });
  return { response: JSON.stringify(response.content) };
};

const negativeReviewResponsePrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `You are a customer support assistant responsible for responding to negative customer reviews.

Your task is to generate a professional, empathetic, and helpful response to the customer.

Use the following diagnosis information to understand the complaint:
- tone: The emotional tone of the customer.
- topic: The main issue or complaint.
- urgency: How urgently the issue should be addressed.

Rules:
1. Acknowledge the customer's concern or frustration.
2. Apologize when appropriate.
3. Address the main topic of the complaint directly.
4. Match the response to the customer's tone.
5. Adjust the response based on the urgency.
6. Never blame or argue with the customer.
7. Do not make unsupported promises or guarantees.
8. When appropriate, invite the customer to contact support so the issue can be resolved.
9. Keep the response concise, natural, empathetic, and professional.
10. Do not repeat the customer's entire review.
11. Never mention the diagnosis, tone, topic, or urgency in the response.
12. Return only the customer-facing response.`,
  ],
  [
    "human",
    `Customer review:
{review}

Diagnosis:
Tone: {tone}
Topic: {topic}
Urgency: {urgency}`,
  ],
]);

const generateResponseForNegativeReview: GraphNode<
  typeof reviewHandlingState
> = async (state) => {
  const { customer_review, tone, topic, urgency } = state;
  const response = await negativeReviewResponsePrompt.pipe(model).invoke({
    review: customer_review,
    tone,
    topic,
    urgency,
  });
  return { response: JSON.stringify(response.content) };
};

const checkReviewSentiment: ConditionalEdgeRouter<{
  InputSchema: typeof reviewHandlingState;
  Nodes: "review_diagnosis" | "positive_review_response";
}> = (state) => {
  if (state.review_sentiment == "negative") {
    return "review_diagnosis";
  } else {
    return "positive_review_response";
  }
};

const agent = new StateGraph(reviewHandlingState)
  .addNode("extract_review_sentiment", extractReviewSentiment)
  .addNode("review_diagnosis", reviewDiagnosis)
  .addNode("positive_review_response", generateResponseForPositiveReview)
  .addNode("negative_review_response", generateResponseForNegativeReview)
  .addEdge(START, "extract_review_sentiment")
  .addConditionalEdges("extract_review_sentiment", checkReviewSentiment)
  .addEdge("review_diagnosis", "negative_review_response")
  .addEdge("negative_review_response", END)
  .addEdge("positive_review_response", END)
  .compile();

const runCustomerReviewHandlingAgent = async (review: string) => {
  const initialState = {
    customer_review: review,
  };
  const result = await agent.invoke(initialState);
  return result;
};

export default runCustomerReviewHandlingAgent;
