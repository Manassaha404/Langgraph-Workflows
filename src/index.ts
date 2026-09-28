import "dotenv/config";

// import runSimpleWorkflow from "./linear-workflows/simple-llm-workflow.js";
// await runSimpleWorkflow()


// import runPromptChainingWorkFlow from "./linear-workflows/simple-prompt-chaining-workflow.js"
// await runPromptChainingWorkFlow() 

// import runEssayFeedBackAgent from "./parallel-workflows/essay-feedback-workflow.js";
// const result = await runEssayFeedBackAgent(`Title: The Illusion of Choice in the Digital Age

// We live in an era that prides itself on unprecedented freedom of choice. Streaming platforms offer thousands of shows, social media connects us to millions of voices, and e-commerce puts nearly any product a click away. Yet beneath this abundance lies a quieter truth: the choices we make are increasingly shaped by algorithms designed not to serve our interests, but to hold our attention.

// Recommendation systems learn our habits and feed us more of what we already like, narrowing our exposure even as they appear to expand it. A person who watches one true-crime documentary is soon offered dozens more, while genres outside that pattern quietly disappear from view. This creates a paradox: we feel we are choosing freely, but the menu itself has been curated by forces optimizing for engagement, not enrichment.

// Some argue this is no different from traditional media, which has always filtered what reaches audiences. But there is a meaningful distinction. Editors and broadcasters, for all their flaws, operated with some accountability to public standards. Algorithms optimize for a single, measurable goal — time spent — with little regard for diversity of thought or long-term wellbeing. The result is not censorship in the traditional sense, but a subtler narrowing, one we mistake for personal preference because it feels tailored to us.

// The danger, then, is not that we have too few choices, but that we no longer notice how our choices are being made for us. Reclaiming genuine autonomy may require less trust in convenience and more deliberate effort to seek out what the algorithm would not have shown us.`)

// console.log(result);


// import runCustomerReviewHandlingAgent from "./conditional-workflows/customer-review-handling.js";
// const result = await runCustomerReviewHandlingAgent("I absolutely love this product! The quality is excellent, it arrived earlier than expected, and everything worked perfectly right out of the box. The whole experience was smooth and hassle-free. I’ll definitely buy from you again.")

// const result1 = await runCustomerReviewHandlingAgent("I’m really disappointed with my purchase. The product arrived three days late and had scratches on it. I contacted customer support twice, but nobody responded. I expected much better service and would like this issue resolved as soon as possible.")
// console.log(result1);

// import generateXPost from "./looping-workflow/x-post-generator.js";
// const result = await generateXPost("I learned that knowing React isn't enough to become a good frontend developer. You also need to understand browser fundamentals, HTTP, accessibility, performance, and how the web actually works.")
// console.log(result) 

import startChatBot from "./chat-bot/chat-bot-workflow.js";
startChatBot() 
