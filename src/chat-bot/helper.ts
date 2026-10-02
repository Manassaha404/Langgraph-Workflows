// convert message content to plain string
export const textOf = (content: unknown): string =>
  typeof content === "string" ? content : JSON.stringify(content);

// find latest message of given type (human or ai)
export const lastOfType = (messages: any[], type: "human" | "ai") =>
  [...messages].reverse().find((m) => m.type === type);

// check if any tool failed after the latest user message
export const lastTurnHadToolError = (messages: any[]): boolean => {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.type === "human") break;
    if (m.type !== "tool") continue;
    if (m.status === "error") return true;
    const c = textOf(m.content);
    if (/"successful"\s*:\s*false/i.test(c) || /^error/i.test(c.trim())) {
      return true;
    }
  }
  return false;
};