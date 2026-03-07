export type ConversationId = string;
export type MessageId = string;

// Markdown source is the canonical message text format.
export type MarkdownTextContent = string;

// Aligned with the current OpenAI conversation message roles.
export type MessageType = "assistant" | "developer" | "system" | "user";

export interface Message {
  id: MessageId;
  type: MessageType;
  text_content: MarkdownTextContent | null;
  media_content: URL | null;
  last_create_ts: Date;
  last_update_ts: Date;
}

export interface Conversation {
  id: ConversationId;
  messages: Message[];
  last_create_ts: Date;
  last_update_ts: Date;
}

export * from "./conversation-query.js";
export * from "./message-query.js";
