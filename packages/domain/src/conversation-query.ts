import type { Conversation, ConversationId } from "./index.js";

export interface CreateConversationQuery {
  conversation: Conversation;
}

export interface UpdateConversationQuery {
  conversation: Conversation;
}

export interface DeleteConversationQuery {
  conversationId: ConversationId;
}

export interface ConversationQuery {
  createConversation(input: CreateConversationQuery): Promise<Conversation>;
  getConversationById(
    conversationId: ConversationId,
  ): Promise<Conversation | null>;
  listConversations(): Promise<Conversation[]>;
  updateConversation(input: UpdateConversationQuery): Promise<Conversation>;
  deleteConversation(input: DeleteConversationQuery): Promise<void>;
}
