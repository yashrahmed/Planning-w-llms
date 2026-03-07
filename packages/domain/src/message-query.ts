import type {
  ConversationId,
  Message,
  MessageId,
} from "./index.js";

export interface CreateMessageQuery {
  conversationId: ConversationId;
  message: Message;
}

export interface UpdateMessageQuery {
  conversationId: ConversationId;
  message: Message;
}

export interface DeleteMessageQuery {
  conversationId: ConversationId;
  messageId: MessageId;
}

export interface MessageQuery {
  createMessage(input: CreateMessageQuery): Promise<Message>;
  getMessageById(messageId: MessageId): Promise<Message | null>;
  listMessagesByConversationId(
    conversationId: ConversationId,
  ): Promise<Message[]>;
  updateMessage(input: UpdateMessageQuery): Promise<Message>;
  deleteMessage(input: DeleteMessageQuery): Promise<void>;
}
