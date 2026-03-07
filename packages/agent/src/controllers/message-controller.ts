import type {
  CreateMessageQuery,
  DeleteMessageQuery,
  Message,
  MessageQuery,
  MessageType,
  UpdateMessageQuery,
} from "@planning-w-llms/domain";

interface MessageBody {
  id: string;
  type: MessageType;
  text_content: string | null;
  media_content: string | null;
  last_create_ts: string;
  last_update_ts: string;
}

interface MessageMutationBody {
  conversationId: string;
  message: MessageBody;
}

class ValidationError extends Error {}

const MESSAGE_TYPES: MessageType[] = [
  "assistant",
  "developer",
  "system",
  "user",
];

export class MessageController {
  constructor(private readonly messageRepository: MessageQuery) {}

  async create(req: Request): Promise<Response> {
    const query = await this.parseMutationQuery(req);

    if (query instanceof Response) {
      return query;
    }

    const message = await this.messageRepository.createMessage(query);

    return Response.json(this.serializeMessage(message), { status: 201 });
  }

  async show(req: Request): Promise<Response> {
    const url = new URL(req.url);
    const messageId = url.searchParams.get("messageId");

    if (!messageId) {
      return Response.json(
        { error: "messageId query parameter is required." },
        { status: 400 },
      );
    }

    const message = await this.messageRepository.getMessageById(messageId);

    if (!message) {
      return Response.json(
        { error: `Message with id "${messageId}" was not found.` },
        { status: 404 },
      );
    }

    return Response.json(this.serializeMessage(message));
  }

  async showAll(req: Request): Promise<Response> {
    const url = new URL(req.url);
    const conversationId = url.searchParams.get("conversationId");

    if (!conversationId) {
      return Response.json(
        { error: "conversationId query parameter is required." },
        { status: 400 },
      );
    }

    const messages =
      await this.messageRepository.listMessagesByConversationId(
        conversationId,
      );

    return Response.json(
      messages.map((message: Message) => this.serializeMessage(message)),
    );
  }

  async update(req: Request): Promise<Response> {
    const query = await this.parseMutationQuery(req);

    if (query instanceof Response) {
      return query;
    }

    const message = await this.messageRepository.updateMessage(query);

    return Response.json(this.serializeMessage(message));
  }

  async delete(req: Request): Promise<Response> {
    const url = new URL(req.url);
    const messageId = url.searchParams.get("messageId");
    const conversationId = url.searchParams.get("conversationId");

    if (!messageId || !conversationId) {
      return Response.json(
        {
          error:
            "messageId and conversationId query parameters are required.",
        },
        { status: 400 },
      );
    }

    const query: DeleteMessageQuery = {
      conversationId,
      messageId,
    };

    await this.messageRepository.deleteMessage(query);

    return new Response(null, { status: 204 });
  }

  private async parseMutationQuery(
    req: Request,
  ): Promise<CreateMessageQuery | UpdateMessageQuery | Response> {
    try {
      const body = (await req.json()) as unknown;

      return this.parseMutationBody(body);
    } catch (error) {
      if (error instanceof ValidationError) {
        return Response.json({ error: error.message }, { status: 400 });
      }

      if (error instanceof SyntaxError) {
        return Response.json(
          { error: "Request body must be valid JSON." },
          { status: 400 },
        );
      }

      throw error;
    }
  }

  private parseMutationBody(
    body: unknown,
  ): CreateMessageQuery | UpdateMessageQuery {
    if (!this.isRecord(body)) {
      throw new ValidationError("Request body must be a JSON object.");
    }

    const { conversationId, message } = body;

    if (typeof conversationId !== "string" || conversationId.length === 0) {
      throw new ValidationError("conversationId must be a non-empty string.");
    }

    return {
      conversationId,
      message: this.parseMessage(message),
    };
  }

  private parseMessage(message: unknown): Message {
    if (!this.isRecord(message)) {
      throw new ValidationError("message must be a JSON object.");
    }

    const id = this.parseNonEmptyString(message.id, "message.id");
    const type = this.parseMessageType(message.type);
    const text_content = this.parseNullableString(
      message.text_content,
      "message.text_content",
    );
    const media_content = this.parseNullableUrl(
      message.media_content,
      "message.media_content",
    );
    const last_create_ts = this.parseDate(
      message.last_create_ts,
      "message.last_create_ts",
    );
    const last_update_ts = this.parseDate(
      message.last_update_ts,
      "message.last_update_ts",
    );

    return {
      id,
      type,
      text_content,
      media_content,
      last_create_ts,
      last_update_ts,
    };
  }

  private serializeMessage(message: Message) {
    return {
      ...message,
      media_content: message.media_content?.toString() ?? null,
      last_create_ts: message.last_create_ts.toISOString(),
      last_update_ts: message.last_update_ts.toISOString(),
    };
  }

  private parseMessageType(value: unknown): MessageType {
    if (
      typeof value === "string" &&
      MESSAGE_TYPES.includes(value as MessageType)
    ) {
      return value as MessageType;
    }

    throw new ValidationError(
      `message.type must be one of: ${MESSAGE_TYPES.join(", ")}.`,
    );
  }

  private parseNonEmptyString(value: unknown, fieldName: string): string {
    if (typeof value !== "string" || value.length === 0) {
      throw new ValidationError(`${fieldName} must be a non-empty string.`);
    }

    return value;
  }

  private parseNullableString(
    value: unknown,
    fieldName: string,
  ): string | null {
    if (value === null) {
      return null;
    }

    if (typeof value !== "string") {
      throw new ValidationError(`${fieldName} must be a string or null.`);
    }

    return value;
  }

  private parseNullableUrl(value: unknown, fieldName: string): URL | null {
    if (value === null) {
      return null;
    }

    if (typeof value !== "string") {
      throw new ValidationError(`${fieldName} must be a string or null.`);
    }

    try {
      return new URL(value);
    } catch {
      throw new ValidationError(`${fieldName} must be a valid URL.`);
    }
  }

  private parseDate(value: unknown, fieldName: string): Date {
    if (typeof value !== "string") {
      throw new ValidationError(`${fieldName} must be an ISO timestamp.`);
    }

    const date = new Date(value);

    if (Number.isNaN(date.getTime())) {
      throw new ValidationError(`${fieldName} must be a valid ISO timestamp.`);
    }

    return date;
  }

  private isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null;
  }
}
