import type {
  Conversation,
  ConversationQuery,
  CreateConversationQuery,
  DeleteConversationQuery,
  UpdateConversationQuery,
} from "@planning-w-llms/domain";

interface ConversationBody {
  id: string;
  last_create_ts: string;
  last_update_ts: string;
}

interface ConversationMutationBody {
  conversation: ConversationBody;
}

class ValidationError extends Error {}

export class ConversationController {
  constructor(private readonly conversationRepository: ConversationQuery) {}

  async create(req: Request): Promise<Response> {
    const query = await this.parseMutationQuery(req);

    if (query instanceof Response) {
      return query;
    }

    const conversation =
      await this.conversationRepository.createConversation(query);

    return Response.json(this.serializeConversation(conversation), {
      status: 201,
    });
  }

  async show(req: Request): Promise<Response> {
    const url = new URL(req.url);
    const conversationId = url.searchParams.get("conversationId");

    if (!conversationId) {
      return Response.json(
        { error: "conversationId query parameter is required." },
        { status: 400 },
      );
    }

    const conversation =
      await this.conversationRepository.getConversationById(conversationId);

    if (!conversation) {
      return Response.json(
        { error: `Conversation with id "${conversationId}" was not found.` },
        { status: 404 },
      );
    }

    return Response.json(this.serializeConversation(conversation));
  }

  async showAll(): Promise<Response> {
    const conversations =
      await this.conversationRepository.listConversations();

    return Response.json(
      conversations.map((conversation: Conversation) =>
        this.serializeConversation(conversation),
      ),
    );
  }

  async update(req: Request): Promise<Response> {
    const query = await this.parseMutationQuery(req);

    if (query instanceof Response) {
      return query;
    }

    try {
      const conversation =
        await this.conversationRepository.updateConversation(query);

      return Response.json(this.serializeConversation(conversation));
    } catch (error) {
      if (
        error instanceof Error &&
        error.message.includes("does not exist")
      ) {
        return Response.json({ error: error.message }, { status: 404 });
      }

      throw error;
    }
  }

  async delete(req: Request): Promise<Response> {
    const url = new URL(req.url);
    const conversationId = url.searchParams.get("conversationId");

    if (!conversationId) {
      return Response.json(
        { error: "conversationId query parameter is required." },
        { status: 400 },
      );
    }

    const query: DeleteConversationQuery = {
      conversationId,
    };

    await this.conversationRepository.deleteConversation(query);

    return new Response(null, { status: 204 });
  }

  private async parseMutationQuery(
    req: Request,
  ): Promise<CreateConversationQuery | UpdateConversationQuery | Response> {
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
  ): CreateConversationQuery | UpdateConversationQuery {
    if (!this.isRecord(body)) {
      throw new ValidationError("Request body must be a JSON object.");
    }

    return {
      conversation: this.parseConversation(body.conversation),
    };
  }

  private parseConversation(conversation: unknown): Conversation {
    if (!this.isRecord(conversation)) {
      throw new ValidationError("conversation must be a JSON object.");
    }

    return {
      id: this.parseNonEmptyString(conversation.id, "conversation.id"),
      messages: [],
      last_create_ts: this.parseDate(
        conversation.last_create_ts,
        "conversation.last_create_ts",
      ),
      last_update_ts: this.parseDate(
        conversation.last_update_ts,
        "conversation.last_update_ts",
      ),
    };
  }

  private serializeConversation(conversation: Conversation) {
    return {
      id: conversation.id,
      last_create_ts: conversation.last_create_ts.toISOString(),
      last_update_ts: conversation.last_update_ts.toISOString(),
    };
  }

  private parseNonEmptyString(value: unknown, fieldName: string): string {
    if (typeof value !== "string" || value.length === 0) {
      throw new ValidationError(`${fieldName} must be a non-empty string.`);
    }

    return value;
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
