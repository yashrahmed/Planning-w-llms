import type {
  CreateMessageQuery,
  DeleteMessageQuery,
  Message,
  MessageId,
  MessageQuery,
  UpdateMessageQuery,
  ConversationId,
} from "@planning-w-llms/domain";
import { Pool } from "pg";

interface MessageRow {
  id: string;
  conversation_id: string;
  type: Message["type"];
  text_content: string | null;
  media_content: string | null;
  last_create_ts: Date;
  last_update_ts: Date;
}

interface ConvStoreDatabaseConfig {
  host: string;
  port: number;
  database: string;
  user: string;
  password: string;
  ssl: boolean;
}

function requireEnv(name: string): string {
  const value = process.env[name];

  if (!value) {
    throw new Error(`${name} is required.`);
  }

  return value;
}

function getConvStoreDatabaseConfig(): ConvStoreDatabaseConfig {
  return {
    host: requireEnv("CONV_STORE_DB_HOST"),
    port: Number(requireEnv("CONV_STORE_DB_PORT")),
    database: requireEnv("CONV_STORE_DB_NAME"),
    user: requireEnv("CONV_STORE_DB_USER"),
    password: requireEnv("CONV_STORE_DB_PASSWORD"),
    ssl: process.env.CONV_STORE_DB_SSL === "true",
  };
}

export class MessageRepository implements MessageQuery {
  private readonly pool: Pool;

  constructor(databaseConfig = getConvStoreDatabaseConfig()) {
    if (Number.isNaN(databaseConfig.port)) {
      throw new Error("CONV_STORE_DB_PORT must be a valid number.");
    }

    this.pool = new Pool({
      host: databaseConfig.host,
      port: databaseConfig.port,
      database: databaseConfig.database,
      user: databaseConfig.user,
      password: databaseConfig.password,
      ssl: databaseConfig.ssl,
    });
  }

  async createMessage(input: CreateMessageQuery): Promise<Message> {
    const { conversationId, message } = input;
    const result = await this.pool.query<MessageRow>(
      `
        INSERT INTO public.messages (
          id,
          conversation_id,
          type,
          text_content,
          media_content,
          last_create_ts,
          last_update_ts
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        RETURNING
          id,
          conversation_id,
          type,
          text_content,
          media_content,
          last_create_ts,
          last_update_ts
      `,
      [
        message.id,
        conversationId,
        message.type,
        message.text_content,
        message.media_content?.toString() ?? null,
        message.last_create_ts,
        message.last_update_ts,
      ],
    );

    const row = result.rows[0];

    if (!row) {
      throw new Error("Failed to create message.");
    }

    return this.mapRowToMessage(row);
  }

  async getMessageById(messageId: MessageId): Promise<Message | null> {
    const result = await this.pool.query<MessageRow>(
      `
        SELECT
          id,
          conversation_id,
          type,
          text_content,
          media_content,
          last_create_ts,
          last_update_ts
        FROM public.messages
        WHERE id = $1
      `,
      [messageId],
    );

    if (result.rows.length === 0) {
      return null;
    }

    const row = result.rows[0];

    if (!row) {
      return null;
    }

    return this.mapRowToMessage(row);
  }

  async listMessagesByConversationId(
    conversationId: ConversationId,
  ): Promise<Message[]> {
    const result = await this.pool.query<MessageRow>(
      `
        SELECT
          id,
          conversation_id,
          type,
          text_content,
          media_content,
          last_create_ts,
          last_update_ts
        FROM public.messages
        WHERE conversation_id = $1
        ORDER BY last_create_ts ASC
      `,
      [conversationId],
    );

    return result.rows.map((row: MessageRow) => this.mapRowToMessage(row));
  }

  async updateMessage(input: UpdateMessageQuery): Promise<Message> {
    const { conversationId, message } = input;
    const result = await this.pool.query<MessageRow>(
      `
        UPDATE public.messages
        SET
          type = $3,
          text_content = $4,
          media_content = $5,
          last_update_ts = $6
        WHERE id = $1 AND conversation_id = $2
        RETURNING
          id,
          conversation_id,
          type,
          text_content,
          media_content,
          last_create_ts,
          last_update_ts
      `,
      [
        message.id,
        conversationId,
        message.type,
        message.text_content,
        message.media_content?.toString() ?? null,
        message.last_update_ts,
      ],
    );

    if (result.rows.length === 0) {
      throw new Error(`Message with id "${message.id}" does not exist.`);
    }

    const row = result.rows[0];

    if (!row) {
      throw new Error(`Message with id "${message.id}" does not exist.`);
    }

    return this.mapRowToMessage(row);
  }

  async deleteMessage(input: DeleteMessageQuery): Promise<void> {
    const { conversationId, messageId } = input;
    await this.pool.query(
      `
        DELETE FROM public.messages
        WHERE id = $1 AND conversation_id = $2
      `,
      [messageId, conversationId],
    );
  }

  private mapRowToMessage(row: MessageRow): Message {
    return {
      id: row.id,
      type: row.type,
      text_content: row.text_content,
      media_content: row.media_content ? new URL(row.media_content) : null,
      last_create_ts: new Date(row.last_create_ts),
      last_update_ts: new Date(row.last_update_ts),
    };
  }
}
