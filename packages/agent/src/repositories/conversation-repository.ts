import type {
  Conversation,
  ConversationId,
  ConversationQuery,
  CreateConversationQuery,
  DeleteConversationQuery,
  UpdateConversationQuery,
} from "@planning-w-llms/domain";
import { Pool } from "pg";

interface ConversationRow {
  id: string;
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

export class ConversationRepository implements ConversationQuery {
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

  async createConversation(
    input: CreateConversationQuery,
  ): Promise<Conversation> {
    const { conversation } = input;
    const result = await this.pool.query<ConversationRow>(
      `
        INSERT INTO public.conversations (
          id,
          last_create_ts,
          last_update_ts
        )
        VALUES ($1, $2, $3)
        RETURNING
          id,
          last_create_ts,
          last_update_ts
      `,
      [
        conversation.id,
        conversation.last_create_ts,
        conversation.last_update_ts,
      ],
    );

    const row = result.rows[0];

    if (!row) {
      throw new Error("Failed to create conversation.");
    }

    return this.mapRowToConversation(row);
  }

  async getConversationById(
    conversationId: ConversationId,
  ): Promise<Conversation | null> {
    const result = await this.pool.query<ConversationRow>(
      `
        SELECT
          id,
          last_create_ts,
          last_update_ts
        FROM public.conversations
        WHERE id = $1
      `,
      [conversationId],
    );

    if (result.rows.length === 0) {
      return null;
    }

    const row = result.rows[0];

    if (!row) {
      return null;
    }

    return this.mapRowToConversation(row);
  }

  async listConversations(): Promise<Conversation[]> {
    const result = await this.pool.query<ConversationRow>(
      `
        SELECT
          id,
          last_create_ts,
          last_update_ts
        FROM public.conversations
        ORDER BY last_update_ts DESC
      `,
    );

    return result.rows.map((row: ConversationRow) =>
      this.mapRowToConversation(row),
    );
  }

  async updateConversation(
    input: UpdateConversationQuery,
  ): Promise<Conversation> {
    const { conversation } = input;
    const result = await this.pool.query<ConversationRow>(
      `
        UPDATE public.conversations
        SET
          last_update_ts = $2
        WHERE id = $1
        RETURNING
          id,
          last_create_ts,
          last_update_ts
      `,
      [conversation.id, conversation.last_update_ts],
    );

    if (result.rows.length === 0) {
      throw new Error(
        `Conversation with id "${conversation.id}" does not exist.`,
      );
    }

    const row = result.rows[0];

    if (!row) {
      throw new Error(
        `Conversation with id "${conversation.id}" does not exist.`,
      );
    }

    return this.mapRowToConversation(row);
  }

  async deleteConversation(input: DeleteConversationQuery): Promise<void> {
    const { conversationId } = input;
    await this.pool.query(
      `
        DELETE FROM public.conversations
        WHERE id = $1
      `,
      [conversationId],
    );
  }

  private mapRowToConversation(row: ConversationRow): Conversation {
    return {
      id: row.id,
      messages: [],
      last_create_ts: new Date(row.last_create_ts),
      last_update_ts: new Date(row.last_update_ts),
    };
  }
}
