import { ConversationController } from "./controllers/conversation-controller.js";
import { ConversationRepository } from "./repositories/conversation-repository.js";
import { MessageController } from "./controllers/message-controller.js";
import { MessageRepository } from "./repositories/message-repository.js";

const DEFAULT_AGENT_PORT = 3001;

const conversationRepository = new ConversationRepository();
const conversationController = new ConversationController(
  conversationRepository,
);
const messageRepository = new MessageRepository();
const messageController = new MessageController(messageRepository);

const port = Number(process.env.CONV_AGENT_PORT ?? DEFAULT_AGENT_PORT);

const server = Bun.serve({
  port,
  async fetch(req: Request) {
    const url = new URL(req.url);

    if (url.pathname === "/health") {
      return Response.json({ status: "ok", service: "agent" });
    }

    if (url.pathname === "/messages") {
      if (req.method === "POST") {
        return messageController.create(req);
      }

      if (req.method === "GET") {
        return url.searchParams.has("messageId")
          ? messageController.show(req)
          : messageController.showAll(req);
      }

      if (req.method === "PUT") {
        return messageController.update(req);
      }

      if (req.method === "DELETE") {
        return messageController.delete(req);
      }

      return Response.json(
        { error: `Method ${req.method} is not supported on /messages.` },
        { status: 405 },
      );
    }

    if (url.pathname === "/conversations") {
      if (req.method === "POST") {
        return conversationController.create(req);
      }

      if (req.method === "GET") {
        return url.searchParams.has("conversationId")
          ? conversationController.show(req)
          : conversationController.showAll();
      }

      if (req.method === "PUT") {
        return conversationController.update(req);
      }

      if (req.method === "DELETE") {
        return conversationController.delete(req);
      }

      return Response.json(
        { error: `Method ${req.method} is not supported on /conversations.` },
        { status: 405 },
      );
    }

    return Response.json({ name: "planning-w-llms-agent", version: "0.1.0" });
  },
});

console.log(
  `Planning-w-llms agent running at http://localhost:${server.port}`,
);
