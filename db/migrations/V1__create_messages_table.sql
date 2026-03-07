CREATE TABLE IF NOT EXISTS public.messages (
  id UUID PRIMARY KEY,
  conversation_id UUID NOT NULL,
  type TEXT NOT NULL CHECK (type IN ('assistant', 'developer', 'system', 'user')),
  text_content TEXT,
  media_content TEXT,
  last_create_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  last_update_ts TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS messages_conversation_id_idx
  ON public.messages (conversation_id);

CREATE INDEX IF NOT EXISTS messages_last_update_ts_idx
  ON public.messages (last_update_ts DESC);
