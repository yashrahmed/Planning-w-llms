CREATE TABLE IF NOT EXISTS public.conversations (
  id UUID PRIMARY KEY,
  last_create_ts TIMESTAMPTZ NOT NULL,
  last_update_ts TIMESTAMPTZ NOT NULL
);

INSERT INTO public.conversations (
  id,
  last_create_ts,
  last_update_ts
)
SELECT
  m.conversation_id,
  MIN(m.last_create_ts),
  MAX(m.last_update_ts)
FROM public.messages AS m
LEFT JOIN public.conversations AS c
  ON c.id = m.conversation_id
WHERE c.id IS NULL
GROUP BY m.conversation_id;

CREATE INDEX IF NOT EXISTS conversations_last_update_ts_idx
  ON public.conversations (last_update_ts DESC);

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'messages_conversation_id_fkey'
  ) THEN
    ALTER TABLE public.messages
    ADD CONSTRAINT messages_conversation_id_fkey
      FOREIGN KEY (conversation_id)
      REFERENCES public.conversations (id)
      ON DELETE CASCADE;
  END IF;
END $$;
