/**
 * API route to manually override validation result for a rollout.
 *
 * This is intended for human-in-the-loop correction in the monitor UI.
 * It updates BOTH:
 * - rollout.validation_passed (used by list / summary)
 * - validation.success (used by validation detail panel)
 *
 * Additionally, it records override metadata into validation.details_json.
 */

import { NextResponse } from 'next/server';
import { getDatabase, serializeRow } from '@/lib/db';

type PatchBody = {
  success: boolean;
  reason?: string | null;
  actor?: string | null;
};

function parseDetailsJson(value: unknown): Record<string, unknown> {
  if (!value) return {};
  if (typeof value === 'object') return value as Record<string, unknown>;
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed) return {};
    try {
      const parsed = JSON.parse(trimmed);
      if (parsed && typeof parsed === 'object') return parsed as Record<string, unknown>;
      return {};
    } catch {
      return {};
    }
  }
  return {};
}

export async function PATCH(
  request: Request,
  { params }: { params: Promise<{ id: string }> | { id: string } }
) {
  const resolvedParams = params instanceof Promise ? await params : params;
  const rolloutId = parseInt(resolvedParams.id, 10);
  if (Number.isNaN(rolloutId)) {
    return NextResponse.json({ error: 'Invalid rollout ID' }, { status: 400 });
  }

  let body: PatchBody;
  try {
    body = (await request.json()) as PatchBody;
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }

  if (typeof body?.success !== 'boolean') {
    return NextResponse.json({ error: '`success` must be a boolean' }, { status: 400 });
  }

  const db = getDatabase();
  const client = await db.connect();
  try {
    await client.query('BEGIN');

    // Ensure rollout exists.
    const rolloutRes = await client.query('SELECT * FROM rollout WHERE id = $1', [rolloutId]);
    if (rolloutRes.rows.length === 0) {
      await client.query('ROLLBACK');
      return NextResponse.json({ error: 'Rollout not found' }, { status: 404 });
    }

    // Fetch validation row (may be missing for some rollouts).
    const validationRes = await client.query(
      'SELECT id, rollout_id, success, details_json FROM validation WHERE rollout_id = $1',
      [rolloutId]
    );
    const validationRow = validationRes.rows[0] ?? null;

    // Always update rollout.validation_passed as the "final" status.
    await client.query('UPDATE rollout SET validation_passed = $1 WHERE id = $2', [
      body.success,
      rolloutId,
    ]);

    let updatedValidation: any = null;
    if (validationRow) {
      const prevSuccess = (() => {
        const v = validationRow.success;
        if (typeof v === 'boolean') return v;
        if (typeof v === 'number') return v === 1;
        if (typeof v === 'string') return v === 't' || v === 'true' || v === '1';
        return Boolean(v);
      })();

      const nowIso = new Date().toISOString();
      const details = parseDetailsJson(validationRow.details_json);
      const historyRaw = (details as any).human_override_history;
      const history: unknown[] = Array.isArray(historyRaw) ? historyRaw : [];

      const entry = {
        overridden_at: nowIso,
        overridden_success: body.success,
        previous_success: prevSuccess,
        reason: body.reason ?? null,
        actor: body.actor ?? null,
        source: 'training-monitor',
      };

      (details as any).human_override = entry;
      (details as any).human_override_history = [...history, entry];

      await client.query(
        'UPDATE validation SET success = $1, details_json = $2 WHERE rollout_id = $3 RETURNING *',
        [body.success, JSON.stringify(details), rolloutId]
      );

      const updated = await client.query('SELECT * FROM validation WHERE rollout_id = $1', [rolloutId]);
      updatedValidation = updated.rows[0] ? serializeRow(updated.rows[0]) : null;
    }

    await client.query('COMMIT');

    const updatedRollout = await client.query('SELECT * FROM rollout WHERE id = $1', [rolloutId]);

    return NextResponse.json({
      rollout: updatedRollout.rows[0] ? serializeRow(updatedRollout.rows[0]) : null,
      validation: updatedValidation,
      validation_exists: Boolean(validationRow),
    });
  } catch (error: any) {
    try {
      await client.query('ROLLBACK');
    } catch {
      // ignore
    }
    console.error('Error overriding validation:', error);
    return NextResponse.json(
      { error: error?.message || 'Failed to override validation' },
      { status: 500 }
    );
  } finally {
    client.release();
  }
}

