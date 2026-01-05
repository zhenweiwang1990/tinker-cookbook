/**
 * API route to get details of a specific turn (lazy loading).
 * This endpoint loads actions and observations only when needed.
 */

import { NextResponse } from 'next/server';
import { query, serializeRow } from '@/lib/db';

export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string; turnId: string }> | { id: string; turnId: string } }
) {
  try {
    const resolvedParams = params instanceof Promise ? await params : params;
    const rolloutId = parseInt(resolvedParams.id);
    const turnId = parseInt(resolvedParams.turnId);

    if (isNaN(rolloutId) || isNaN(turnId)) {
      return NextResponse.json(
        { error: 'Invalid rollout ID or turn ID' },
        { status: 400 }
      );
    }

    // Verify turn belongs to rollout
    const turnResult = await query('SELECT id FROM turn WHERE id = $1 AND rollout_id = $2', [turnId, rolloutId]);
    if (turnResult.rows.length === 0) {
      return NextResponse.json(
        { error: 'Turn not found or does not belong to this rollout' },
        { status: 404 }
      );
    }

    // Get actions
    const actionsResult = await query(
      'SELECT * FROM action WHERE turn_id = $1 ORDER BY created_at ASC',
      [turnId]
    );
    const actions = actionsResult.rows.map(serializeRow);

    // Get observations (exclude model_input_json, screenshot_uri will be file path)
    const observationsResult = await query(
      `SELECT id, turn_id, obs_type, text_content, screenshot_uri, created_at 
       FROM obs WHERE turn_id = $1 ORDER BY created_at ASC`,
      [turnId]
    );
    const observations = observationsResult.rows.map(serializeRow);

    // Set cache headers
    return NextResponse.json(
      { actions, observations },
      {
        headers: {
          'Cache-Control': 'public, max-age=300, stale-while-revalidate=600',
        },
      }
    );
  } catch (error: any) {
    console.error('Error fetching turn details:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to fetch turn details' },
      { status: 500 }
    );
  }
}

