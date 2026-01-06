/**
 * API route to get model input for a specific turn (lazy loading).
 * This endpoint is cached to improve performance.
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

    // Get model_input_json from obs table
    const obsResult = await query(
      `SELECT model_input_json FROM obs 
       WHERE turn_id = $1 AND obs_type = 'screenshot_before' AND model_input_json IS NOT NULL
       ORDER BY created_at ASC LIMIT 1`,
      [turnId]
    );

    if (obsResult.rows.length === 0) {
      return NextResponse.json(
        { error: 'Model input not found for this turn' },
        { status: 404 }
      );
    }

    const obs = serializeRow(obsResult.rows[0]);
    let modelInput = null;
    
    if (obs.model_input_json) {
      try {
        modelInput = typeof obs.model_input_json === 'string'
          ? JSON.parse(obs.model_input_json)
          : obs.model_input_json;
      } catch (e) {
        // If parsing fails, return as string
        modelInput = obs.model_input_json;
      }
    }

    // Set cache headers for better performance
    // Also return screenshot_uri for resolving placeholder image URLs in modelInput
    const screenshotResult = await query(
      `SELECT screenshot_uri FROM obs
       WHERE turn_id = $1 AND obs_type = 'screenshot_before' AND screenshot_uri IS NOT NULL
       ORDER BY created_at ASC LIMIT 1`,
      [turnId]
    );
    const screenshotRow = screenshotResult.rows[0] ? serializeRow(screenshotResult.rows[0]) : null;

    return NextResponse.json(
      { modelInput, screenshot_uri: screenshotRow?.screenshot_uri ?? null },
      {
        headers: {
          'Cache-Control': 'public, max-age=3600, stale-while-revalidate=86400',
        },
      }
    );
  } catch (error: any) {
    console.error('Error fetching model input:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to fetch model input' },
      { status: 500 }
    );
  }
}

