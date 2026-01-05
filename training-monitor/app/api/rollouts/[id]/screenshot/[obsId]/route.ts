/**
 * API route to get screenshot URI for a specific observation (lazy loading).
 * This endpoint is cached to improve performance.
 */

import { NextResponse } from 'next/server';
import { query, serializeRow } from '@/lib/db';

export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string; obsId: string }> | { id: string; obsId: string } }
) {
  try {
    const resolvedParams = params instanceof Promise ? await params : params;
    const rolloutId = parseInt(resolvedParams.id);
    const obsId = parseInt(resolvedParams.obsId);

    if (isNaN(rolloutId) || isNaN(obsId)) {
      return NextResponse.json(
        { error: 'Invalid rollout ID or observation ID' },
        { status: 400 }
      );
    }

    // Verify observation belongs to a turn in this rollout
    const obsResult = await query(
      `SELECT o.screenshot_uri, o.obs_type 
       FROM obs o
       JOIN turn t ON o.turn_id = t.id
       WHERE o.id = $1 AND t.rollout_id = $2`,
      [obsId, rolloutId]
    );

    if (obsResult.rows.length === 0) {
      return NextResponse.json(
        { error: 'Observation not found or does not belong to this rollout' },
        { status: 404 }
      );
    }

    const obs = serializeRow(obsResult.rows[0]);

    // Set cache headers for better performance
    return NextResponse.json(
      { 
        screenshot_uri: obs.screenshot_uri,
        obs_type: obs.obs_type
      },
      {
        headers: {
          'Cache-Control': 'public, max-age=3600, stale-while-revalidate=86400',
        },
      }
    );
  } catch (error: any) {
    console.error('Error fetching screenshot:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to fetch screenshot' },
      { status: 500 }
    );
  }
}

