/**
 * API route to get details of a timeline item (baseline, step, or eval).
 */

import { NextResponse } from 'next/server';
import { query, serializeRow } from '@/lib/db';

export async function GET(
  request: Request,
  { params }: { params: Promise<{ type: string; id: string }> | { type: string; id: string } }
) {
  try {
    const resolvedParams = params instanceof Promise ? await params : params;
    const { type, id } = resolvedParams;
    const itemId = parseInt(id);

    if (isNaN(itemId)) {
      return NextResponse.json(
        { error: 'Invalid ID' },
        { status: 400 }
      );
    }

    let item: any = null;
    let rollouts: any[] = [];

    if (type === 'baseline') {
      const itemResult = await query('SELECT * FROM baseline WHERE id = $1', [itemId]);
      item = itemResult.rows[0] ? serializeRow(itemResult.rows[0]) : null;
      
      if (item) {
        const rolloutsResult = await query(`
          SELECT * FROM rollout 
          WHERE source_type = 'baseline' AND baseline_id = $1
          ORDER BY created_at ASC
        `, [itemId]);
        rollouts = rolloutsResult.rows.map(serializeRow);
      }
    } else if (type === 'step') {
      const itemResult = await query('SELECT * FROM step WHERE id = $1', [itemId]);
      item = itemResult.rows[0] ? serializeRow(itemResult.rows[0]) : null;
      
      if (item) {
        const rolloutsResult = await query(`
          SELECT * FROM rollout 
          WHERE source_type = 'step' AND step_id = $1
          ORDER BY created_at ASC
        `, [itemId]);
        rollouts = rolloutsResult.rows.map(serializeRow);
      }
    } else if (type === 'eval') {
      const itemResult = await query('SELECT * FROM eval WHERE id = $1', [itemId]);
      item = itemResult.rows[0] ? serializeRow(itemResult.rows[0]) : null;
      
      if (item) {
        const rolloutsResult = await query(`
          SELECT * FROM rollout 
          WHERE source_type = 'eval' AND eval_id = $1
          ORDER BY created_at ASC
        `, [itemId]);
        rollouts = rolloutsResult.rows.map(serializeRow);
      }
    } else {
      return NextResponse.json(
        { error: 'Invalid type. Must be baseline, step, or eval' },
        { status: 400 }
      );
    }

    if (!item) {
      return NextResponse.json(
        { error: `${type} not found` },
        { status: 404 }
      );
    }

    return NextResponse.json({
      item,
      rollouts,
    });
  } catch (error: any) {
    console.error('Error fetching timeline item:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to fetch timeline item' },
      { status: 500 }
    );
  }
}

