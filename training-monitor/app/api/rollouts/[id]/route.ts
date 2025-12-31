/**
 * API route to get details of a rollout including all turns.
 */

import { NextResponse } from 'next/server';
import { query, serializeRow } from '@/lib/db';

export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string }> | { id: string } }
) {
  try {
    const resolvedParams = params instanceof Promise ? await params : params;
    const rolloutId = parseInt(resolvedParams.id);

    if (isNaN(rolloutId)) {
      return NextResponse.json(
        { error: 'Invalid rollout ID' },
        { status: 400 }
      );
    }

    // Get rollout
    const rolloutResult = await query('SELECT * FROM rollout WHERE id = $1', [rolloutId]);

    if (rolloutResult.rows.length === 0) {
      return NextResponse.json(
        { error: 'Rollout not found' },
        { status: 404 }
      );
    }

    const rollout = serializeRow(rolloutResult.rows[0]);
    const taskId = (rollout as any).task_id;

    // Get task
    let task = null;
    if (taskId) {
      const taskResult = await query('SELECT * FROM task WHERE id = $1', [taskId]);
      task = taskResult.rows[0] ? serializeRow(taskResult.rows[0]) : null;
    }

    // Get validation
    const validationResult = await query('SELECT * FROM validation WHERE rollout_id = $1', [rolloutId]);
    const validation = validationResult.rows[0] ? serializeRow(validationResult.rows[0]) : null;

    // Get environment
    const environmentResult = await query('SELECT * FROM environment WHERE rollout_id = $1', [rolloutId]);
    const environment = environmentResult.rows[0] ? serializeRow(environmentResult.rows[0]) : null;

    // Get turns with actions and observations
    const turnsResult = await query(`
      SELECT * FROM turn 
      WHERE rollout_id = $1
      ORDER BY turn ASC
    `, [rolloutId]);

    const turns = turnsResult.rows.map(serializeRow);

    // For each turn, get actions and observations
    const turnsWithDetails = await Promise.all(
      turns.map(async (turn: any) => {
        const actionsResult = await query(
          'SELECT * FROM action WHERE turn_id = $1 ORDER BY created_at ASC',
          [turn.id]
        );
        const actions = actionsResult.rows.map(serializeRow);

        const observationsResult = await query(
          'SELECT * FROM obs WHERE turn_id = $1 ORDER BY created_at ASC',
          [turn.id]
        );
        const observations = observationsResult.rows.map(serializeRow);

        return {
          ...turn,
          actions,
          observations,
        };
      })
    );

    return NextResponse.json({
      rollout,
      task,
      validation,
      environment,
      turns: turnsWithDetails,
    });
  } catch (error: any) {
    console.error('Error fetching rollout:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to fetch rollout' },
      { status: 500 }
    );
  }
}

