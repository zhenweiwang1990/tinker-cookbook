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
    const envId = (rollout as any).env_id;
    
    // Optimize trajectory_data_json: only keep execution_details, remove training_data
    if ((rollout as any).trajectory_data_json) {
      try {
        const trajectoryData = typeof (rollout as any).trajectory_data_json === 'string'
          ? JSON.parse((rollout as any).trajectory_data_json)
          : (rollout as any).trajectory_data_json;
        
        // Keep only execution_details if it exists
        if (trajectoryData.execution_details) {
          (rollout as any).trajectory_data_json = { execution_details: trajectoryData.execution_details };
        } else if (trajectoryData.turns) {
          // Backward compatibility: if old format with just turns, keep it
          (rollout as any).trajectory_data_json = { turns: trajectoryData.turns };
        } else {
          // Empty object if nothing useful
          (rollout as any).trajectory_data_json = {};
        }
      } catch (e) {
        // If parsing fails, remove it entirely
        (rollout as any).trajectory_data_json = null;
      }
    }

    // Get task
    let task = null;
    if (taskId) {
      const taskResult = await query('SELECT * FROM task WHERE id = $1', [taskId]);
      task = taskResult.rows[0] ? serializeRow(taskResult.rows[0]) : null;
    }

    // Get validation
    const validationResult = await query('SELECT * FROM validation WHERE rollout_id = $1', [rolloutId]);
    const validation = validationResult.rows[0] ? serializeRow(validationResult.rows[0]) : null;

    // Get environment (now using env_id from rollout)
    let environment = null;
    if (envId) {
      const environmentResult = await query('SELECT * FROM environment WHERE id = $1', [envId]);
      environment = environmentResult.rows[0] ? serializeRow(environmentResult.rows[0]) : null;
    }

    // Get turns - only basic info, details will be loaded on demand
    const turnsResult = await query(`
      SELECT id, rollout_id, turn, start_time, end_time, turn_time, reward, 
             episode_done, metrics_json, model_response, created_at
      FROM turn 
      WHERE rollout_id = $1
      ORDER BY turn ASC
    `, [rolloutId]);

    const turns = turnsResult.rows.map(serializeRow);

    return NextResponse.json({
      rollout,
      task,
      validation,
      environment,
      turns,
    });
  } catch (error: any) {
    console.error('Error fetching rollout:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to fetch rollout' },
      { status: 500 }
    );
  }
}

