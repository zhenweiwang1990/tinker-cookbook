/**
 * API route to get a specific training session.
 */

import { NextResponse } from 'next/server';
import { query, serializeRow } from '@/lib/db';

export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string }> | { id: string } }
) {
  try {
    const resolvedParams = params instanceof Promise ? await params : params;
    const trainingId = parseInt(resolvedParams.id);
    
    if (isNaN(trainingId)) {
      return NextResponse.json(
        { error: 'Invalid training ID' },
        { status: 400 }
      );
    }

    const result = await query(
      `SELECT 
        id, run_name, log_path, model_name, lora_rank, learning_rate,
        batch_size, group_size, groups_per_batch, max_tokens, temperature,
        kl_penalty_coef, num_substeps, max_turns, seed, box_type,
        renderer_name, wandb_project, wandb_name, status, progress_percent,
        current_step, total_steps, current_phase, status_message, error_message,
        start_time, end_time, last_heartbeat, config_json, created_at, updated_at
      FROM training WHERE id = $1`,
      [trainingId]
    );

    if (result.rows.length === 0) {
      return NextResponse.json(
        { error: 'Training not found' },
        { status: 404 }
      );
    }

    return NextResponse.json({ training: serializeRow(result.rows[0]) });
  } catch (error: any) {
    console.error('Error fetching training:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to fetch training' },
      { status: 500 }
    );
  }
}

