/**
 * API route to get timeline items (baselines, steps, evals) for a training.
 */

import { NextResponse } from 'next/server';
import { query, serializeRow } from '@/lib/db';

export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string }> | { id: string } }
) {
  try {
    // Handle both Promise and direct params (Next.js 13+ vs 14+)
    const resolvedParams = params instanceof Promise ? await params : params;
    const trainingId = parseInt(resolvedParams.id);
    
    if (isNaN(trainingId)) {
      return NextResponse.json(
        { error: 'Invalid training ID' },
        { status: 400 }
      );
    }

    // Get baselines
    const baselinesResult = await query(`
      SELECT id, status, progress_percent, eval_time, success_rate, 
             avg_reward, avg_turns, avg_turn_time, estimated_total_time, estimated_remaining_time,
             start_time, end_time, created_at
      FROM baseline
      WHERE training_id = $1
      ORDER BY created_at ASC
    `, [trainingId]);
    
    const baselines = baselinesResult.rows.map((row: any) => ({
      ...serializeRow(row),
      type: 'baseline',
      display_name: 'Baseline',
    }));

    // Get steps
    const stepsResult = await query(`
      SELECT id, step, batch, status, progress_percent, 
             reward_mean, reward_std, loss, num_trajectories,
             avg_turn_time, estimated_total_time, estimated_remaining_time,
             start_time, end_time, created_at
      FROM step
      WHERE training_id = $1
      ORDER BY step ASC
    `, [trainingId]);
    
    const steps = stepsResult.rows.map((row: any) => ({
      ...serializeRow(row),
      type: 'step',
      display_name: `Step ${row.step}`,
    }));

    // Get evals
    const evalsResult = await query(`
      SELECT id, step, status, progress_percent, eval_time,
             success_rate, avg_reward, avg_turns,
             avg_turn_time, estimated_total_time, estimated_remaining_time,
             start_time, end_time, created_at
      FROM eval
      WHERE training_id = $1
      ORDER BY step ASC
    `, [trainingId]);
    
    const evals = evalsResult.rows.map((row: any) => ({
      ...serializeRow(row),
      type: 'eval',
      display_name: `Eval @ Step ${row.step}`,
    }));

    // Combine and sort by created_at (handle null values)
    const timeline = [...baselines, ...steps, ...evals].sort((a, b) => {
      const timeA = a.created_at ? new Date(a.created_at).getTime() : 0;
      const timeB = b.created_at ? new Date(b.created_at).getTime() : 0;
      return timeA - timeB;
    });

    console.log(`Timeline for training ${trainingId}:`, {
      baselines: baselines.length,
      steps: steps.length,
      evals: evals.length,
      total: timeline.length,
    });

    // Log first few items for debugging
    if (timeline.length > 0) {
      console.log('First timeline items:', timeline.slice(0, 3).map(item => ({
        type: item.type,
        id: item.id,
        display_name: item.display_name,
        status: item.status,
      })));
    }

    return NextResponse.json({ timeline });
  } catch (error: any) {
    console.error('Error fetching timeline:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to fetch timeline' },
      { status: 500 }
    );
  }
}

