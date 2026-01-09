/**
 * API route to get a specific training session.
 */

import { NextResponse } from 'next/server';
import { query, serializeRow } from '@/lib/db';

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

function toPosInt(value: any, fallback: number): number {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  if (n <= 0) return fallback;
  return Math.floor(n);
}

function getConcurrencyFromConfig(cfg: any): number {
  if (!cfg || typeof cfg !== 'object' || Array.isArray(cfg)) return 1;
  const v =
    Number((cfg as any).max_concurrent_rollouts ?? (cfg as any).max_concurrent ?? 1);
  if (!Number.isFinite(v) || v <= 0) return 1;
  return Math.floor(v);
}

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

    const training = serializeRow(result.rows[0]);

    // Compute turn-based training progress on-the-fly (UI should not depend on stored progress_percent).
    const trainingIdNum = Number(training.id);
    const maxTurns = toPosInt(training.max_turns, 20);
    const totalSteps = Math.max(0, Math.floor(Number(training.total_steps ?? 0)));
    const groupsPerBatch = Math.max(0, Math.floor(Number(training.groups_per_batch ?? 0)));
    const groupSize = Math.max(0, Math.floor(Number(training.group_size ?? 0)));

    // Baseline tasks
    const baselineRow = await query(
      `SELECT COALESCE(MAX(total_tasks), 0) AS baseline_total_tasks
       FROM baseline
       WHERE training_id = $1`,
      [trainingIdNum]
    );
    const baselineTotalTasks = Math.max(0, Math.floor(Number(baselineRow.rows?.[0]?.baseline_total_tasks ?? 0)));

    // Eval tasks + count
    const evalRow = await query(
      `SELECT
         COALESCE(SUM(total_tasks), 0) AS eval_total_tasks_sum,
         COALESCE(MIN(total_tasks), 0) AS eval_default_tasks,
         COUNT(*) AS eval_count
       FROM eval
       WHERE training_id = $1`,
      [trainingIdNum]
    );
    const evalTotalTasksSum = Number(evalRow.rows?.[0]?.eval_total_tasks_sum ?? 0);
    const evalDefaultTasks = Number(evalRow.rows?.[0]?.eval_default_tasks ?? 0);
    const evalCount = Number(evalRow.rows?.[0]?.eval_count ?? 0);

    // Completed turns + effective total turns from rollouts (and counts per source) for effective-denominator progress.
    const rolloutAgg = await query(
      `
      SELECT
        COALESCE(SUM(CASE WHEN r.baseline_id IS NOT NULL THEN 1 ELSE 0 END), 0) AS baseline_rollout_count,
        COALESCE(SUM(CASE WHEN r.step_id IS NOT NULL THEN 1 ELSE 0 END), 0) AS step_rollout_count,
        COALESCE(SUM(CASE WHEN r.eval_id IS NOT NULL THEN 1 ELSE 0 END), 0) AS eval_rollout_count,
        COALESCE(SUM(
          LEAST(
            GREATEST(
              CASE
                WHEN r.status = 'running' THEN COALESCE(r.current_turn, 0)
                WHEN r.status IN ('completed', 'failed', 'cancelled') THEN COALESCE(r.num_turns, r.current_turn, 0)
                ELSE COALESCE(r.current_turn, 0)
              END,
              0
            ),
            COALESCE(r.max_turns, $2, 20)
          )
        ), 0) AS completed_turns,
        COALESCE(SUM(
          CASE
            WHEN r.status IN ('completed', 'failed', 'cancelled') THEN
              LEAST(GREATEST(COALESCE(r.num_turns, r.current_turn, 0), 0), COALESCE(r.max_turns, $2, 20))
            ELSE COALESCE(r.max_turns, $2, 20)
          END
        ), 0) AS effective_total_turns
      FROM rollout r
      LEFT JOIN step s ON r.step_id = s.id
      LEFT JOIN eval e ON r.eval_id = e.id
      LEFT JOIN baseline b ON r.baseline_id = b.id
      WHERE COALESCE(s.training_id, e.training_id, b.training_id) = $1
      `,
      [trainingIdNum, maxTurns]
    );
    const completedTurns = Math.max(0, Number(rolloutAgg.rows?.[0]?.completed_turns ?? 0));
    const effectiveTotalTurnsExisting = Math.max(0, Number(rolloutAgg.rows?.[0]?.effective_total_turns ?? 0));
    const baselineRolloutCount = Math.max(0, Number(rolloutAgg.rows?.[0]?.baseline_rollout_count ?? 0));
    const stepRolloutCount = Math.max(0, Number(rolloutAgg.rows?.[0]?.step_rollout_count ?? 0));
    const evalRolloutCount = Math.max(0, Number(rolloutAgg.rows?.[0]?.eval_rollout_count ?? 0));

    // Determine expected eval count and default eval task count (best-effort)
    let evalEvery = 0;
    const cfg = training.config_json;
    if (cfg && typeof cfg === 'object' && !Array.isArray(cfg)) {
      evalEvery = Math.max(0, Math.floor(Number((cfg as any).eval_every ?? 0)));
    }
    let expectedEvalCount = evalCount;
    if (evalEvery > 0 && totalSteps > 0) {
      expectedEvalCount = Math.ceil(totalSteps / evalEvery);
    }
    let defaultEvalTasks =
      Math.max(0, Math.floor(evalDefaultTasks ?? 0)) ||
      (evalCount > 0 ? Math.max(0, Math.round((evalTotalTasksSum ?? 0) / evalCount)) : 0) ||
      baselineTotalTasks ||
      0;

    const expectedBaselineRollouts = baselineTotalTasks > 0 ? baselineTotalTasks : 0;
    const expectedStepRollouts =
      totalSteps > 0 && groupsPerBatch > 0 && groupSize > 0 ? totalSteps * groupsPerBatch * groupSize : 0;
    const expectedEvalRollouts =
      expectedEvalCount > 0 && defaultEvalTasks > 0 ? expectedEvalCount * defaultEvalTasks : 0;

    const missingBaselineRollouts = Math.max(0, expectedBaselineRollouts - baselineRolloutCount);
    const missingStepRollouts = Math.max(0, expectedStepRollouts - stepRolloutCount);
    const missingEvalRollouts = Math.max(0, expectedEvalRollouts - evalRolloutCount);

    const totalTurnsBudget =
      effectiveTotalTurnsExisting + (missingBaselineRollouts + missingStepRollouts + missingEvalRollouts) * maxTurns;

    if (totalTurnsBudget > 0) {
      training.progress_percent = clamp((completedTurns / totalTurnsBudget) * 100, 0, 100);
      let avgTurnTime = Number(training.avg_turn_time ?? null);
      if (!Number.isFinite(avgTurnTime) || avgTurnTime <= 0) {
        // Fallback: compute avg turn time from completed rollouts in this training
        try {
          const avgRes = await query(
            `
            SELECT
              COALESCE(SUM(r.rollout_time), 0) AS total_time,
              COALESCE(SUM(r.num_turns), 0) AS total_turns
            FROM rollout r
            LEFT JOIN step s ON r.step_id = s.id
            LEFT JOIN eval e ON r.eval_id = e.id
            LEFT JOIN baseline b ON r.baseline_id = b.id
            WHERE COALESCE(s.training_id, e.training_id, b.training_id) = $1
              AND r.status = 'completed'
              AND r.rollout_time IS NOT NULL
              AND r.num_turns IS NOT NULL
              AND r.num_turns > 0
            `,
            [trainingIdNum]
          );
          const totalTime = Number(avgRes.rows?.[0]?.total_time ?? 0);
          const totalTurns = Number(avgRes.rows?.[0]?.total_turns ?? 0);
          if (Number.isFinite(totalTime) && Number.isFinite(totalTurns) && totalTurns > 0) {
            avgTurnTime = totalTime / totalTurns;
          }
        } catch {
          // ignore
        }
      }
      if (Number.isFinite(avgTurnTime) && avgTurnTime > 0) {
        training.avg_turn_time = avgTurnTime;
        const concurrency = getConcurrencyFromConfig(training.config_json);
        training.estimated_total_time = (totalTurnsBudget * avgTurnTime) / concurrency;
        training.estimated_remaining_time = Math.max(
          0,
          ((totalTurnsBudget - completedTurns) * avgTurnTime) / concurrency
        );
      }
    }

    return NextResponse.json({ training });
  } catch (error: any) {
    console.error('Error fetching training:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to fetch training' },
      { status: 500 }
    );
  }
}

