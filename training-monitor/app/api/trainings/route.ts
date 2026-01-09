/**
 * API route to get all training sessions.
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

export async function GET() {
  try {
    const result = await query(`
      SELECT 
        id, run_name, log_path, model_name, status, 
        progress_percent, current_step, total_steps,
        avg_turn_time, estimated_total_time, estimated_remaining_time,
        group_size, groups_per_batch, max_turns, config_json,
        start_time, end_time, last_heartbeat,
        created_at, updated_at
      FROM training
      ORDER BY COALESCE(start_time, created_at) DESC
    `);

    const trainings = result.rows.map(serializeRow);

    // Pre-fetch aggregates for turn-based progress computation (effective total turns)
    const [baselineAgg, evalAgg, rolloutAgg, avgTurnAgg] = await Promise.all([
      query(
        `
        SELECT training_id, COALESCE(MAX(total_tasks), 0) AS baseline_total_tasks
        FROM baseline
        GROUP BY training_id
        `
      ),
      query(
        `
        SELECT
          training_id,
          COALESCE(SUM(total_tasks), 0) AS eval_total_tasks_sum,
          COALESCE(MIN(total_tasks), 0) AS eval_default_tasks,
          COUNT(*) AS eval_count
        FROM eval
        GROUP BY training_id
        `
      ),
      query(
        `
        SELECT
          COALESCE(s.training_id, e.training_id, b.training_id) AS training_id,
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
              COALESCE(r.max_turns, t.max_turns, 20)
            )
          ), 0) AS completed_turns,
          COALESCE(SUM(
            CASE
              WHEN r.status IN ('completed', 'failed', 'cancelled') THEN
                LEAST(GREATEST(COALESCE(r.num_turns, r.current_turn, 0), 0), COALESCE(r.max_turns, t.max_turns, 20))
              ELSE COALESCE(r.max_turns, t.max_turns, 20)
            END
          ), 0) AS effective_total_turns
        FROM rollout r
        LEFT JOIN step s ON r.step_id = s.id
        LEFT JOIN eval e ON r.eval_id = e.id
        LEFT JOIN baseline b ON r.baseline_id = b.id
        LEFT JOIN training t ON t.id = COALESCE(s.training_id, e.training_id, b.training_id)
        WHERE COALESCE(s.training_id, e.training_id, b.training_id) IS NOT NULL
        GROUP BY 1
        `
      ),
      query(
        `
        SELECT
          COALESCE(s.training_id, e.training_id, b.training_id) AS training_id,
          COALESCE(SUM(r.rollout_time), 0) AS total_time,
          COALESCE(SUM(r.num_turns), 0) AS total_turns
        FROM rollout r
        LEFT JOIN step s ON r.step_id = s.id
        LEFT JOIN eval e ON r.eval_id = e.id
        LEFT JOIN baseline b ON r.baseline_id = b.id
        WHERE COALESCE(s.training_id, e.training_id, b.training_id) IS NOT NULL
          AND r.status = 'completed'
          AND r.rollout_time IS NOT NULL
          AND r.num_turns IS NOT NULL
          AND r.num_turns > 0
        GROUP BY 1
        `
      ),
    ]);

    const baselineTasksByTraining = new Map<number, number>();
    for (const row of baselineAgg.rows ?? []) {
      baselineTasksByTraining.set(Number(row.training_id), Number(row.baseline_total_tasks ?? 0));
    }

    const evalByTraining = new Map<
      number,
      { eval_total_tasks_sum: number; eval_default_tasks: number; eval_count: number }
    >();
    for (const row of evalAgg.rows ?? []) {
      evalByTraining.set(Number(row.training_id), {
        eval_total_tasks_sum: Number(row.eval_total_tasks_sum ?? 0),
        eval_default_tasks: Number(row.eval_default_tasks ?? 0),
        eval_count: Number(row.eval_count ?? 0),
      });
    }

    const rolloutAggByTraining = new Map<
      number,
      {
        completed_turns: number;
        effective_total_turns: number;
        baseline_rollout_count: number;
        step_rollout_count: number;
        eval_rollout_count: number;
      }
    >();
    for (const row of rolloutAgg.rows ?? []) {
      rolloutAggByTraining.set(Number(row.training_id), {
        completed_turns: Number(row.completed_turns ?? 0),
        effective_total_turns: Number(row.effective_total_turns ?? 0),
        baseline_rollout_count: Number(row.baseline_rollout_count ?? 0),
        step_rollout_count: Number(row.step_rollout_count ?? 0),
        eval_rollout_count: Number(row.eval_rollout_count ?? 0),
      });
    }

    const avgTurnTimeByTraining = new Map<number, number>();
    for (const row of avgTurnAgg.rows ?? []) {
      const trainingId = Number(row.training_id);
      const totalTime = Number(row.total_time ?? 0);
      const totalTurns = Number(row.total_turns ?? 0);
      if (!Number.isFinite(trainingId) || trainingId <= 0) continue;
      if (Number.isFinite(totalTime) && Number.isFinite(totalTurns) && totalTurns > 0) {
        avgTurnTimeByTraining.set(trainingId, totalTime / totalTurns);
      }
    }

    for (const training of trainings) {
      const trainingId = Number(training.id);
      const maxTurns = toPosInt(training.max_turns, 20);
      const totalSteps = Math.max(0, Math.floor(Number(training.total_steps ?? 0)));
      const groupsPerBatch = Math.max(0, Math.floor(Number(training.groups_per_batch ?? 0)));
      const groupSize = Math.max(0, Math.floor(Number(training.group_size ?? 0)));

      const baselineTotalTasks = Math.max(
        0,
        Math.floor(baselineTasksByTraining.get(trainingId) ?? 0)
      );

      const evalInfo = evalByTraining.get(trainingId) ?? {
        eval_total_tasks_sum: 0,
        eval_default_tasks: 0,
        eval_count: 0,
      };

      // Determine eval_every from config_json (best-effort)
      let evalEvery = 0;
      const cfg = training.config_json;
      if (cfg && typeof cfg === 'object' && !Array.isArray(cfg)) {
        evalEvery = Math.max(0, Math.floor(Number((cfg as any).eval_every ?? 0)));
      }

      let expectedEvalCount = evalInfo.eval_count;
      if (evalEvery > 0 && totalSteps > 0) {
        expectedEvalCount = Math.ceil(totalSteps / evalEvery);
      }

      let defaultEvalTasks =
        Math.max(0, Math.floor(evalInfo.eval_default_tasks ?? 0)) ||
        (evalInfo.eval_count > 0
          ? Math.max(0, Math.round((evalInfo.eval_total_tasks_sum ?? 0) / evalInfo.eval_count))
          : 0) ||
        baselineTotalTasks ||
        0;

      const baselineTurnsBudget = baselineTotalTasks > 0 ? baselineTotalTasks * maxTurns : 0;
      const expectedStepRollouts =
        totalSteps > 0 && groupsPerBatch > 0 && groupSize > 0 ? totalSteps * groupsPerBatch * groupSize : 0;
      const expectedEvalRollouts =
        expectedEvalCount > 0 && defaultEvalTasks > 0 ? expectedEvalCount * defaultEvalTasks : 0;
      const expectedBaselineRollouts = baselineTotalTasks > 0 ? baselineTotalTasks : 0;

      const agg = rolloutAggByTraining.get(trainingId) ?? {
        completed_turns: 0,
        effective_total_turns: 0,
        baseline_rollout_count: 0,
        step_rollout_count: 0,
        eval_rollout_count: 0,
      };

      const missingBaselineRollouts = Math.max(0, expectedBaselineRollouts - agg.baseline_rollout_count);
      const missingStepRollouts = Math.max(0, expectedStepRollouts - agg.step_rollout_count);
      const missingEvalRollouts = Math.max(0, expectedEvalRollouts - agg.eval_rollout_count);

      const totalTurnsBudget =
        Math.max(0, Number(agg.effective_total_turns ?? 0)) +
        (missingBaselineRollouts + missingStepRollouts + missingEvalRollouts) * maxTurns;
      const completedTurns = Math.max(0, Number(agg.completed_turns ?? 0));

      if (totalTurnsBudget > 0) {
        const progress = (completedTurns / totalTurnsBudget) * 100;
        training.progress_percent = clamp(progress, 0, 100);

        const avgTurnTimeRaw = Number(training.avg_turn_time ?? null);
        const avgTurnTime =
          (Number.isFinite(avgTurnTimeRaw) && avgTurnTimeRaw > 0
            ? avgTurnTimeRaw
            : avgTurnTimeByTraining.get(trainingId)) ?? NaN;
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
    }

    return NextResponse.json({ trainings });
  } catch (error: any) {
    console.error('Error fetching trainings:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to fetch trainings' },
      { status: 500 }
    );
  }
}

