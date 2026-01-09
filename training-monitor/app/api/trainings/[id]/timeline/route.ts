/**
 * API route to get timeline items (baselines, steps, evals) for a training.
 */

import { NextResponse } from 'next/server';
import { query, serializeRow } from '@/lib/db';

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

function rolloutCompletedTurns(rollout: any, maxTurns: number): number {
  const status = rollout?.status as string | undefined;
  const total = maxTurns > 0 ? maxTurns : 0;

  let done = 0;
  if (status === 'running') {
    done = Number(rollout?.current_turn ?? 0);
  } else if (status === 'completed' || status === 'failed' || status === 'cancelled') {
    done = Number(rollout?.num_turns ?? rollout?.current_turn ?? 0);
  } else {
    done = Number(rollout?.current_turn ?? 0);
  }

  if (!Number.isFinite(done)) done = 0;
  return clamp(done, 0, total);
}

function rolloutEffectiveTotalTurns(rollout: any, maxTurns: number): number {
  const status = rollout?.status as string | undefined;
  const total = maxTurns > 0 ? maxTurns : 0;
  if (total === 0) return 0;

  if (status === 'completed' || status === 'failed' || status === 'cancelled') {
    const completed = rolloutCompletedTurns(rollout, maxTurns);
    return completed > 0 ? completed : total;
  }
  return total;
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
    // Handle both Promise and direct params (Next.js 13+ vs 14+)
    const resolvedParams = params instanceof Promise ? await params : params;
    const trainingId = parseInt(resolvedParams.id);
    
    if (isNaN(trainingId)) {
      return NextResponse.json(
        { error: 'Invalid training ID' },
        { status: 400 }
      );
    }

    // Resolve training params (used for turn-based progress + ETA)
    let trainingMaxTurns = 20;
    let trainingGroupSize = 0;
    let trainingGroupsPerBatch = 0;
    let trainingTotalSteps = 0;
    let trainingConfigJson: any = null;
    try {
      const trainingResult = await query(
        `SELECT max_turns, group_size, groups_per_batch, total_steps, config_json FROM training WHERE id = $1`,
        [trainingId]
      );
      const trow = trainingResult.rows?.[0] ?? {};
      trainingMaxTurns = Number(trow.max_turns ?? 20);
      if (!Number.isFinite(trainingMaxTurns) || trainingMaxTurns <= 0) trainingMaxTurns = 20;
      trainingGroupSize = Number(trow.group_size ?? 0);
      if (!Number.isFinite(trainingGroupSize) || trainingGroupSize < 0) trainingGroupSize = 0;
      trainingGroupsPerBatch = Number(trow.groups_per_batch ?? 0);
      if (!Number.isFinite(trainingGroupsPerBatch) || trainingGroupsPerBatch < 0) trainingGroupsPerBatch = 0;
      trainingTotalSteps = Number(trow.total_steps ?? 0);
      if (!Number.isFinite(trainingTotalSteps) || trainingTotalSteps < 0) trainingTotalSteps = 0;
      trainingConfigJson = serializeRow({ config_json: trow.config_json }).config_json ?? null;
    } catch {
      trainingMaxTurns = 20;
    }

    // Best-effort avg_turn_time across the training based on completed rollouts
    let trainingAvgTurnTime: number | null = null;
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
        [trainingId]
      );
      const totalTime = Number(avgRes.rows?.[0]?.total_time ?? 0);
      const totalTurns = Number(avgRes.rows?.[0]?.total_turns ?? 0);
      if (Number.isFinite(totalTime) && Number.isFinite(totalTurns) && totalTurns > 0) {
        trainingAvgTurnTime = totalTime / totalTurns;
      }
    } catch {
      trainingAvgTurnTime = null;
    }

    const concurrency = getConcurrencyFromConfig(trainingConfigJson);

    // Get baselines
    const baselinesResult = await query(`
      SELECT id, status, progress_percent, eval_time, success_rate, 
             avg_reward, avg_turns, avg_turn_time, estimated_total_time, estimated_remaining_time,
             start_time, end_time, created_at, total_tasks
      FROM baseline
      WHERE training_id = $1
      ORDER BY created_at ASC
    `, [trainingId]);
    
    const baselines = [];
    for (const row of baselinesResult.rows) {
      const baseline = serializeRow(row);

      // Compute baseline progress on-the-fly (turn-based), so UI is correct even if DB progress_percent was written with old logic.
      try {
        const groupsAndRollouts = await query(
          `
          SELECT
            g.group_num,
            r.status,
            r.current_turn,
            r.num_turns
          FROM "group" g
          LEFT JOIN rollout r ON r.group_id = g.id
          WHERE g.source_type = 'baseline' AND g.baseline_id = $1
          ORDER BY g.group_num ASC
          `,
          [baseline.id]
        );

        const expectedTasks = Number(baseline.total_tasks ?? 0);
        const groupNums = new Set<number>();
        const completedByGroup = new Map<number, number>();
        const totalByGroup = new Map<number, number>();

        for (const r of groupsAndRollouts.rows ?? []) {
          const groupNum = Number(r.group_num);
          if (!Number.isFinite(groupNum)) continue;
          groupNums.add(groupNum);
          const prevC = completedByGroup.get(groupNum) ?? 0;
          completedByGroup.set(groupNum, prevC + rolloutCompletedTurns(r, trainingMaxTurns));
          const prevT = totalByGroup.get(groupNum) ?? 0;
          totalByGroup.set(groupNum, prevT + rolloutEffectiveTotalTurns(r, trainingMaxTurns));
        }

        const existingGroups = groupNums.size;
        const totalGroupsExpected = expectedTasks > 0 ? expectedTasks : existingGroups;

        let totalTurns = 0;
        let completedTurns = 0;
        for (const g of groupNums.values()) {
          const c = clamp(completedByGroup.get(g) ?? 0, 0, trainingMaxTurns);
          const tRaw = totalByGroup.get(g);
          const t = clamp((tRaw ?? trainingMaxTurns) || trainingMaxTurns, 0, trainingMaxTurns);
          totalTurns += t;
          completedTurns += clamp(c, 0, t);
        }
        totalTurns += Math.max(0, totalGroupsExpected - existingGroups) * trainingMaxTurns;

        baseline.progress_percent = totalTurns > 0 ? (completedTurns / totalTurns) * 100 : 0;

        const avgTurnTime =
          (baseline.avg_turn_time !== null && baseline.avg_turn_time !== undefined
            ? Number(baseline.avg_turn_time)
            : null) ?? trainingAvgTurnTime;
        if (avgTurnTime !== null && Number.isFinite(avgTurnTime) && avgTurnTime > 0) {
          baseline.avg_turn_time = avgTurnTime;
          baseline.estimated_total_time = (totalTurns * avgTurnTime) / concurrency;
          baseline.estimated_remaining_time =
            Math.max(0, ((totalTurns - completedTurns) * avgTurnTime) / concurrency);
        }
      } catch {
        // If compute fails, fall back to stored progress_percent
      }

      baselines.push({
        ...baseline,
        type: 'baseline',
        display_name: 'Baseline',
      });
    }

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
    
    // Precompute per-step completed turns (turn-based)
    const stepCompletedTurns = new Map<number, number>();
    try {
      const stepAgg = await query(
        `
        SELECT
          s.id AS step_id,
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
          ), 0) AS completed_turns
        FROM step s
        LEFT JOIN rollout r ON r.step_id = s.id
        WHERE s.training_id = $1
        GROUP BY s.id
        `,
        [trainingId, trainingMaxTurns]
      );
      for (const r of stepAgg.rows ?? []) {
        stepCompletedTurns.set(Number(r.step_id), Number(r.completed_turns ?? 0));
      }
    } catch {
      // ignore
    }

    const steps = stepsResult.rows.map((row: any) => {
      const stepItem = serializeRow(row);
      const completedTurns = Math.max(0, Number(stepCompletedTurns.get(Number(stepItem.id)) ?? 0));
      const totalTurns =
        trainingGroupsPerBatch > 0 && trainingGroupSize > 0
          ? trainingGroupsPerBatch * trainingGroupSize * trainingMaxTurns
          : 0;

      if (totalTurns > 0) {
        stepItem.progress_percent = (completedTurns / totalTurns) * 100;
        const avgTurnTime =
          (stepItem.avg_turn_time !== null && stepItem.avg_turn_time !== undefined
            ? Number(stepItem.avg_turn_time)
            : null) ?? trainingAvgTurnTime;
        if (avgTurnTime !== null && Number.isFinite(avgTurnTime) && avgTurnTime > 0) {
          stepItem.avg_turn_time = avgTurnTime;
          stepItem.estimated_total_time = (totalTurns * avgTurnTime) / concurrency;
          stepItem.estimated_remaining_time =
            Math.max(0, ((totalTurns - completedTurns) * avgTurnTime) / concurrency);
        }
      }

      return {
        ...stepItem,
        type: 'step',
        display_name: `Step ${row.step}`,
      };
    });

    // Get evals
    const evalsResult = await query(`
      SELECT id, step, status, progress_percent, eval_time,
             success_rate, avg_reward, avg_turns,
             avg_turn_time, estimated_total_time, estimated_remaining_time,
             start_time, end_time, created_at, total_tasks
      FROM eval
      WHERE training_id = $1
      ORDER BY step ASC
    `, [trainingId]);
    
    const evals = [];
    for (const row of evalsResult.rows) {
      const evalItem = serializeRow(row);

      try {
        const groupsAndRollouts = await query(
          `
          SELECT
            g.group_num,
            r.status,
            r.current_turn,
            r.num_turns
          FROM "group" g
          LEFT JOIN rollout r ON r.group_id = g.id
          WHERE g.source_type = 'eval' AND g.eval_id = $1
          ORDER BY g.group_num ASC
          `,
          [evalItem.id]
        );

        const expectedTasks = Number(evalItem.total_tasks ?? 0);
        const groupNums = new Set<number>();
        const completedByGroup = new Map<number, number>();
        const totalByGroup = new Map<number, number>();

        for (const r of groupsAndRollouts.rows ?? []) {
          const groupNum = Number(r.group_num);
          if (!Number.isFinite(groupNum)) continue;
          groupNums.add(groupNum);
          const prevC = completedByGroup.get(groupNum) ?? 0;
          completedByGroup.set(groupNum, prevC + rolloutCompletedTurns(r, trainingMaxTurns));
          const prevT = totalByGroup.get(groupNum) ?? 0;
          totalByGroup.set(groupNum, prevT + rolloutEffectiveTotalTurns(r, trainingMaxTurns));
        }

        const existingGroups = groupNums.size;
        const totalGroupsExpected = expectedTasks > 0 ? expectedTasks : existingGroups;

        let totalTurns = 0;
        let completedTurns = 0;
        for (const g of groupNums.values()) {
          const c = clamp(completedByGroup.get(g) ?? 0, 0, trainingMaxTurns);
          const tRaw = totalByGroup.get(g);
          const t = clamp((tRaw ?? trainingMaxTurns) || trainingMaxTurns, 0, trainingMaxTurns);
          totalTurns += t;
          completedTurns += clamp(c, 0, t);
        }
        totalTurns += Math.max(0, totalGroupsExpected - existingGroups) * trainingMaxTurns;

        evalItem.progress_percent = totalTurns > 0 ? (completedTurns / totalTurns) * 100 : 0;

        const avgTurnTime =
          (evalItem.avg_turn_time !== null && evalItem.avg_turn_time !== undefined
            ? Number(evalItem.avg_turn_time)
            : null) ?? trainingAvgTurnTime;
        if (avgTurnTime !== null && Number.isFinite(avgTurnTime) && avgTurnTime > 0) {
          evalItem.avg_turn_time = avgTurnTime;
          evalItem.estimated_total_time = (totalTurns * avgTurnTime) / concurrency;
          evalItem.estimated_remaining_time =
            Math.max(0, ((totalTurns - completedTurns) * avgTurnTime) / concurrency);
        }
      } catch {
        // fall back to stored progress_percent
      }

      evals.push({
        ...evalItem,
        type: 'eval',
        display_name: `Eval @ Step ${row.step}`,
      });
    }

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

