/**
 * API route to get details of a timeline item (baseline, step, or eval).
 */

import { NextResponse } from 'next/server';
import { query, serializeRow } from '@/lib/db';

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

function rolloutCompletedTurns(
  rollout: any,
  maxTurns: number
): number {
  const status = rollout?.status as string | undefined;
  const total = maxTurns > 0 ? maxTurns : 0;

  let done = 0;
  if (status === 'running') {
    done = Number(rollout?.current_turn ?? 0);
  } else if (status === 'completed' || status === 'failed' || status === 'cancelled') {
    // Strictly turn-based: prefer num_turns, fallback to current_turn.
    done = Number(rollout?.num_turns ?? rollout?.current_turn ?? 0);
  } else {
    done = Number(rollout?.current_turn ?? 0);
  }

  if (!Number.isFinite(done)) done = 0;
  return clamp(done, 0, total);
}

function rolloutEffectiveTotalTurns(
  rollout: any,
  maxTurns: number
): number {
  const status = rollout?.status as string | undefined;
  const total = maxTurns > 0 ? maxTurns : 0;

  if (total === 0) return 0;

  // For finished rollouts, effective total is the actual turns taken.
  if (status === 'completed' || status === 'failed' || status === 'cancelled') {
    const completed = rolloutCompletedTurns(rollout, maxTurns);
    return completed > 0 ? completed : total;
  }

  // For running/pending, we still don't know if it will early-stop, so use full budget.
  return total;
}

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
    let trainingMaxTurns: number = 20;

    if (type === 'baseline') {
      const itemResult = await query('SELECT * FROM baseline WHERE id = $1', [itemId]);
      item = itemResult.rows[0] ? serializeRow(itemResult.rows[0]) : null;
      
      if (item) {
        // Resolve max_turns from the parent training (baseline table doesn't store it)
        try {
          const trainingResult = await query(
            'SELECT max_turns FROM training WHERE id = $1',
            [item.training_id]
          );
          trainingMaxTurns = Number(trainingResult.rows?.[0]?.max_turns ?? 20);
          if (!Number.isFinite(trainingMaxTurns) || trainingMaxTurns <= 0) trainingMaxTurns = 20;
        } catch {
          trainingMaxTurns = 20;
        }

        // Expose max_turns to UI (used for rollout mini progress bars)
        item.max_turns = trainingMaxTurns;

        // Get groups for this baseline
        const groupsResult = await query(`
          SELECT 
            id, group_num, status, progress_percent, completed_rollouts, 
            num_rollouts, reward_mean, reward_std, start_time, end_time, 
            created_at, updated_at
          FROM "group" 
          WHERE source_type = 'baseline' AND baseline_id = $1
          ORDER BY group_num ASC
        `, [itemId]);
        const groups = groupsResult.rows.map(serializeRow);
        
        // Get rollouts with group information (exclude large fields like trajectory_data_json)
        const rolloutsResult = await query(`
          SELECT 
            r.id, r.rollout_id, r.task_id, t.name AS task_name, t.task_id AS task_key, r.status, r.task_success, 
            r.validation_passed, r.num_turns, r.current_turn, r.reward, 
            r.rollout_time, r.env_index, r.created_at,
            g.group_num as group_number, g.status as group_status
          FROM rollout r
          LEFT JOIN task t ON r.task_id = t.id
          LEFT JOIN "group" g ON r.group_id = g.id
          WHERE r.source_type = 'baseline' AND r.baseline_id = $1
          ORDER BY g.group_num ASC, r.env_index ASC
        `, [itemId]);
        rollouts = rolloutsResult.rows.map(serializeRow);
        
        // Compute progress (turn-based) for baseline + groups, so UI is correct even if DB progress_percent was written with old logic.
        const rolloutsByGroup = new Map<number, any[]>();
        for (const r of rollouts) {
          const g = Number(r.group_number);
          if (!Number.isFinite(g)) continue;
          if (!rolloutsByGroup.has(g)) rolloutsByGroup.set(g, []);
          rolloutsByGroup.get(g)!.push(r);
        }

        const expectedTasks = Number(item.total_tasks ?? groups.length ?? 0);
        const totalGroupsExpected = expectedTasks > 0 ? expectedTasks : groups.length;

        let totalTurns = 0;
        let completedTurns = 0;

        const updatedGroups = groups.map((g: any) => {
          const groupNum = Number(g.group_num);
          const groupRollouts = rolloutsByGroup.get(groupNum) ?? [];
          // Baseline/eval semantics: one task per group, effective total shrinks for early-stop.
          const groupCompletedTurnsRaw = groupRollouts.reduce(
            (acc, r) => acc + rolloutCompletedTurns(r, trainingMaxTurns),
            0
          );
          const groupTotalTurnsRaw =
            groupRollouts.length > 0
              ? groupRollouts.reduce(
                  (acc, r) => acc + rolloutEffectiveTotalTurns(r, trainingMaxTurns),
                  0
                )
              : trainingMaxTurns;
          const groupTotalTurns = clamp(groupTotalTurnsRaw, 0, trainingMaxTurns);
          const groupCompletedTurns = clamp(groupCompletedTurnsRaw, 0, groupTotalTurns);

          totalTurns += groupTotalTurns;
          completedTurns += groupCompletedTurns;

          return {
            ...g,
            progress_percent: groupTotalTurns > 0 ? (groupCompletedTurns / groupTotalTurns) * 100 : 0,
          };
        });

        // Add not-yet-created groups (not-started tasks) into denominator.
        const missingGroups = Math.max(0, totalGroupsExpected - groups.length);
        totalTurns += missingGroups * trainingMaxTurns;

        item.progress_percent = totalTurns > 0 ? (completedTurns / totalTurns) * 100 : 0;
        item.groups = updatedGroups;

        // Add groups to item
        // (groups already attached above with computed progress)
      }
    } else if (type === 'step') {
      const itemResult = await query('SELECT * FROM step WHERE id = $1', [itemId]);
      item = itemResult.rows[0] ? serializeRow(itemResult.rows[0]) : null;
      
      if (item) {
        // Resolve training params for turn budget
        let groupsPerBatch = 0;
        let groupSize = 1;
        try {
          const trainingResult = await query(
            'SELECT max_turns, groups_per_batch, group_size FROM training WHERE id = $1',
            [item.training_id]
          );
          const row = trainingResult.rows?.[0] ?? {};
          trainingMaxTurns = Number(row.max_turns ?? 20);
          if (!Number.isFinite(trainingMaxTurns) || trainingMaxTurns <= 0) trainingMaxTurns = 20;
          groupsPerBatch = Number(row.groups_per_batch ?? 0);
          if (!Number.isFinite(groupsPerBatch) || groupsPerBatch < 0) groupsPerBatch = 0;
          groupSize = Number(row.group_size ?? 1);
          if (!Number.isFinite(groupSize) || groupSize <= 0) groupSize = 1;
        } catch {
          trainingMaxTurns = 20;
          groupsPerBatch = 0;
          groupSize = 1;
        }

        item.max_turns = trainingMaxTurns;

        // Get groups for this step
        const groupsResult = await query(`
          SELECT 
            id, group_num, status, progress_percent, completed_rollouts, 
            num_rollouts, reward_mean, reward_std, start_time, end_time, 
            created_at, updated_at
          FROM "group" 
          WHERE source_type = 'step' AND step_id = $1
          ORDER BY group_num ASC
        `, [itemId]);
        const groups = groupsResult.rows.map(serializeRow);
        
        // Get rollouts with group information (exclude large fields like trajectory_data_json)
        const rolloutsResult = await query(`
          SELECT 
            r.id, r.rollout_id, r.task_id, t.name AS task_name, t.task_id AS task_key, r.status, r.task_success, 
            r.validation_passed, r.num_turns, r.current_turn, r.reward, 
            r.rollout_time, r.env_index, r.created_at,
            g.group_num as group_number, g.status as group_status
          FROM rollout r
          LEFT JOIN task t ON r.task_id = t.id
          LEFT JOIN "group" g ON r.group_id = g.id
          WHERE r.source_type = 'step' AND r.step_id = $1
          ORDER BY g.group_num ASC, r.env_index ASC
        `, [itemId]);
        rollouts = rolloutsResult.rows.map(serializeRow);
        
        // Compute progress (turn-based) for step + groups
        const rolloutsByGroup = new Map<number, any[]>();
        for (const r of rollouts) {
          const g = Number(r.group_number);
          if (!Number.isFinite(g)) continue;
          if (!rolloutsByGroup.has(g)) rolloutsByGroup.set(g, []);
          rolloutsByGroup.get(g)!.push(r);
        }

        const expectedGroups = groupsPerBatch > 0 ? groupsPerBatch : groups.length;
        let totalTurns = 0;
        let completedTurns = 0;

        const updatedGroups = groups.map((g: any) => {
          const groupNum = Number(g.group_num);
          const groupRollouts = rolloutsByGroup.get(groupNum) ?? [];

          const groupCompletedTurnsRaw = groupRollouts.reduce(
            (acc, r) => acc + rolloutCompletedTurns(r, trainingMaxTurns),
            0
          );
          const groupTotalTurnsRaw =
            groupRollouts.length > 0
              ? groupRollouts.reduce(
                  (acc, r) => acc + rolloutEffectiveTotalTurns(r, trainingMaxTurns),
                  0
                )
              : groupSize * trainingMaxTurns;
          const groupTotalTurns = clamp(groupTotalTurnsRaw, 0, groupSize * trainingMaxTurns);
          const groupCompletedTurns = clamp(groupCompletedTurnsRaw, 0, groupTotalTurns);

          totalTurns += groupTotalTurns;
          completedTurns += groupCompletedTurns;

          return {
            ...g,
            progress_percent: groupTotalTurns > 0 ? (groupCompletedTurns / groupTotalTurns) * 100 : 0,
          };
        });

        // Add missing groups into denominator (0 turns completed).
        const missingGroups = Math.max(0, expectedGroups - groups.length);
        totalTurns += missingGroups * groupSize * trainingMaxTurns;

        item.progress_percent = totalTurns > 0 ? (completedTurns / totalTurns) * 100 : 0;
        item.groups = updatedGroups;

        // Add groups to item
        // (groups already attached above with computed progress)
      }
    } else if (type === 'eval') {
      const itemResult = await query('SELECT * FROM eval WHERE id = $1', [itemId]);
      item = itemResult.rows[0] ? serializeRow(itemResult.rows[0]) : null;
      
      if (item) {
        // Resolve max_turns from the parent training (eval table doesn't store it)
        try {
          const trainingResult = await query(
            'SELECT max_turns FROM training WHERE id = $1',
            [item.training_id]
          );
          trainingMaxTurns = Number(trainingResult.rows?.[0]?.max_turns ?? 20);
          if (!Number.isFinite(trainingMaxTurns) || trainingMaxTurns <= 0) trainingMaxTurns = 20;
        } catch {
          trainingMaxTurns = 20;
        }

        item.max_turns = trainingMaxTurns;

        // Get groups for this eval
        const groupsResult = await query(`
          SELECT 
            id, group_num, status, progress_percent, completed_rollouts, 
            num_rollouts, reward_mean, reward_std, start_time, end_time, 
            created_at, updated_at
          FROM "group" 
          WHERE source_type = 'eval' AND eval_id = $1
          ORDER BY group_num ASC
        `, [itemId]);
        const groups = groupsResult.rows.map(serializeRow);
        
        // Get rollouts with group information (exclude large fields like trajectory_data_json)
        const rolloutsResult = await query(`
          SELECT 
            r.id, r.rollout_id, r.task_id, t.name AS task_name, t.task_id AS task_key, r.status, r.task_success, 
            r.validation_passed, r.num_turns, r.current_turn, r.reward, 
            r.rollout_time, r.env_index, r.created_at,
            g.group_num as group_number, g.status as group_status
          FROM rollout r
          LEFT JOIN task t ON r.task_id = t.id
          LEFT JOIN "group" g ON r.group_id = g.id
          WHERE r.source_type = 'eval' AND r.eval_id = $1
          ORDER BY g.group_num ASC, r.env_index ASC
        `, [itemId]);
        rollouts = rolloutsResult.rows.map(serializeRow);
        
        // Compute progress (turn-based) for eval + groups
        const rolloutsByGroup = new Map<number, any[]>();
        for (const r of rollouts) {
          const g = Number(r.group_number);
          if (!Number.isFinite(g)) continue;
          if (!rolloutsByGroup.has(g)) rolloutsByGroup.set(g, []);
          rolloutsByGroup.get(g)!.push(r);
        }

        const expectedTasks = Number(item.total_tasks ?? groups.length ?? 0);
        const totalGroupsExpected = expectedTasks > 0 ? expectedTasks : groups.length;

        let totalTurns = 0;
        let completedTurns = 0;

        const updatedGroups = groups.map((g: any) => {
          const groupNum = Number(g.group_num);
          const groupRollouts = rolloutsByGroup.get(groupNum) ?? [];
          const groupCompletedTurnsRaw = groupRollouts.reduce(
            (acc, r) => acc + rolloutCompletedTurns(r, trainingMaxTurns),
            0
          );
          const groupTotalTurnsRaw =
            groupRollouts.length > 0
              ? groupRollouts.reduce(
                  (acc, r) => acc + rolloutEffectiveTotalTurns(r, trainingMaxTurns),
                  0
                )
              : trainingMaxTurns;
          const groupTotalTurns = clamp(groupTotalTurnsRaw, 0, trainingMaxTurns);
          const groupCompletedTurns = clamp(groupCompletedTurnsRaw, 0, groupTotalTurns);

          totalTurns += groupTotalTurns;
          completedTurns += groupCompletedTurns;

          return {
            ...g,
            progress_percent: groupTotalTurns > 0 ? (groupCompletedTurns / groupTotalTurns) * 100 : 0,
          };
        });

        const missingGroups = Math.max(0, totalGroupsExpected - groups.length);
        totalTurns += missingGroups * trainingMaxTurns;

        item.progress_percent = totalTurns > 0 ? (completedTurns / totalTurns) * 100 : 0;
        item.groups = updatedGroups;

        // Add groups to item
        // (groups already attached above with computed progress)
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

