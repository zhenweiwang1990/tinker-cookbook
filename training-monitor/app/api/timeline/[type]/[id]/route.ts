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
            r.id, r.rollout_id, r.task_id, r.status, r.task_success, 
            r.validation_passed, r.num_turns, r.current_turn, r.reward, 
            r.rollout_time, r.env_index, r.created_at,
            g.group_num as group_number, g.status as group_status
          FROM rollout r
          LEFT JOIN "group" g ON r.group_id = g.id
          WHERE r.source_type = 'baseline' AND r.baseline_id = $1
          ORDER BY g.group_num ASC, r.env_index ASC
        `, [itemId]);
        rollouts = rolloutsResult.rows.map(serializeRow);
        
        // Add groups to item
        item.groups = groups;
      }
    } else if (type === 'step') {
      const itemResult = await query('SELECT * FROM step WHERE id = $1', [itemId]);
      item = itemResult.rows[0] ? serializeRow(itemResult.rows[0]) : null;
      
      if (item) {
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
            r.id, r.rollout_id, r.task_id, r.status, r.task_success, 
            r.validation_passed, r.num_turns, r.current_turn, r.reward, 
            r.rollout_time, r.env_index, r.created_at,
            g.group_num as group_number, g.status as group_status
          FROM rollout r
          LEFT JOIN "group" g ON r.group_id = g.id
          WHERE r.source_type = 'step' AND r.step_id = $1
          ORDER BY g.group_num ASC, r.env_index ASC
        `, [itemId]);
        rollouts = rolloutsResult.rows.map(serializeRow);
        
        // Add groups to item
        item.groups = groups;
      }
    } else if (type === 'eval') {
      const itemResult = await query('SELECT * FROM eval WHERE id = $1', [itemId]);
      item = itemResult.rows[0] ? serializeRow(itemResult.rows[0]) : null;
      
      if (item) {
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
            r.id, r.rollout_id, r.task_id, r.status, r.task_success, 
            r.validation_passed, r.num_turns, r.current_turn, r.reward, 
            r.rollout_time, r.env_index, r.created_at,
            g.group_num as group_number, g.status as group_status
          FROM rollout r
          LEFT JOIN "group" g ON r.group_id = g.id
          WHERE r.source_type = 'eval' AND r.eval_id = $1
          ORDER BY g.group_num ASC, r.env_index ASC
        `, [itemId]);
        rollouts = rolloutsResult.rows.map(serializeRow);
        
        // Add groups to item
        item.groups = groups;
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

