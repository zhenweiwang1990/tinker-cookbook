/**
 * API route to get all training sessions.
 */

import { NextResponse } from 'next/server';
import { query, serializeRow } from '@/lib/db';

export async function GET() {
  try {
    const result = await query(`
      SELECT 
        id, run_name, log_path, model_name, status, 
        progress_percent, current_step, total_steps,
        start_time, end_time, last_heartbeat,
        created_at, updated_at
      FROM training
      ORDER BY COALESCE(start_time, created_at) DESC
    `);

    const trainings = result.rows.map(serializeRow);

    return NextResponse.json({ trainings });
  } catch (error: any) {
    console.error('Error fetching trainings:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to fetch trainings' },
      { status: 500 }
    );
  }
}

