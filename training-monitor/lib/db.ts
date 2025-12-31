/**
 * Database connection and query utilities for PostgreSQL.
 */

import { Pool, QueryResult } from 'pg';

let pool: Pool | null = null;

export function getDatabase(): Pool {
  if (pool) {
    return pool;
  }

  // Get database connection from environment variables
  const dbConfig = {
    host: process.env.POSTGRES_HOST || process.env.DATABASE_URL?.match(/@([^:]+):/)?.[1] || 'localhost',
    port: parseInt(process.env.POSTGRES_PORT || process.env.DATABASE_URL?.match(/:(\d+)\//)?.[1] || '5432'),
    database: process.env.POSTGRES_DB || process.env.DATABASE_URL?.match(/\/([^?]+)/)?.[1] || 'training_db',
    user: process.env.POSTGRES_USER || process.env.DATABASE_URL?.match(/:\/\/([^:]+):/)?.[1] || 'training_user',
    password: process.env.POSTGRES_PASSWORD || process.env.DATABASE_URL?.match(/:[^:]+:([^@]+)@/)?.[1] || 'training_password',
  };

  // If DATABASE_URL is provided, use it directly
  if (process.env.DATABASE_URL) {
    pool = new Pool({
      connectionString: process.env.DATABASE_URL,
      max: 20, // Maximum number of clients in the pool
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 2000,
    });
  } else {
    pool = new Pool({
      ...dbConfig,
      max: 20,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 2000,
    });
  }

  // Handle pool errors
  pool.on('error', (err) => {
    console.error('Unexpected error on idle client', err);
  });

  return pool;
}

export function closeDatabase(): Promise<void> {
  if (pool) {
    return pool.end();
  }
  return Promise.resolve();
}

// Helper to serialize dates and handle JSON fields
export function serializeRow(row: any): any {
  const result: any = {};
  const booleanFields = [
    'is_eval', 'task_completed', 'task_success', 'validation_passed',
    'agent_reported_success', 'episode_done', 'ran_out_of_turns',
    'attempted_completion', 'success'
  ];
  
  for (const [key, value] of Object.entries(row)) {
    if (value === null || value === undefined) {
      result[key] = value;
    } else if (typeof value === 'string') {
      // Try to parse JSON fields
      if (key.includes('_json') && (value.startsWith('{') || value.startsWith('['))) {
        try {
          result[key] = JSON.parse(value);
        } catch {
          result[key] = value;
        }
      } else {
        result[key] = value;
      }
    } else if (typeof value === 'boolean') {
      result[key] = value;
    } else if (typeof value === 'number' && booleanFields.includes(key)) {
      // Convert PostgreSQL boolean or integer (0/1) to boolean
      result[key] = value === 1;
    } else if (value instanceof Date) {
      // Convert Date to ISO string
      result[key] = value.toISOString();
    } else {
      result[key] = value;
    }
  }
  return result;
}

// Helper to execute queries
export async function query(text: string, params?: any[]): Promise<QueryResult> {
  const db = getDatabase();
  return db.query(text, params);
}
