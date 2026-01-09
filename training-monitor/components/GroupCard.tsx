'use client';

import styles from './GroupCard.module.css';
import ProgressBar from './ProgressBar';

interface Rollout {
  id: number;
  rollout_id: string;
  task_id: number;
  task_name?: string;
  task_key?: string; // fallback: string task id (e.g. task_adapter_...)
  status: string;
  task_success: boolean;
  validation_passed: boolean;
  num_turns: number | null;
  current_turn: number | null;
  reward: number | null;
  rollout_time: number | null;
  created_at: string;
  group_number?: number;
  group_status?: string;
  env_index?: number;
  max_turns?: number | null;
}

interface GroupCardProps {
  group: any;
  rollouts: Rollout[];
  onSelectRollout: (id: number) => void;
  maxTurns?: number;
  taskName?: string | null;
}

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

function getRolloutPercent(rollout: Rollout, maxTurns: number): number {
  const total = maxTurns > 0 ? maxTurns : 0;
  if (total === 0) return 0;
  const status = rollout.status;
  let done = 0;
  if (status === 'running') {
    done = Number(rollout.current_turn ?? 0);
  } else if (status === 'completed' || status === 'failed' || status === 'cancelled') {
    done = Number(rollout.num_turns ?? rollout.current_turn ?? 0);
  } else {
    done = Number(rollout.current_turn ?? 0);
  }
  if (!Number.isFinite(done)) done = 0;
  return (clamp(done, 0, total) / total) * 100;
}

function getRolloutColor(rollout: Rollout): string {
  if (rollout.status === 'completed') {
    return rollout.task_success ? '#28a745' : '#dc3545'; // green / red
  }
  if (rollout.status === 'failed') return '#dc3545';
  if (rollout.status === 'running') return '#ff9800';
  return '#6c757d';
}

export default function GroupCard({ group, rollouts, onSelectRollout, maxTurns = 20, taskName }: GroupCardProps) {
  // Calculate success rate for this group
  const completedRollouts = rollouts.filter(r => r.status === 'completed');
  const successCount = completedRollouts.filter(r => r.task_success).length;
  const isRunning = group.status === 'running' || rollouts.some(r => r.status === 'running');
  const isCompleted = group.status === 'completed';
  
  // Calculate progress percentage
  const progressPercent = group.progress_percent !== null && group.progress_percent !== undefined 
    ? group.progress_percent 
    : 0;

  return (
    <div className={`${styles.card} ${isRunning ? styles.cardRunning : ''} ${isCompleted ? styles.cardCompleted : ''}`}>
      <div className={styles.header}>
        <div className={styles.headerTop}>
          <div className={styles.groupTitle}>
            <h3 className={styles.groupNumber}>
              {isRunning && <span className={styles.runningIndicator}>●</span>}
              Group {group.group_num}
            </h3>
            <span className={`${styles.statusBadge} ${styles[`status${group.status}`]}`}>
              {group.status}
            </span>
          </div>
        </div>

        {taskName ? (
          <div className={styles.taskName} title={taskName}>
            {taskName}
          </div>
        ) : null}

        {/* Progress Bar */}
        <div style={{ marginBottom: '8px' }}>
          <ProgressBar 
            percent={progressPercent} 
            showLabel={true} 
            height="12px"
            isRunning={group.status === 'running'}
            useThresholdColors={false}
            color="#17a2b8"
          />
        </div>

        <div className={styles.stats}>
          <span className={styles.statItem}>
            📊 {group.completed_rollouts || 0}/{group.num_rollouts || 0}
          </span>
          {group.reward_mean !== null && (
            <span className={styles.statItem}>
              🎯 {group.reward_mean.toFixed(3)}
            </span>
          )}
          <span className={styles.statItem}>
            ✓ {successCount}/{completedRollouts.length}
          </span>
        </div>
      </div>

      <div className={styles.rolloutsContainer}>
        <div className={styles.rolloutsList}>
          {rollouts.map((rollout) => (
            <div
              key={rollout.id}
              className={styles.rolloutCard}
              onClick={() => onSelectRollout(rollout.id)}
            >
              <div className={styles.rolloutHeader}>
                <span className={styles.rolloutEnv}>
                  {rollout.status === 'running' && <span className={styles.runningDot}>●</span>}
                  Env {rollout.env_index !== undefined ? rollout.env_index : '?'}
                </span>
                <div className={styles.rolloutBadges}>
                  {rollout.status === 'running' && (
                    <span className={`${styles.badge} ${styles.running}`}>Running</span>
                  )}
                  {rollout.status === 'completed' && rollout.task_success && (
                    <span className={`${styles.badge} ${styles.success}`}>✓ Success</span>
                  )}
                  {rollout.status === 'completed' && !rollout.task_success && (
                    <span className={`${styles.badge} ${styles.failed}`}>✗ Failed</span>
                  )}
                  {rollout.status === 'failed' && (
                    <span className={`${styles.badge} ${styles.error}`}>⚠ Error</span>
                  )}
                </div>
              </div>

              {/* Rollout-level mini progress bar (progress + success/failure coloring) */}
              <div className={styles.rolloutProgress}>
                <ProgressBar
                  percent={getRolloutPercent(rollout, maxTurns)}
                  showLabel={false}
                  height="7px"
                  isRunning={rollout.status === 'running'}
                  useThresholdColors={false}
                  color={getRolloutColor(rollout)}
                />
              </div>

              <div className={styles.rolloutStats}>
                <span className={styles.rolloutStat}>
                  🔄 {rollout.num_turns !== null ? rollout.num_turns : rollout.current_turn !== null ? rollout.current_turn : 0} turns
                </span>
                {rollout.reward !== null && (
                  <span className={styles.rolloutStat}>
                    🎯 Reward: {rollout.reward.toFixed(3)}
                  </span>
                )}
                {rollout.rollout_time !== null && rollout.rollout_time !== undefined && (
                  <span className={styles.rolloutStat}>
                    ⏱️ {rollout.rollout_time.toFixed(1)}s
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

