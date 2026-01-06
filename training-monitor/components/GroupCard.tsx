'use client';

import styles from './GroupCard.module.css';

interface Rollout {
  id: number;
  rollout_id: string;
  task_id: number;
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
}

interface GroupCardProps {
  group: any;
  rollouts: Rollout[];
  onSelectRollout: (id: number) => void;
}

export default function GroupCard({ group, rollouts, onSelectRollout }: GroupCardProps) {
  // Calculate success rate for this group
  const completedRollouts = rollouts.filter(r => r.status === 'completed');
  const successCount = completedRollouts.filter(r => r.task_success).length;
  const successRate = completedRollouts.length > 0 ? (successCount / completedRollouts.length) * 100 : 0;
  const isRunning = group.status === 'running' || rollouts.some(r => r.status === 'running');
  const isCompleted = group.status === 'completed';
  
  // Get success rate color
  const getSuccessRateClass = () => {
    if (successRate >= 80) return styles.successHigh;
    if (successRate >= 50) return styles.successMedium;
    return styles.successLow;
  };

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
          <div className={`${styles.successRate} ${getSuccessRateClass()}`}>
            <span className={styles.successRateValue}>{successRate.toFixed(0)}%</span>
            <span className={styles.successRateLabel}>success</span>
          </div>
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

