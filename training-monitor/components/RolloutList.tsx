'use client';

import { useState } from 'react';
import styles from './RolloutList.module.css';

interface Rollout {
  id: number;
  rollout_id: string;
  task_id: number;
  status: string;
  task_success: boolean;
  validation_passed: boolean;
  num_turns: number;
  reward: number;
  created_at: string;
}

interface RolloutListProps {
  rollouts: Rollout[];
  onSelect: (id: number) => void;
}

export default function RolloutList({ rollouts, onSelect }: RolloutListProps) {
  const [expandedIds, setExpandedIds] = useState<Set<number>>(new Set());

  const toggleExpand = (id: number) => {
    const newExpanded = new Set(expandedIds);
    if (newExpanded.has(id)) {
      newExpanded.delete(id);
    } else {
      newExpanded.add(id);
    }
    setExpandedIds(newExpanded);
  };

  if (rollouts.length === 0) {
    return <div className={styles.empty}>No rollouts yet</div>;
  }

  return (
    <div className={styles.container}>
      {rollouts.map((rollout) => {
        const isExpanded = expandedIds.has(rollout.id);
        return (
          <div key={rollout.id} className={styles.rollout}>
            <div
              className={styles.header}
              onClick={() => toggleExpand(rollout.id)}
            >
              <span className={styles.expandIcon}>
                {isExpanded ? '▼' : '▶'}
              </span>
              <span className={styles.rolloutId}>
                Rollout {rollout.rollout_id}
              </span>
              <div className={styles.badges}>
                {rollout.task_success && (
                  <span className={`${styles.badge} ${styles.success}`}>
                    ✓ Success
                  </span>
                )}
                {rollout.validation_passed && (
                  <span className={`${styles.badge} ${styles.validated}`}>
                    ✓ Validated
                  </span>
                )}
                {!rollout.task_success && (
                  <span className={`${styles.badge} ${styles.failed}`}>
                    ✗ Failed
                  </span>
                )}
              </div>
              <div className={styles.metrics}>
                <span className={styles.metric}>
                  Turns: {rollout.num_turns}
                </span>
                {rollout.reward !== null && (
                  <span className={styles.metric}>
                    Reward: {rollout.reward.toFixed(2)}
                  </span>
                )}
              </div>
            </div>
            {isExpanded && (
              <div className={styles.details}>
                <div className={styles.detailRow}>
                  <span className={styles.detailLabel}>Status:</span>
                  <span className={styles.detailValue}>{rollout.status}</span>
                </div>
                <div className={styles.detailRow}>
                  <span className={styles.detailLabel}>Task Success:</span>
                  <span className={styles.detailValue}>
                    {rollout.task_success ? 'Yes' : 'No'}
                  </span>
                </div>
                <div className={styles.detailRow}>
                  <span className={styles.detailLabel}>Validation:</span>
                  <span className={styles.detailValue}>
                    {rollout.validation_passed ? 'Passed' : 'Failed'}
                  </span>
                </div>
                <button
                  className={styles.viewButton}
                  onClick={() => onSelect(rollout.id)}
                >
                  View Full Details →
                </button>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

