'use client';

import { useState, useEffect } from 'react';
import RolloutList from './RolloutList';
import RolloutDetail from './RolloutDetail';
import styles from './DetailPanel.module.css';

interface DetailPanelProps {
  type: 'baseline' | 'step' | 'eval';
  id: number;
  selectedRolloutId: number | null;
  selectedTurnIndex: number | null;
  onSelectRollout: (id: number | null) => void;
  onTurnChange: (turnIndex: number | null) => void;
}

export default function DetailPanel({
  type,
  id,
  selectedRolloutId,
  selectedTurnIndex,
  onSelectRollout,
  onTurnChange,
}: DetailPanelProps) {
  const [item, setItem] = useState<any>(null);
  const [rollouts, setRollouts] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchDetails = async () => {
    try {
      const res = await fetch(`/api/timeline/${type}/${id}`);
      const data = await res.json();
      setItem(data.item);
      setRollouts(data.rollouts || []);
    } catch (error) {
      console.error('Failed to fetch details:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDetails();
    // Auto-refresh disabled - use manual refresh button instead
    // const interval = setInterval(fetchDetails, 2000); // Auto-refresh every 2 seconds
    // return () => clearInterval(interval);
  }, [type, id]);

  if (loading) {
    return <div className={styles.loading}>Loading...</div>;
  }

  if (!item) {
    return <div className={styles.error}>Item not found</div>;
  }

  // If a rollout is selected, show only RolloutDetail (full height)
  if (selectedRolloutId) {
    return (
      <RolloutDetail
        rolloutId={selectedRolloutId}
        selectedTurnIndex={selectedTurnIndex}
        onClose={() => onSelectRollout(null)}
        onTurnChange={onTurnChange}
      />
    );
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <h2 className={styles.title}>
            {type === 'baseline' && '📊 Baseline Evaluation'}
            {type === 'step' && `⚙️ Step ${item.step}`}
            {type === 'eval' && `📈 Evaluation @ Step ${item.step}`}
          </h2>
          <div className={styles.meta}>
            <span className={styles.status}>{item.status}</span>
            {item.progress_percent !== null && (
              <span className={styles.progress}>
                {item.progress_percent.toFixed(1)}%
              </span>
            )}
          </div>
        </div>
        <button 
          className={styles.refreshButton} 
          onClick={fetchDetails}
          title="刷新"
        >
          🔄
        </button>
      </div>

      <div className={styles.content}>
        <div className={styles.section}>
          <h3 className={styles.sectionTitle}>Summary</h3>
          <div className={styles.summaryGrid}>
            {type === 'step' && (
              <>
                {item.reward_mean !== null && (
                  <div className={styles.summaryItem}>
                    <span className={styles.label}>Reward Mean:</span>
                    <span className={styles.value}>{item.reward_mean.toFixed(4)}</span>
                  </div>
                )}
                {item.reward_std !== null && (
                  <div className={styles.summaryItem}>
                    <span className={styles.label}>Reward Std:</span>
                    <span className={styles.value}>{item.reward_std.toFixed(4)}</span>
                  </div>
                )}
                {item.loss !== null && (
                  <div className={styles.summaryItem}>
                    <span className={styles.label}>Loss:</span>
                    <span className={styles.value}>{item.loss.toFixed(4)}</span>
                  </div>
                )}
                {item.num_trajectories !== null && (
                  <div className={styles.summaryItem}>
                    <span className={styles.label}>Trajectories:</span>
                    <span className={styles.value}>{item.num_trajectories}</span>
                  </div>
                )}
              </>
            )}
            {(type === 'baseline' || type === 'eval') && (
              <>
                {item.success_rate !== null && (
                  <div className={styles.summaryItem}>
                    <span className={styles.label}>Success Rate:</span>
                    <span className={styles.value}>
                      {(item.success_rate * 100).toFixed(1)}%
                    </span>
                  </div>
                )}
                {item.avg_reward !== null && (
                  <div className={styles.summaryItem}>
                    <span className={styles.label}>Avg Reward:</span>
                    <span className={styles.value}>{item.avg_reward.toFixed(4)}</span>
                  </div>
                )}
                {item.avg_turns !== null && (
                  <div className={styles.summaryItem}>
                    <span className={styles.label}>Avg Turns:</span>
                    <span className={styles.value}>{item.avg_turns.toFixed(1)}</span>
                  </div>
                )}
                {item.successful_tasks !== null && item.total_tasks !== null && (
                  <div className={styles.summaryItem}>
                    <span className={styles.label}>Success:</span>
                    <span className={styles.value}>
                      {item.successful_tasks}/{item.total_tasks}
                    </span>
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        {(type === 'step' || type === 'eval' || type === 'baseline') && item.groups && item.groups.length > 0 && (
          <div className={styles.section}>
            <h3 className={styles.sectionTitle}>Groups ({item.groups.length})</h3>
            <div className={styles.groupsList}>
              {item.groups.map((group: any) => (
                <div key={group.id} className={styles.groupItem}>
                  <div className={styles.groupHeader}>
                    <span className={styles.groupNumber}>Group {group.group_num}</span>
                    <span className={styles.status}>{group.status}</span>
                    {group.progress_percent !== null && (
                      <span className={styles.progress}>
                        {group.progress_percent.toFixed(1)}%
                      </span>
                    )}
                  </div>
                  <div className={styles.groupStats}>
                    {group.completed_rollouts !== null && group.num_rollouts !== null && (
                      <span>Rollouts: {group.completed_rollouts}/{group.num_rollouts}</span>
                    )}
                    {group.reward_mean !== null && (
                      <span>Reward: {group.reward_mean.toFixed(4)}</span>
                    )}
                    {group.success_count !== null && (
                      <span>Success: {group.success_count}</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className={styles.section}>
          <h3 className={styles.sectionTitle}>Rollouts ({rollouts.length})</h3>
          <RolloutList
            rollouts={rollouts}
            onSelect={(id) => {
              console.log('RolloutList onSelect called with id:', id);
              console.log('onSelectRollout function:', onSelectRollout);
              onSelectRollout(id);
            }}
          />
        </div>
      </div>
    </div>
  );
}

