'use client';

import { useState, useEffect } from 'react';
import styles from './TimelineList.module.css';
import ProgressBar from './ProgressBar';

interface TimelineItem {
  id: number;
  type: 'baseline' | 'step' | 'eval';
  display_name: string;
  status: string;
  progress_percent: number;
  avg_turn_time: number | null;
  estimated_total_time: number | null;
  estimated_remaining_time: number | null;
  created_at: string;
  start_time: string | null;
  end_time: string | null;
  step?: number;
}

interface TrainingParams {
  model_name: string | null;
  lora_rank: number | null;
  learning_rate: number | null;
  batch_size: number | null;
  group_size: number | null;
  groups_per_batch: number | null;
  max_tokens: number | null;
  temperature: number | null;
  kl_penalty_coef: number | null;
  num_substeps: number | null;
  max_turns: number | null;
  seed: number | null;
  box_type: string | null;
  renderer_name: string | null;
  wandb_project: string | null;
  wandb_name: string | null;
  log_path: string | null;
}

interface TimelineListProps {
  trainingId: number;
  selectedItem: { type: 'baseline' | 'step' | 'eval'; id: number } | null;
  onSelect: (type: 'baseline' | 'step' | 'eval', id: number) => void;
}

export default function TimelineList({
  trainingId,
  selectedItem,
  onSelect,
}: TimelineListProps) {
  const [items, setItems] = useState<TimelineItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [trainingParams, setTrainingParams] = useState<TrainingParams | null>(null);

  const fetchTimeline = async () => {
    try {
      const res = await fetch(`/api/trainings/${trainingId}/timeline`);
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(`HTTP error! status: ${res.status}, message: ${errorData.error || 'Unknown error'}`);
      }
      const data = await res.json();
      console.log(`Timeline data for training ${trainingId}:`, data);
      setItems(data.timeline || []);
    } catch (error) {
      console.error('Failed to fetch timeline:', error);
      setItems([]);
    } finally {
      setLoading(false);
    }
  };

  const fetchTrainingParams = async () => {
    try {
      const res = await fetch(`/api/trainings/${trainingId}`);
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      const data = await res.json();
      setTrainingParams(data.training || null);
    } catch (error) {
      console.error('Failed to fetch training params:', error);
    }
  };

  useEffect(() => {
    if (trainingId) {
      fetchTimeline();
      fetchTrainingParams();
      // Auto-refresh every 20 seconds
      const interval = setInterval(() => {
        fetchTimeline();
        fetchTrainingParams();
      }, 20000);
      return () => clearInterval(interval);
    }
  }, [trainingId]);

  if (loading) {
    return <div className={styles.loading}>Loading...</div>;
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <span>Timeline</span>
        <button 
          className={styles.refreshButton} 
          onClick={() => {
            fetchTimeline();
            fetchTrainingParams();
          }}
          title="刷新"
        >
          🔄
        </button>
      </div>
      {trainingParams && (
        <div className={styles.paramsPanel}>
          <div className={styles.paramsGrid}>
            {trainingParams.model_name && (
              <div className={styles.paramItem}>
                <span className={styles.paramLabel}>Model:</span>
                <span className={styles.paramValue}>{trainingParams.model_name}</span>
              </div>
            )}
            {trainingParams.lora_rank !== null && (
              <div className={styles.paramItem}>
                <span className={styles.paramLabel}>Rank:</span>
                <span className={styles.paramValue}>{trainingParams.lora_rank}</span>
              </div>
            )}
            {trainingParams.learning_rate !== null && (
              <div className={styles.paramItem}>
                <span className={styles.paramLabel}>Learning Rate:</span>
                <span className={styles.paramValue}>{trainingParams.learning_rate.toExponential(2)}</span>
              </div>
            )}
            {trainingParams.batch_size !== null && (
              <div className={styles.paramItem}>
                <span className={styles.paramLabel}>Batch Size:</span>
                <span className={styles.paramValue}>{trainingParams.batch_size}</span>
              </div>
            )}
            {trainingParams.group_size !== null && (
              <div className={styles.paramItem}>
                <span className={styles.paramLabel}>Group Size:</span>
                <span className={styles.paramValue}>{trainingParams.group_size}</span>
              </div>
            )}
            {trainingParams.groups_per_batch !== null && (
              <div className={styles.paramItem}>
                <span className={styles.paramLabel}>Groups/Batch:</span>
                <span className={styles.paramValue}>{trainingParams.groups_per_batch}</span>
              </div>
            )}
            {trainingParams.temperature !== null && (
              <div className={styles.paramItem}>
                <span className={styles.paramLabel}>Temperature:</span>
                <span className={styles.paramValue}>{trainingParams.temperature.toFixed(2)}</span>
              </div>
            )}
            {trainingParams.max_turns !== null && (
              <div className={styles.paramItem}>
                <span className={styles.paramLabel}>Max Turns:</span>
                <span className={styles.paramValue}>{trainingParams.max_turns}</span>
              </div>
            )}
          </div>
        </div>
      )}
      <div className={styles.list}>
        {items.length === 0 ? (
          <div className={styles.empty}>No timeline items yet</div>
        ) : (
          items.map((item) => {
            const isSelected =
              selectedItem?.type === item.type && selectedItem?.id === item.id;

            // Format times
            const startTime = item.start_time ? new Date(item.start_time) : null;
            const endTime = item.end_time ? new Date(item.end_time) : null;
            const now = new Date();

            // Format start time (short format for display)
            const startTimeText = startTime
              ? startTime.toLocaleTimeString('zh-CN', {
                  hour: '2-digit',
                  minute: '2-digit',
                })
              : null;

            // Calculate duration or show estimated remaining time
            let durationText = '';
            if (endTime && startTime) {
              // Completed: show actual duration
              const durationMs = endTime.getTime() - startTime.getTime();
              const durationSecs = Math.floor(durationMs / 1000);
              if (durationSecs < 60) {
                durationText = `${durationSecs}s`;
              } else if (durationSecs < 3600) {
                const mins = Math.floor(durationSecs / 60);
                const secs = durationSecs % 60;
                durationText = `${mins}m${secs > 0 ? secs + 's' : ''}`;
              } else {
                const hours = Math.floor(durationSecs / 3600);
                const mins = Math.floor((durationSecs % 3600) / 60);
                durationText = `${hours}h${mins > 0 ? mins + 'm' : ''}`;
              }
            } else if (startTime && item.status === 'running' && item.estimated_remaining_time) {
              // Running: show estimated remaining time
              const remainingSecs = Math.floor(item.estimated_remaining_time);
              if (remainingSecs < 60) {
                durationText = `ETA: ${remainingSecs}s`;
              } else if (remainingSecs < 3600) {
                const mins = Math.floor(remainingSecs / 60);
                durationText = `ETA: ${mins}m`;
              } else if (remainingSecs < 86400) {
                const hours = Math.floor(remainingSecs / 3600);
                const mins = Math.floor((remainingSecs % 3600) / 60);
                durationText = `ETA: ${hours}h${mins > 0 ? mins + 'm' : ''}`;
              } else {
                const days = Math.floor(remainingSecs / 86400);
                const hours = Math.floor((remainingSecs % 86400) / 3600);
                durationText = `ETA: ${days}d${hours > 0 ? hours + 'h' : ''}`;
              }
            } else if (startTime && item.status === 'running') {
              // Running but no estimate: show elapsed time
              const elapsedMs = now.getTime() - startTime.getTime();
              const elapsedSecs = Math.floor(elapsedMs / 1000);
              if (elapsedSecs < 60) {
                durationText = `${elapsedSecs}s`;
              } else if (elapsedSecs < 3600) {
                const mins = Math.floor(elapsedSecs / 60);
                durationText = `${mins}m`;
              } else {
                const hours = Math.floor(elapsedSecs / 3600);
                const mins = Math.floor((elapsedSecs % 3600) / 60);
                durationText = `${hours}h${mins > 0 ? mins + 'm' : ''}`;
              }
            }

            return (
              <div
                key={`${item.type}-${item.id}`}
                className={`${styles.item} ${
                  isSelected ? styles.selected : ''
                }`}
                onClick={() => onSelect(item.type, item.id)}
              >
                <div className={styles.itemContent}>
                  <div className={styles.name}>{item.display_name}</div>
                  <div className={styles.metadata}>
                    {startTimeText && (
                      <span className={styles.startTime} title="开始时间">
                        🕐 {startTimeText}
                      </span>
                    )}
                    {durationText && (
                      <span className={styles.duration} title={item.status === 'running' && item.estimated_remaining_time ? '预计剩余时间' : '耗时'}>
                        ⏱️ {durationText}
                      </span>
                    )}
                    {item.status === 'running' && item.avg_turn_time && (
                      <span className={styles.turnSpeed} title="平均每轮耗时">
                        ⚡ {item.avg_turn_time.toFixed(1)}s/turn
                      </span>
                    )}
                  </div>
                  <div className={styles.status}>
                    <span
                      className={`${styles.statusBadge} ${
                        styles[`status${item.status}`]
                      }`}
                    >
                      {item.status}
                    </span>
                  </div>
                  {item.progress_percent !== null && (
                    <div className={styles.progressContainer}>
                      <ProgressBar 
                        percent={item.progress_percent} 
                        showLabel={true} 
                        height="12px"
                        isRunning={item.status === 'running'}
                      />
                    </div>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

