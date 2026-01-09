'use client';

import { useState, useEffect } from 'react';
import styles from './TrainingList.module.css';
import ProgressBar from './ProgressBar';

interface Training {
  id: number;
  run_name: string;
  status: string;
  progress_percent: number;
  estimated_remaining_time?: number | null;
  avg_turn_time?: number | null;
  current_step: number | null;
  total_steps: number | null;
  start_time: string | null;
  created_at: string;
}

interface TrainingListProps {
  selectedId: number | null;
  onSelect: (id: number) => void;
}

export default function TrainingList({ selectedId, onSelect }: TrainingListProps) {
  const [trainings, setTrainings] = useState<Training[]>([]);
  const [loading, setLoading] = useState(true);

  const formatEta = (seconds?: number | null) => {
    if (seconds === null || seconds === undefined) return '';
    const s = Math.floor(seconds);
    if (!Number.isFinite(s) || s <= 0) return '';
    if (s < 60) return `ETA: ${s}s`;
    if (s < 3600) return `ETA: ${Math.floor(s / 60)}m`;
    if (s < 86400) {
      const h = Math.floor(s / 3600);
      const m = Math.floor((s % 3600) / 60);
      return `ETA: ${h}h${m > 0 ? m + 'm' : ''}`;
    }
    const d = Math.floor(s / 86400);
    const h = Math.floor((s % 86400) / 3600);
    return `ETA: ${d}d${h > 0 ? h + 'h' : ''}`;
  };

  const fetchTrainings = async () => {
    try {
      const res = await fetch('/api/trainings');
      const data = await res.json();
      setTrainings(data.trainings || []);
    } catch (error) {
      console.error('Failed to fetch trainings:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTrainings();
    // Auto-refresh every 20 seconds
    const interval = setInterval(fetchTrainings, 20000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return <div className={styles.loading}>Loading...</div>;
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <span>Trainings</span>
        <button 
          className={styles.refreshButton} 
          onClick={fetchTrainings}
          title="刷新"
        >
          🔄
        </button>
      </div>
      <div className={styles.list}>
        {trainings.map((training) => (
          <div
            key={training.id}
            className={`${styles.item} ${
              selectedId === training.id ? styles.selected : ''
            }`}
            onClick={() => onSelect(training.id)}
            title={training.run_name}
          >
            <div className={styles.time}>
              {training.start_time
                ? new Date(training.start_time).toLocaleString('zh-CN', {
                    month: '2-digit',
                    day: '2-digit',
                    hour: '2-digit',
                    minute: '2-digit',
                  })
                : new Date(training.created_at).toLocaleString('zh-CN', {
                    month: '2-digit',
                    day: '2-digit',
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
            </div>
            <div className={styles.name}>
              {training.run_name.length > 30
                ? training.run_name.substring(0, 30) + '...'
                : training.run_name}
            </div>
            <div className={styles.status}>
              <span
                className={`${styles.statusBadge} ${
                  styles[`status${training.status}`]
                }`}
              >
                {training.status}
              </span>
            </div>
            {training.progress_percent !== null && (
              <div className={styles.progress}>
                <ProgressBar 
                  percent={training.progress_percent} 
                  showLabel={true} 
                  height="12px"
                  isRunning={training.status === 'running'}
                />
                {training.status === 'running' && training.estimated_remaining_time ? (
                  <div className={styles.eta}>
                    {formatEta(training.estimated_remaining_time)}
                  </div>
                ) : null}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

