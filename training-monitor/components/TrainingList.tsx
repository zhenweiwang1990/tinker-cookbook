'use client';

import { useState, useEffect } from 'react';
import styles from './TrainingList.module.css';

interface Training {
  id: number;
  run_name: string;
  status: string;
  progress_percent: number;
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
                {training.progress_percent.toFixed(1)}%
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

