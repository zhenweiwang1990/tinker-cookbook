'use client';

import { useState, useEffect } from 'react';
import styles from './TimelineList.module.css';

interface TimelineItem {
  id: number;
  type: 'baseline' | 'step' | 'eval';
  display_name: string;
  status: string;
  progress_percent: number;
  created_at: string;
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
      const interval = setInterval(fetchTimeline, 2000);
      return () => clearInterval(interval);
    }
  }, [trainingId]);

  if (loading) {
    return <div className={styles.loading}>Loading...</div>;
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>Timeline</div>
      {trainingParams && (
        <div className={styles.paramsPanel}>
          <div className={styles.paramsGrid}>
            {trainingParams.model_name && (
              <div className={styles.paramItem}>
                <span className={styles.paramLabel}>模型:</span>
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
                <span className={styles.paramLabel}>学习率:</span>
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
            {trainingParams.max_tokens !== null && (
              <div className={styles.paramItem}>
                <span className={styles.paramLabel}>Max Tokens:</span>
                <span className={styles.paramValue}>{trainingParams.max_tokens}</span>
              </div>
            )}
            {trainingParams.temperature !== null && (
              <div className={styles.paramItem}>
                <span className={styles.paramLabel}>Temperature:</span>
                <span className={styles.paramValue}>{trainingParams.temperature}</span>
              </div>
            )}
            {trainingParams.kl_penalty_coef !== null && (
              <div className={styles.paramItem}>
                <span className={styles.paramLabel}>KL Penalty:</span>
                <span className={styles.paramValue}>{trainingParams.kl_penalty_coef}</span>
              </div>
            )}
            {trainingParams.num_substeps !== null && (
              <div className={styles.paramItem}>
                <span className={styles.paramLabel}>Substeps:</span>
                <span className={styles.paramValue}>{trainingParams.num_substeps}</span>
              </div>
            )}
            {trainingParams.max_turns !== null && (
              <div className={styles.paramItem}>
                <span className={styles.paramLabel}>Max Turns:</span>
                <span className={styles.paramValue}>{trainingParams.max_turns}</span>
              </div>
            )}
            {trainingParams.seed !== null && (
              <div className={styles.paramItem}>
                <span className={styles.paramLabel}>Seed:</span>
                <span className={styles.paramValue}>{trainingParams.seed}</span>
              </div>
            )}
            {trainingParams.box_type && (
              <div className={styles.paramItem}>
                <span className={styles.paramLabel}>Box Type:</span>
                <span className={styles.paramValue}>{trainingParams.box_type}</span>
              </div>
            )}
            {trainingParams.renderer_name && (
              <div className={styles.paramItem}>
                <span className={styles.paramLabel}>Renderer:</span>
                <span className={styles.paramValue}>{trainingParams.renderer_name}</span>
              </div>
            )}
            {trainingParams.wandb_project && (
              <div className={styles.paramItem}>
                <span className={styles.paramLabel}>WandB Project:</span>
                <span className={styles.paramValue}>{trainingParams.wandb_project}</span>
              </div>
            )}
            {trainingParams.wandb_name && (
              <div className={styles.paramItem}>
                <span className={styles.paramLabel}>WandB Name:</span>
                <span className={styles.paramValue}>{trainingParams.wandb_name}</span>
              </div>
            )}
            {trainingParams.log_path && (
              <div className={styles.paramItem} style={{ gridColumn: '1 / -1' }}>
                <span className={styles.paramLabel}>训练数据源:</span>
                <span className={styles.paramValue} title={trainingParams.log_path}>
                  {trainingParams.log_path.length > 60
                    ? trainingParams.log_path.substring(0, 60) + '...'
                    : trainingParams.log_path}
                </span>
              </div>
            )}
          </div>
        </div>
      )}
      <div className={styles.list}>
        {items.length === 0 ? (
          <div className={styles.empty}>
            No timeline items yet
          </div>
        ) : (
          items.map((item) => {
            const isSelected =
              selectedItem?.type === item.type && selectedItem?.id === item.id;
            return (
              <div
                key={`${item.type}-${item.id}`}
                className={`${styles.item} ${
                  isSelected ? styles.selected : ''
                } ${styles[item.type]}`}
                onClick={() => onSelect(item.type, item.id)}
              >
                <div className={styles.typeIcon}>
                  {item.type === 'baseline' && '📊'}
                  {item.type === 'step' && '⚙️'}
                  {item.type === 'eval' && '📈'}
                </div>
                <div className={styles.content}>
                  <div className={styles.name}>{item.display_name}</div>
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
                    <div className={styles.progress}>
                      {item.progress_percent.toFixed(1)}%
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

