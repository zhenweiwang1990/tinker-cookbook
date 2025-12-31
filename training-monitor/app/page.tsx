'use client';

import { useState, useEffect } from 'react';
import TrainingList from '@/components/TrainingList';
import TimelineList from '@/components/TimelineList';
import DetailPanel from '@/components/DetailPanel';
import styles from './page.module.css';

export default function Home() {
  const [selectedTrainingId, setSelectedTrainingId] = useState<number | null>(null);
  const [selectedTimelineItem, setSelectedTimelineItem] = useState<{
    type: 'baseline' | 'step' | 'eval';
    id: number;
  } | null>(null);
  const [selectedRolloutId, setSelectedRolloutId] = useState<number | null>(null);

  // Auto-refresh every 2 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      // Trigger re-render by updating a dummy state
      setSelectedTrainingId((prev) => prev);
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className={styles.container}>
      <div className={styles.leftPanel}>
        <TrainingList
          selectedId={selectedTrainingId}
          onSelect={(id) => {
            setSelectedTrainingId(id);
            setSelectedTimelineItem(null);
            setSelectedRolloutId(null);
          }}
        />
      </div>
      <div className={styles.middlePanel}>
        {selectedTrainingId ? (
          <TimelineList
            trainingId={selectedTrainingId}
            selectedItem={selectedTimelineItem}
            onSelect={(type, id) => {
              setSelectedTimelineItem({ type, id });
              setSelectedRolloutId(null);
            }}
          />
        ) : (
          <div className={styles.placeholder}>
            Select a training session
          </div>
        )}
      </div>
      <div className={styles.rightPanel}>
        {selectedTimelineItem ? (
          <DetailPanel
            type={selectedTimelineItem.type}
            id={selectedTimelineItem.id}
            selectedRolloutId={selectedRolloutId}
            onSelectRollout={setSelectedRolloutId}
          />
        ) : (
          <div className={styles.placeholder}>
            Select a timeline item to view details
          </div>
        )}
      </div>
    </div>
  );
}

