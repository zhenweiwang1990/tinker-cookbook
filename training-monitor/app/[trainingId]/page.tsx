'use client';

import { useParams, useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import TrainingList from '@/components/TrainingList';
import TimelineList from '@/components/TimelineList';

export default function TrainingDetailPage() {
  const params = useParams();
  const router = useRouter();
  
  const trainingId = params?.trainingId ? parseInt(params.trainingId as string) : null;

  const [selectedTrainingId, setSelectedTrainingId] = useState<number | null>(trainingId);

  // Sync URL params to state
  useEffect(() => {
    if (trainingId !== selectedTrainingId) {
      setSelectedTrainingId(trainingId);
    }
  }, [trainingId]);

  // Auto-refresh disabled - use manual refresh button in components instead
  // useEffect(() => {
  //   const interval = setInterval(() => {
  //     setSelectedTrainingId((prev) => prev);
  //   }, 2000);
  //   return () => clearInterval(interval);
  // }, []);

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      <div style={{ width: '300px', borderRight: '1px solid #e0e0e0', overflow: 'auto' }}>
        <TrainingList
          selectedId={selectedTrainingId}
          onSelect={(id) => {
            setSelectedTrainingId(id);
            router.push(`/${id}`, { scroll: false });
          }}
        />
      </div>
      <div style={{ flex: 1, overflow: 'auto' }}>
        {selectedTrainingId ? (
          <TimelineList
            trainingId={selectedTrainingId}
            selectedItem={null}
            onSelect={(type, id) => {
              router.push(`/${selectedTrainingId}/${type}/${id}`, { scroll: false });
            }}
          />
        ) : (
          <div style={{ padding: '20px', color: '#666' }}>
            Select a training session
          </div>
        )}
      </div>
    </div>
  );
}

