'use client';

import { useParams, useRouter, useSearchParams } from 'next/navigation';
import { useEffect, useState, Suspense } from 'react';
import TrainingList from '@/components/TrainingList';
import TimelineList from '@/components/TimelineList';
import DetailPanel from '@/components/DetailPanel';

function TimelineDetailContent() {
  const params = useParams();
  const router = useRouter();
  const searchParams = useSearchParams();
  
  const trainingId = params?.trainingId ? parseInt(params.trainingId as string) : null;
  const type = params?.type as 'baseline' | 'step' | 'eval' | null;
  const id = params?.id ? parseInt(params.id as string) : null;
  const rolloutId = searchParams?.get('rollout') ? parseInt(searchParams.get('rollout')!) : null;
  const turnIndex = searchParams?.get('turn') ? parseInt(searchParams.get('turn')!) : null;

  const [selectedTrainingId, setSelectedTrainingId] = useState<number | null>(trainingId);
  const [selectedTimelineItem, setSelectedTimelineItem] = useState<{
    type: 'baseline' | 'step' | 'eval';
    id: number;
  } | null>(type && id ? { type, id } : null);
  const [selectedRolloutId, setSelectedRolloutId] = useState<number | null>(rolloutId);
  const [selectedTurnIndex, setSelectedTurnIndex] = useState<number | null>(turnIndex);

  // Update URL when selections change
  useEffect(() => {
    if (selectedTrainingId && selectedTimelineItem) {
      const path = `/${selectedTrainingId}/${selectedTimelineItem.type}/${selectedTimelineItem.id}`;
      const params = new URLSearchParams();
      if (selectedRolloutId) {
        params.set('rollout', selectedRolloutId.toString());
      }
      if (selectedTurnIndex !== null && selectedTurnIndex !== undefined) {
        params.set('turn', selectedTurnIndex.toString());
      }
      const query = params.toString() ? `?${params.toString()}` : '';
      router.replace(path + query, { scroll: false });
    }
  }, [selectedTrainingId, selectedTimelineItem, selectedRolloutId, selectedTurnIndex, router]);

  // Sync URL params to state (only when URL changes externally, not from our own state updates)
  useEffect(() => {
    if (trainingId !== selectedTrainingId) {
      setSelectedTrainingId(trainingId);
    }
    if (type && id && (!selectedTimelineItem || selectedTimelineItem.type !== type || selectedTimelineItem.id !== id)) {
      setSelectedTimelineItem({ type, id });
    }
    // Only sync rolloutId from URL if it's different and we're not in the middle of updating it
    if (rolloutId !== selectedRolloutId && rolloutId !== null) {
      setSelectedRolloutId(rolloutId);
      // Reset turn index when rollout changes
      setSelectedTurnIndex(null);
    }
    if (turnIndex !== selectedTurnIndex && turnIndex !== null) {
      setSelectedTurnIndex(turnIndex);
    }
  }, [trainingId, type, id, rolloutId, turnIndex]);

  // Auto-refresh disabled - use manual refresh button in components instead
  // useEffect(() => {
  //   const interval = setInterval(() => {
  //     // Trigger re-render by updating a dummy state
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
            setSelectedTimelineItem(null);
            setSelectedRolloutId(null);
            setSelectedTurnIndex(null);
            router.push(`/${id}`, { scroll: false });
          }}
        />
      </div>
      <div style={{ width: '300px', borderRight: '1px solid #e0e0e0', overflow: 'auto' }}>
        {selectedTrainingId ? (
          <TimelineList
            trainingId={selectedTrainingId}
            selectedItem={selectedTimelineItem}
            onSelect={(type, id) => {
              setSelectedTimelineItem({ type, id });
              setSelectedRolloutId(null);
              setSelectedTurnIndex(null);
              router.push(`/${selectedTrainingId}/${type}/${id}`, { scroll: false });
            }}
          />
        ) : (
          <div style={{ padding: '20px', color: '#666' }}>
            Select a training session
          </div>
        )}
      </div>
      <div style={{ flex: 1, overflow: 'auto' }}>
        {selectedTimelineItem ? (
          <DetailPanel
            type={selectedTimelineItem.type}
            id={selectedTimelineItem.id}
            selectedRolloutId={selectedRolloutId}
            selectedTurnIndex={selectedTurnIndex}
            onSelectRollout={(id) => {
              console.log('onSelectRollout called with id:', id);
              // Just update state - the useEffect will handle URL update
              setSelectedRolloutId(id);
              setSelectedTurnIndex(null);
            }}
            onTurnChange={(turnIndex) => {
              setSelectedTurnIndex(turnIndex);
              const params = new URLSearchParams();
              if (selectedRolloutId) {
                params.set('rollout', selectedRolloutId.toString());
              }
              if (turnIndex !== null && turnIndex !== undefined) {
                params.set('turn', turnIndex.toString());
              }
              const query = params.toString() ? `?${params.toString()}` : '';
              router.replace(`/${selectedTrainingId}/${selectedTimelineItem.type}/${selectedTimelineItem.id}${query}`, { scroll: false });
            }}
          />
        ) : (
          <div style={{ padding: '20px', color: '#666' }}>
            Select a timeline item to view details
          </div>
        )}
      </div>
    </div>
  );
}

export default function TimelineDetailPage() {
  return (
    <Suspense fallback={<div style={{ padding: '20px' }}>Loading...</div>}>
      <TimelineDetailContent />
    </Suspense>
  );
}

