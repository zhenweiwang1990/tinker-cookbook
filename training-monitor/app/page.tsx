'use client';

import { useRouter } from 'next/navigation';
import { useEffect } from 'react';
import TrainingList from '@/components/TrainingList';

export default function Home() {
  const router = useRouter();

  // Auto-refresh disabled - use manual refresh button in components instead
  // useEffect(() => {
  //   const interval = setInterval(() => {
  //     // Just keep the page alive
  //   }, 2000);
  //   return () => clearInterval(interval);
  // }, []);

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      <div style={{ width: '300px', borderRight: '1px solid #e0e0e0', overflow: 'auto' }}>
        <TrainingList
          selectedId={null}
          onSelect={(id) => {
            router.push(`/${id}`, { scroll: false });
          }}
        />
      </div>
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#666' }}>
        Select a training session to get started
      </div>
    </div>
  );
}

