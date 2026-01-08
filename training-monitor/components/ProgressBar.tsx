'use client';

import React from 'react';
import styles from './ProgressBar.module.css';

interface ProgressBarProps {
  percent: number;
  showLabel?: boolean;
  height?: string;
  className?: string;
  isRunning?: boolean;  // New prop to control animation
}

export default function ProgressBar({ 
  percent, 
  showLabel = true, 
  height = '14px',  // Reduced default height
  className = '',
  isRunning = false  // Default no animation
}: ProgressBarProps) {
  const clampedPercent = Math.max(0, Math.min(100, percent));
  
  // Color coding based on progress
  const getColor = () => {
    if (clampedPercent >= 90) return '#28a745'; // Green
    if (clampedPercent >= 60) return '#17a2b8'; // Blue
    if (clampedPercent >= 30) return '#ffc107'; // Yellow
    return '#6c757d'; // Gray
  };

  return (
    <div className={`${styles.container} ${className}`}>
      <div className={styles.barContainer} style={{ height }}>
        <div
          className={`${styles.barFill} ${isRunning ? styles.animated : ''}`}
          style={{
            width: `${clampedPercent}%`,
            backgroundColor: getColor(),
          }}
        />
        {showLabel && (
          <div className={styles.label}>
            {clampedPercent.toFixed(1)}%
          </div>
        )}
      </div>
    </div>
  );
}

