/**
 * TurnTimeline - Compact timeline showing stage-by-stage timing
 * Displays a horizontal bar with color-coded segments representing each stage
 * Supports hover tooltips and status colors (gray/yellow/green/red)
 */

import React, { useState } from 'react';

interface StageTimings {
  screenshot_before?: number;
  model_input_prep?: number;
  model_inference?: number;
  action_parse?: number;
  action_coord?: number;
  action_exec?: number;
  screenshot_after?: number;
}

interface TurnTimelineProps {
  stageTimings: StageTimings;
  parseSuccess?: boolean;
  className?: string;
  style?: React.CSSProperties;
}

interface Stage {
  name: string;
  key: keyof StageTimings;
  time: number;
  baseColor: string;
  status: 'pending' | 'running' | 'completed' | 'error';
}

export const TurnTimeline: React.FC<TurnTimelineProps> = ({ 
  stageTimings, 
  parseSuccess,
  className,
  style 
}) => {
  const [hoveredStage, setHoveredStage] = useState<number | null>(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  // Define stages with precise times and intelligent status detection
  const stages: Stage[] = [
    { 
      name: 'Screenshot (Before)', 
      key: 'screenshot_before',
      time: stageTimings.screenshot_before || 0, 
      baseColor: '#007bff',
      status: stageTimings.screenshot_before && stageTimings.screenshot_before > 0 ? 'completed' : 'pending'
    },
    { 
      name: 'Model Input Prep', 
      key: 'model_input_prep',
      time: stageTimings.model_input_prep || 0, 
      baseColor: '#17a2b8',
      status: stageTimings.model_input_prep && stageTimings.model_input_prep > 0 ? 'completed' : 'pending'
    },
    { 
      name: 'Model Inference', 
      key: 'model_inference',
      time: stageTimings.model_inference || 0, 
      baseColor: '#28a745',
      status: stageTimings.model_inference && stageTimings.model_inference > 0 ? 'completed' : 'pending'
    },
    { 
      name: 'Action Parse', 
      key: 'action_parse',
      time: stageTimings.action_parse || 0, 
      baseColor: '#ffc107',
      status: parseSuccess === false ? 'error' : (stageTimings.action_parse && stageTimings.action_parse > 0 ? 'completed' : 'pending')
    },
    { 
      name: 'Action Coord', 
      key: 'action_coord',
      time: stageTimings.action_coord || 0, 
      baseColor: '#fd7e14',
      status: stageTimings.action_coord && stageTimings.action_coord > 0 ? 'completed' : 'pending'
    },
    { 
      name: 'Action Exec', 
      key: 'action_exec',
      time: stageTimings.action_exec || 0, 
      baseColor: '#e83e8c',
      status: stageTimings.action_exec && stageTimings.action_exec > 0 ? 'completed' : 'pending'
    },
    { 
      name: 'Screenshot (After)', 
      key: 'screenshot_after',
      time: stageTimings.screenshot_after || 0, 
      baseColor: '#6c757d',
      status: stageTimings.screenshot_after && stageTimings.screenshot_after > 0 ? 'completed' : 'pending'
    },
  ];
  
  const totalTime = stages.reduce((sum, stage) => sum + stage.time, 0);
  
  if (totalTime === 0) {
    return null; // Don't show timeline if no timing data
  }
  
  // Get color based on status (user requirement: use baseColor when completed for easy distinction)
  const getStatusColor = (stage: Stage): string => {
    switch (stage.status) {
      case 'completed':
        return stage.baseColor; // Use stage's own color when completed (for easy distinction)
      case 'running':
        return '#ffc107'; // Yellow for running
      case 'error':
        return '#dc3545'; // Red for error
      case 'pending':
      default:
        return '#dee2e6'; // Gray for pending
    }
  };
  
  return (
    <>
      {/* Total time label */}
      <div style={{ 
        fontSize: '10px', 
        color: '#666', 
        marginBottom: '2px',
        fontWeight: 600
      }}>
        Total: {totalTime.toFixed(3)}s
      </div>
      
      <div 
        className={className}
        style={{ 
          display: 'flex', 
          height: '16px',
          backgroundColor: '#e9ecef',
          borderRadius: '3px',
          overflow: 'visible',
          border: '1px solid #dee2e6',
          position: 'relative',
          ...style
        }}
      >
        {stages.map((stage, index) => {
          const widthPercent = totalTime > 0 ? (stage.time / totalTime) * 100 : 0;
          if (widthPercent < 0.5) return null; // Skip tiny segments
          
          const bgColor = getStatusColor(stage);
          const animationClass = stage.status === 'running' ? 'timeline-pulse' : '';
          
          return (
            <div
              key={index}
              className={animationClass}
              style={{
                width: `${widthPercent}%`,
                backgroundColor: bgColor,
                borderRight: index < stages.length - 1 ? '1px solid rgba(255,255,255,0.5)' : 'none',
                transition: 'all 0.3s ease',
                position: 'relative',
                cursor: 'pointer'
              }}
              onMouseEnter={(e) => {
                setHoveredStage(index);
                setMousePosition({ x: e.clientX, y: e.clientY });
              }}
              onMouseMove={(e) => {
                setMousePosition({ x: e.clientX, y: e.clientY });
              }}
              onMouseLeave={() => setHoveredStage(null)}
            />
          );
        })}
      </div>
      
      {/* Custom Tooltip */}
      {hoveredStage !== null && (
        <div
          style={{
            position: 'fixed',
            left: `${mousePosition.x + 10}px`,
            top: `${mousePosition.y + 10}px`,
            backgroundColor: 'rgba(0, 0, 0, 0.9)',
            color: 'white',
            padding: '6px 10px',
            borderRadius: '4px',
            fontSize: '12px',
            zIndex: 10000,
            pointerEvents: 'none',
            whiteSpace: 'nowrap',
            boxShadow: '0 2px 8px rgba(0,0,0,0.3)'
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: '2px' }}>
            {stages[hoveredStage].name}
          </div>
          <div style={{ fontSize: '11px', color: '#ccc' }}>
            {stages[hoveredStage].time > 0 
              ? `${stages[hoveredStage].time.toFixed(3)}s` 
              : 'N/A'}
          </div>
          <div style={{ fontSize: '10px', color: '#aaa', marginTop: '2px' }}>
            {stages[hoveredStage].status === 'completed' && '✓ Completed'}
            {stages[hoveredStage].status === 'running' && '⏳ Running...'}
            {stages[hoveredStage].status === 'error' && '✗ Error'}
            {stages[hoveredStage].status === 'pending' && '⋯ Pending'}
          </div>
        </div>
      )}
      
      {/* Breathing animation CSS */}
      <style jsx>{`
        @keyframes pulse {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.6;
          }
        }
        
        .timeline-pulse {
          animation: pulse 1.5s ease-in-out infinite;
        }
      `}</style>
    </>
  );
};

export default TurnTimeline;

