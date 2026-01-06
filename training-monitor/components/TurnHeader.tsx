/**
 * TurnHeader - Compact turn info bar with integrated timeline
 * Displays: Model Input button + Timeline + Reward + Tokens
 * Removes duplicated timing info (now in timeline tooltips)
 */

import React from 'react';
import TurnTimeline from './TurnTimeline';

interface TurnHeaderProps {
  turn: any; // Turn object
  action: any; // Action object
  stageTimings: any; // StageTimings object
  parseSuccess?: boolean;
  loadingModelInput?: boolean;
  onOpenModelInput: () => void;
}

export const TurnHeader: React.FC<TurnHeaderProps> = ({
  turn,
  action,
  stageTimings,
  parseSuccess,
  loadingModelInput,
  onOpenModelInput
}) => {
  return (
    <div style={{ 
      display: 'flex', 
      alignItems: 'center', 
      gap: '8px', 
      flexWrap: 'wrap',
      padding: '6px 10px',
      backgroundColor: '#f8f9fa',
      borderRadius: '4px',
      marginBottom: '10px',
      border: '1px solid #dee2e6'
    }}>
      {/* Model Input Button */}
      <button
        onClick={onOpenModelInput}
        disabled={loadingModelInput}
        style={{
          border: '1px solid #007bff',
          backgroundColor: loadingModelInput ? '#6c757d' : '#007bff',
          color: 'white',
          padding: '4px 10px',
          borderRadius: '3px',
          cursor: loadingModelInput ? 'not-allowed' : 'pointer',
          fontSize: '12px',
          fontWeight: 600,
          whiteSpace: 'nowrap',
          transition: 'background-color 0.2s'
        }}
        onMouseOver={(e) => {
          if (!loadingModelInput) {
            e.currentTarget.style.backgroundColor = '#0056b3';
          }
        }}
        onMouseOut={(e) => {
          if (!loadingModelInput) {
            e.currentTarget.style.backgroundColor = '#007bff';
          }
        }}
      >
        {loadingModelInput ? '⏳' : '📝 Input'}
      </button>
      
      {/* Compact Timeline */}
      <TurnTimeline 
        stageTimings={stageTimings}
        parseSuccess={parseSuccess}
        style={{ flex: 1, minWidth: '200px' }}
      />
      
      {/* Reward */}
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        gap: '4px',
        fontSize: '11px',
        color: '#666',
        whiteSpace: 'nowrap'
      }}>
        <span style={{ fontWeight: 600 }}>Reward:</span>
        <span style={{ 
          fontFamily: 'monospace',
          color: turn.reward && turn.reward > 0 ? '#28a745' : turn.reward && turn.reward < 0 ? '#dc3545' : '#666'
        }}>
          {turn.reward !== null && turn.reward !== undefined ? turn.reward.toFixed(4) : 'N/A'}
        </span>
      </div>
      
      {/* Token Stats */}
      {action && action.num_tokens !== null && action.num_tokens !== undefined && (
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: '4px',
          fontSize: '11px',
          color: '#666',
          whiteSpace: 'nowrap'
        }}>
          <span style={{ fontWeight: 600 }}>Tokens:</span>
          <span style={{ fontFamily: 'monospace' }}>{action.num_tokens}</span>
        </div>
      )}
    </div>
  );
};

export default TurnHeader;

